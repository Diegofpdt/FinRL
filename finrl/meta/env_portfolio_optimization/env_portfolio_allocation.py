"""
PortfolioAllocationEnv — Portfolio allocation environment for IL+DAgger training.

Mirrors PortfolioOptimizationEnv structure (memory, terminal metrics, plots) while
adding configurable step_size, all 4 action types, and Dict observation space
compatible with YFDagger's flatten_obs utility.

Action types
------------
- weights      : Box(0, 1, n_stocks)  — target portfolio weight per stock
- continuous   : Box(-1, 1, n_stocks) — signed trade intensity (−1 full sell, +1 max buy)
- directions   : MultiDiscrete([3]*n)  — {−1 sell, 0 hold, +1 buy} (raw −1/0/+1)
- multidiscrete: MultiDiscrete([11]*n) — graded signal in [−5, +5] (raw integers)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "QuantStats not found. Install with: pip install quantstats --upgrade --no-cache-dir"
    )


class PortfolioAllocationEnv(gym.Env):
    """Portfolio allocation gymnasium environment for IL+DAgger training.

    Accepts a long-format DataFrame (same format as FinRL's PortfolioOptimizationEnv)
    and exposes a Dict observation space compatible with the flatten_obs utility.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format with columns [date, tic, close, <features...>].
    features : list[str]
        Feature column names to include in the observation (must exist in df).
    start_date : str
        Active window start date (YYYY-MM-DD). Can be updated via set_dates().
    end_date : str
        Active window end date (YYYY-MM-DD).
    step_size : int
        Number of trading days between portfolio rebalancing decisions. Default 91 (≈ quarter).
    action_type : str
        One of "weights", "continuous", "directions", "multidiscrete".
    initial_amount : float
        Starting cash in portfolio. Default 1_000_000.
    comission_fee_pct : float
        Transaction cost as a fraction (e.g. 0.002 = 0.2%). Default 0.002.
    reward_scaling : float
        Scalar multiplied by the log-return reward. Default 1.0.
    time_column : str
        Name of the date column in df. Default "date".
    tic_column : str
        Name of the ticker column in df. Default "tic".
    valuation_feature : str
        Column used for price valuation and trading. Default "close".
    cwd : str
        Directory for saving result plots. Default "./results/".
    new_gym_api : bool
        If True, step() returns (obs, reward, terminated, truncated, info).
        If False, returns (obs, reward, done, info). Default True.
    """

    metadata = {"render_modes": ["human"]}

    # Max shares that can be bought/sold in a single action (same as YFDagger K=1000)
    _K = 1_000

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        start_date: str,
        end_date: str,
        step_size: int = 91,
        action_type: str = "weights",
        initial_amount: float = 1_000_000,
        comission_fee_pct: float = 0.002,
        reward_scaling: float = 1.0,
        time_column: str = "date",
        tic_column: str = "tic",
        valuation_feature: str = "close",
        cwd: str = "./results/",
        new_gym_api: bool = True,
        render_mode=None,
    ):
        super().__init__()

        self._df               = df.copy()
        self._features         = list(features)
        self._start_date       = start_date
        self._end_date         = end_date
        self._step_size        = step_size
        self.action_type       = action_type
        self._initial_amount   = float(initial_amount)
        self._comission_fee_pct = comission_fee_pct
        self._reward_scaling   = reward_scaling
        self._time_column      = time_column
        self._tic_column       = tic_column
        self._valuation_feature = valuation_feature
        self._new_gym_api      = new_gym_api
        self.render_mode       = render_mode

        self._cwd          = Path(cwd)
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # Build full data_matrix from df (all dates), populate _tic_list, columns_map
        self._build_data_matrix()

        # Set active sorted_times to [start_date, end_date]
        self._update_sorted_times(start_date, end_date)

        # Define spaces
        self._build_spaces()

        # Initialise memory & state
        self._reset_memory()
        self._holdings      = np.zeros(self.portfolio_size, dtype=np.float64)
        self._balance       = self._initial_amount
        self._portfolio_value = self._initial_amount
        self._terminal      = False
        self._reward        = 0.0
        self._info          = {}

    # ================================================================ #
    #  Public API (mirrors PortfolioOptimizationEnv + YFDagger compat) #
    # ================================================================ #

    @property
    def step_size(self) -> int:
        return self._step_size

    @property
    def current_pos(self) -> int:
        """Index into data_matrix's time axis at the current step."""
        if self._time_index >= len(self._sorted_times):
            return self._time_to_matrix_idx.get(self._sorted_times[-1], 0)
        return self._time_to_matrix_idx[self._sorted_times[self._time_index]]

    def set_dates(self, start_date: str, end_date: str) -> None:
        """Change the active date window. Call reset() afterwards."""
        self._start_date = start_date
        self._end_date   = end_date
        self._update_sorted_times(start_date, end_date)
        print(f"Dates set to: {start_date} → {end_date}")

    def set_start(self, offset: int) -> None:
        """Advance _time_index by offset steps (for diverse DAgger rollouts)."""
        self._time_index = min(self._time_index + offset, len(self._sorted_times) - 1)

    def get_info(self) -> dict:
        """Return the current info dict without taking a step."""
        return self._build_info()

    def set_reward_type(self, reward_type: str) -> None:
        """Switch between 'log_return' (default) and 'sharpe' (terminal-only)."""
        if reward_type in ("log_return", "sharpe"):
            self._reward_type = reward_type

    def get_sb_env(self, env_number: int = 1):
        """Return a DummyVecEnv wrapping this env plus an initial observation."""
        e   = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs

    # ================================================================ #
    #  gymnasium.Env interface                                          #
    # ================================================================ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._time_index    = 0
        self._holdings      = np.zeros(self.portfolio_size, dtype=np.float64)
        self._balance       = self._initial_amount
        self._portfolio_value = self._initial_amount
        self._terminal      = False
        self._reward        = 0.0
        self._reset_memory()

        obs        = self._get_obs()
        self._info = self._build_info()

        if self._new_gym_api:
            return obs, self._info
        return obs

    def step(self, action):
        # Terminal check: not enough room to advance a full step
        self._terminal = (self._time_index + self._step_size) >= len(self._sorted_times)

        if self._terminal:
            return self._handle_terminal()

        # Current prices at this time index
        prices = self._get_prices(self._time_index)

        # Save initial portfolio value for reward computation
        pv_before = self._balance + float(np.nansum(self._holdings * prices))

        # Convert action to share-count deltas and execute trades
        share_deltas = self._action_fix(action, prices)
        self._execute_trades(share_deltas, prices)

        # Advance time and record a daily portfolio value for every day in the window.
        # Holdings are fixed after the trade, so daily_pv = balance + sum(h * day_prices).
        old_time_index    = self._time_index
        self._time_index += self._step_size
        close_idx         = self.columns_map[self._valuation_feature]

        for day in range(old_time_index + 1, self._time_index + 1):
            if day >= len(self._sorted_times):
                break
            m = self._time_to_matrix_idx[self._sorted_times[day]]
            day_prices = np.nan_to_num(self.data_matrix[:, m, close_idx], nan=0.0)
            self._daily_portfolio_values.append(
                self._balance + float(np.sum(self._holdings * day_prices))
            )
            self._daily_dates.append(self._sorted_times[day])

        # New prices after step_size days
        new_prices  = self._get_prices(self._time_index)
        pv_after    = self._balance + float(np.nansum(self._holdings * new_prices))
        self._portfolio_value = pv_after

        # Reward: log-return (PortfolioOptimizationEnv convention)
        rate_of_return   = pv_after / max(pv_before, 1e-10)
        portfolio_return = rate_of_return - 1
        portfolio_reward = math.log(max(rate_of_return, 1e-10)) * self._reward_scaling

        # Update memory (mirrors _asset_memory structure of PortfolioOptimizationEnv)
        self._asset_memory["initial"].append(pv_before)
        self._asset_memory["final"].append(pv_after)
        self._portfolio_return_memory.append(portfolio_return)
        self._portfolio_reward_memory.append(portfolio_reward)
        self._date_memory.append(self._sorted_times[self._time_index])
        self._actions_memory.append(np.array(action, dtype=np.float32).flatten()[:self.portfolio_size])

        self._reward = portfolio_reward
        obs          = self._get_obs()
        self._info   = self._build_info()

        if self._new_gym_api:
            return obs, self._reward, False, False, self._info
        return obs, self._reward, False, self._info

    def render(self, mode="human"):
        return self._get_obs()

    def close(self):
        pass

    # ================================================================ #
    #  Terminal handling (mirrors PortfolioOptimizationEnv)            #
    # ================================================================ #

    def _handle_terminal(self):
        final_values = self._asset_memory["final"]
        initial_pv   = final_values[0]
        final_pv     = self._portfolio_value

        # ---- Metrics ----
        n_days = max(len(self._daily_portfolio_values) - 1, 1)
        years  = n_days / 252.0
        annual_return = (final_pv / max(initial_pv, 1e-10)) ** (1 / max(years, 1e-10)) - 1

        # Daily series — correct Sharpe regardless of step_size (periods=252 = daily default)
        daily_pv_series = pd.Series(
            self._daily_portfolio_values,
            index=pd.DatetimeIndex(self._daily_dates),
            dtype=float,
        )
        daily_returns = daily_pv_series.pct_change().dropna()

        try:
            sharpe = float(qs.stats.sharpe(daily_returns, periods=252)) if len(daily_returns) > 1 else 0.0
            max_dd = float(qs.stats.max_drawdown(daily_pv_series))
        except Exception:
            sharpe, max_dd = 0.0, 0.0

        # ---- Console output (mirrors PortfolioOptimizationEnv) ----
        print("=================================")
        print(f"Initial portfolio value: {initial_pv:.2f}")
        print(f"Final portfolio value:   {final_pv:.2f}")
        print(f"Accumulative return:     {final_pv / max(initial_pv, 1e-10):.4f}")
        print(f"Annual return:           {annual_return:.4f}")
        print(f"Maximum DrawDown:        {max_dd:.4f}")
        print(f"Sharpe ratio:            {sharpe:.4f}")
        print("=================================")

        # ---- Plots (mirrors PortfolioOptimizationEnv) ----
        # Portfolio value: use daily series for a smooth equity curve
        plt.figure()
        plt.plot(daily_pv_series, "r")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Time")
        plt.ylabel("Portfolio value")
        plt.savefig(self._results_file / "portfolio_value.png")
        plt.close()

        plt.figure()
        plt.plot(self._portfolio_reward_memory, "r")
        plt.title("Reward Over Time")
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.savefig(self._results_file / "reward.png")
        plt.close()

        plt.figure()
        plt.plot(self._actions_memory)
        plt.title("Actions performed")
        plt.xlabel("Time")
        plt.ylabel("Action value")
        plt.savefig(self._results_file / "actions.png")
        plt.close()

        try:
            qs.plots.snapshot(
                daily_returns,
                show=False,
                savefig=str(self._results_file / "portfolio_summary.png"),
            )
        except Exception:
            pass

        self._terminal = True

        info = self._build_info()
        info["annual_return"]    = annual_return
        info["sharpe_ratio"]     = sharpe
        info["daily_pv_series"]  = daily_pv_series
        obs = self._get_obs()

        if self._new_gym_api:
            return obs, self._reward, True, False, info
        return obs, self._reward, True, info

    # ================================================================ #
    #  Memory                                                           #
    # ================================================================ #

    def _reset_memory(self):
        """Mirrors PortfolioOptimizationEnv._reset_memory()."""
        date_time = self._sorted_times[0] if self._sorted_times else None
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final":   [self._initial_amount],
        }
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        self._actions_memory = [np.zeros(self.portfolio_size, dtype=np.float32)]
        self._final_weights  = [np.zeros(self.portfolio_size, dtype=np.float32)]
        self._date_memory    = [date_time]

        # Daily series — one entry per trading day (used for correct Sharpe/drawdown)
        self._daily_portfolio_values = [self._initial_amount]
        self._daily_dates            = [date_time]

        if not hasattr(self, "_reward_type"):
            self._reward_type = "log_return"

    # ================================================================ #
    #  Data matrix construction                                        #
    # ================================================================ #

    def _build_data_matrix(self):
        """
        Pivot the long-format df into data_matrix of shape [N_stocks, T_all, N_features].

        All time steps in the df are included (not filtered to start/end date yet).
        _sorted_times is the active date window set separately by _update_sorted_times().
        """
        df = self._df.copy()
        df[self._time_column] = pd.to_datetime(df[self._time_column])

        # Preserve ticker insertion order (load_data() appends stocks in DJIA_STOCKS /
        # SP500_STOCKS declaration order). Using sorted() would produce alphabetical
        # ordering that differs from the old YFDagger env and makes action vectors
        # incomparable between the two systems.
        seen: dict = {}
        for tic in df[self._tic_column]:
            if tic not in seen:
                seen[tic] = None
        self._tic_list      = np.array(list(seen.keys()))
        self.portfolio_size = len(self._tic_list)

        df = df.sort_values([self._tic_column, self._time_column])

        all_times      = sorted(df[self._time_column].unique())
        self._all_times = all_times
        self._time_to_matrix_idx = {t: i for i, t in enumerate(all_times)}

        # All feature columns = every column except time + tic
        feat_cols = [c for c in df.columns
                     if c not in (self._time_column, self._tic_column)]
        self.columns_map = {col: i for i, col in enumerate(feat_cols)}

        T = len(all_times)
        N = self.portfolio_size
        F = len(feat_cols)

        self.data_matrix = np.full((N, T, F), np.nan, dtype=np.float64)

        for f_idx, col in enumerate(feat_cols):
            try:
                pivoted = df.pivot_table(
                    index=self._time_column,
                    columns=self._tic_column,
                    values=col,
                    aggfunc="first",
                )
                pivoted = pivoted.reindex(index=all_times, columns=self._tic_list)
                pivoted = pivoted.ffill()
                self.data_matrix[:, :, f_idx] = pivoted.values.T
            except Exception as e:
                print(f"Warning: could not pivot column '{col}': {e}")

    def _update_sorted_times(self, start_date: str | None, end_date: str | None):
        """Filter _sorted_times to [start_date, end_date]."""
        start = pd.to_datetime(start_date) if start_date else self._all_times[0]
        end   = pd.to_datetime(end_date)   if end_date   else self._all_times[-1]
        self._sorted_times = [t for t in self._all_times if start <= t <= end]
        if not self._sorted_times:
            raise ValueError(
                f"No data between {start_date} and {end_date}. "
                f"Available range: {self._all_times[0]} – {self._all_times[-1]}"
            )

    # ================================================================ #
    #  Observation & info helpers                                      #
    # ================================================================ #

    def _build_spaces(self):
        """Build gymnasium action and observation spaces for the current active window."""
        n = self.portfolio_size

        # Action space
        if self.action_type == "weights":
            self.action_space = spaces.Box(low=0.0,  high=1.0, shape=(n,), dtype=np.float32)
        elif self.action_type == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        elif self.action_type == "directions":
            self.action_space = spaces.MultiDiscrete(np.array([3] * n), dtype=np.int32)
        elif self.action_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete(np.array([11] * n), dtype=np.int32)
        else:
            raise ValueError(f"Unknown action_type: {self.action_type!r}")

        # Observation space: feature dict + portfolio weights (h) + cash ratio (b)
        obs_dict = {
            feat: spaces.Box(low=-4.0, high=4.0, shape=(n,), dtype=np.float32)
            for feat in self._features
            if feat in self.columns_map
        }
        obs_dict["h"] = spaces.Box(low=0.0, high=1.0, shape=(n,),  dtype=np.float32)
        obs_dict["b"] = spaces.Box(low=0.0, high=1.0, shape=(1,),  dtype=np.float32)
        self.observation_space = spaces.Dict(obs_dict)

    def _get_obs(self) -> dict:
        """Build the current Dict observation."""
        t_idx = min(self._time_index, len(self._sorted_times) - 1)
        m_idx = self._time_to_matrix_idx[self._sorted_times[t_idx]]

        obs = {}
        for feat in self._features:
            if feat in self.columns_map:
                vals = self.data_matrix[:, m_idx, self.columns_map[feat]]
                obs[feat] = np.nan_to_num(vals, nan=0.0).astype(np.float32)

        prices = self.data_matrix[:, m_idx, self.columns_map[self._valuation_feature]]
        prices = np.nan_to_num(prices, nan=0.0)
        pv     = self._balance + float(np.sum(self._holdings * prices))

        if pv > 0:
            weights = self._holdings * prices / pv
            obs["h"] = weights.astype(np.float32)
            obs["b"] = np.array([self._balance / pv], dtype=np.float32)
        else:
            obs["h"] = np.zeros(self.portfolio_size, dtype=np.float32)
            obs["b"] = np.array([1.0], dtype=np.float32)

        return obs

    def _build_info(self) -> dict:
        t_idx  = min(self._time_index, len(self._sorted_times) - 1)
        m_idx  = self._time_to_matrix_idx[self._sorted_times[t_idx]]
        prices = self.data_matrix[:, m_idx, self.columns_map[self._valuation_feature]]
        prices = np.nan_to_num(prices, nan=0.0)
        pv     = self._balance + float(np.sum(self._holdings * prices))
        date   = self._sorted_times[t_idx]

        return {
            # PortfolioOptimizationEnv keys
            "tics":            self._tic_list,
            "start_time":      self._sorted_times[0],
            "end_time":        date,
            "price_variation": prices,
            # Compatibility keys for train.py / evaluate.py
            "portfolio_value": pv,
            "holdings":        self._holdings.tolist(),
            "balance":         self._balance,
            "prices":          prices.tolist(),
            "date":            str(date)[:10],
            "annual_return":   0.0,
            "sharpe_ratio":    0.0,
        }

    # ================================================================ #
    #  Trading helpers                                                 #
    # ================================================================ #

    def _get_prices(self, time_index: int) -> np.ndarray:
        """Close prices at time_index (clamped to valid range)."""
        t_idx  = min(time_index, len(self._sorted_times) - 1)
        m_idx  = self._time_to_matrix_idx[self._sorted_times[t_idx]]
        prices = self.data_matrix[:, m_idx, self.columns_map[self._valuation_feature]]
        return np.nan_to_num(prices, nan=0.0)

    def _action_fix(self, action, prices: np.ndarray) -> np.ndarray:
        """
        Convert model action to integer share-count deltas.

        Ported from YFDagger._action_fix (env.py lines 341–362).
        Returns an array of shape (portfolio_size,) with integer deltas.
        """
        action = np.asarray(action, dtype=np.float64).flatten()[:self.portfolio_size]
        K      = self._K

        if self.action_type == "continuous":
            ret = np.array([round(K * x) for x in action], dtype=np.float64)
        elif self.action_type == "directions":
            ret = np.array([round(K * x) for x in action], dtype=np.float64)
        elif self.action_type == "multidiscrete":
            ret = np.array([round(K * (x / 5)) for x in action], dtype=np.float64)
        elif self.action_type == "weights":
            balance = self._balance
            pv      = balance + float(np.sum(self._holdings * prices))
            predicted = np.array(
                [pv * w / p if p > 0 else 0.0 for w, p in zip(action, prices)],
                dtype=np.float64,
            )
            diffs = predicted - self._holdings
            ret   = np.clip(diffs, -K, K)
        else:
            raise ValueError(f"Unknown action_type: {self.action_type!r}")

        ret = np.nan_to_num(ret, nan=0.0)
        return ret

    def _execute_trades(self, share_deltas: np.ndarray, prices: np.ndarray) -> None:
        """
        Execute sells then buys, applying comission_fee_pct.

        Ported from YFDagger._get_reward_and_state (env.py lines 380–411).
        Modifies self._holdings and self._balance in-place.
        """
        needed_to_buy: float = 0.0
        stocks_to_buy: dict  = {}
        sell_factor          = 1.0 - self._comission_fee_pct

        for i, delta in enumerate(share_deltas):
            p = prices[i]
            if p <= 0:
                continue
            delta = float(delta)
            if delta < 0:
                shares_to_sell       = min(-delta, self._holdings[i])
                self._holdings[i]   -= shares_to_sell
                self._balance       += shares_to_sell * p * sell_factor
            elif delta > 0:
                needed_to_buy     += delta * p
                stocks_to_buy[i]   = delta

        buy_factor = 1.0 + self._comission_fee_pct
        # Reduce buy orders proportionally if not enough cash
        reduce_n = max(1, self._K // 100)
        while self._balance < needed_to_buy * buy_factor and stocks_to_buy:
            for i in list(stocks_to_buy.keys()):
                if stocks_to_buy[i] <= 0:
                    del stocks_to_buy[i]
                    continue
                n = min(stocks_to_buy[i], reduce_n)
                stocks_to_buy[i] -= n
                needed_to_buy    -= prices[i] * n
                if stocks_to_buy[i] <= 0:
                    del stocks_to_buy[i]
            # Avoid infinite loop if no further reduction is possible
            if not stocks_to_buy:
                break

        self._balance -= needed_to_buy * buy_factor
        self._balance  = max(self._balance, 0.0)
        for i, shares in stocks_to_buy.items():
            self._holdings[i] += shares
