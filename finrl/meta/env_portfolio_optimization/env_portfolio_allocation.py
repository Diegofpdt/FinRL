"""
env_portfolio_allocation.py — Unified portfolio trading environment for IL and DRL agents.

This environment replaces both the original PortfolioAllocationEnv (Phase 2, Dict obs)
and StockTradingEnv. It provides a single Gymnasium-compatible environment used by
all training pipelines (train.py for IL/DAgger, train_ppo.py for DRL).

Key design:
- Flat 1D Box observation: [features × stocks] + [portfolio weights (h)] + [cash ratio (b)]
- Box(-1, 1) action space: each element is a fractional share-count request scaled by K
- Configurable step_size: quarterly (91) for IL/DAgger, daily (1) for DRL
- Configurable reward: 'log_return' for IL, 'absolute_return' for DRL
- Per-stock commission arrays (buy_cost_pct, sell_cost_pct)
- Trade execution: sells before buys, largest magnitude first, sequential budget checking
- Daily equity tracking independent of step_size (for Sharpe ratio and equity curve)
"""

from __future__ import annotations

from typing import List

import math
from pathlib import Path

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

import quantstats as qs

matplotlib.use("Agg") # Indicates matplotlib to generate plots directly on memory and not display directly on memory.


class PortfolioAllocationEnv(gym.Env):
    """
    Unified share-based portfolio trading environment for IL and DRL agents.

    The environment models a portfolio of N stocks. At each step the agent
    submits an action vector in [-1, 1]^N where each element, multiplied by
    K, gives the signed share-count delta (positive = buy, negative = sell).

    Observation space (flat 1D Box):
        [feature_0 for stock_0 … feature_0 for stock_N,
         feature_1 for stock_0 … …,
         …,
         portfolio weights h_0 … h_N,
         cash ratio b]

    Reward types (set via set_reward_type):
        'absolute_return' — dollar P&L: pv_after - pv_before (default for DRL)
        'log_return'      — log(pv_after / pv_before) × reward_scaling (default for IL)

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns [time_column, tic_column, features…].
    features : list[str]
        Feature column names to include in the observation (e.g. COLS_DJIA).
    start_date, end_date : str
        Active trading window (ISO format, e.g. '2009-01-01').
    buy_cost_pct, sell_cost_pct : list[float]
        Per-stock transaction cost fractions (length = number of stocks).
    K : int
        Max shares traded per action element (scales [-1, 1] → share deltas).
    step_size : int
        Number of trading days per environment step (91 for IL, 1 for DRL).
    initial_amount : float
        Starting cash balance.
    reward_scaling : float
        Multiplier applied to log_return rewards (ignored for absolute_return).
    cwd : str
        Working directory for saving plots and results.
    new_gym_api : bool
        If True, step() and reset() return the Gymnasium (5-tuple) API.
    model_name : str
        Sub-directory name under cwd/results/ for saving plots.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        start_date: str,
        end_date: str,
        buy_cost_pct: list[float], # STOCKTRADING
        sell_cost_pct: list[float], # STOCKTRADING
        K: int = 1_000, # max number of shares to be traded
        step_size: int = 91,
        initial_amount: float = 1_000_000,
        reward_scaling: float = 1.0,
        turbulence_threshold=None,
        turbulence_col: str | None = None,
        tradeable_col: str | None = None,
        time_column: str = "date",
        tic_column: str = "tic",
        valuation_feature: str = "close",
        cwd: str = "./results/",
        new_gym_api: bool = True,
        render_mode=None,
        model_name: str = "", #STOCKTRADING
    ):
        super().__init__()

        self._df = df.copy()
        self._features = list(features)
        self._start_date = start_date
        self._end_date = end_date
        self._buy_cost_pct = buy_cost_pct
        self._sell_cost_pct = sell_cost_pct
        self._K = K
        self._step_size = step_size
        self._initial_amount = initial_amount
        self._reward_scaling = reward_scaling
        self._turbulence_threshold = turbulence_threshold
        self._turbulence_col = turbulence_col
        self._tradeable_col = tradeable_col
        self._time_column = time_column
        self._tic_column = tic_column
        self._valuation_feature = valuation_feature
        self._cwd = cwd
        self._new_gym_api = new_gym_api
        self.render_mode = render_mode
        self._model_name = model_name

        self._cwd = Path(cwd)
        self._results_file = self._cwd / "results" / self._model_name
        self._results_file.mkdir(parents=True, exist_ok=True)

        self._build_data_matrix() # creates self.data_matrix[N(stock), T(date), F(feature)]
        # There it creates
        # self._tic_list
        # self.portfolio_size
        # self._all_times
        # self._time_to_matrix_idx
        # self.columns_map
        # self.data_matrix

        self._update_sorted_times(start_date, end_date) # Check availability of dates
        # self._sorted_times times valid between start_date and end_date

        self._build_spaces()

        self._reset_memory()
        # self._asset_memory: stores the pv
        # self._portfolio_return_memory : stores returns
        # self._portfolio_reward_memory  : stores rewards
        # self._actions_memory : stores actions
        # self._final_weights  : stores weights
        # self._date_memory : stores dates
        # self._daily_portfolio_values : stores pv
        # self._reward_type: reward type

        # Trading state
        self._holdings = np.zeros(self.portfolio_size, dtype=np.float64)
        self._balance = self._initial_amount
        self._portfolio_value = self._initial_amount
        self._terminal = False
        self._reward = 0
        self._turbulence = 0
        self._cost = 0
        self._trades = 0
        self._info = {}

        self._seed()

    # ================================================================ #
    #  Properties                                                       #
    # ================================================================ #

    @property
    def step_size(self) -> int:
        """Number of trading days per environment step."""
        return self._step_size

    @property
    def current_pos(self) -> int:
        """Matrix index corresponding to the current time step."""
        if self._time_index >= len(self._sorted_times):
            return self._time_to_matrix_idx.get(self._sorted_times[-1], 0)
        return self._time_to_matrix_idx.get(self._sorted_times[self._time_index])

    # ================================================================ #
    #  Configuration helpers                                            #
    # ================================================================ #

    def set_dates(self, start_date: str, end_date: str) -> None:
        """Change the active date window. Call reset() afterwards."""
        self._start_date = start_date
        self._end_date = end_date
        self._update_sorted_times(start_date, end_date)
        print(f"Dates set to: {start_date} -> {end_date}")

    def set_start(self, offset: int) -> None:
        """Advance _time_index by offset steps (for diverse DAgger rollouts)."""
        self._time_index = min(self._time_index + offset, len(self._sorted_times) - 1)

    def get_info(self) -> dict:
        """Return the current info dict without taking a step."""
        return self._build_info()

    def set_reward_type(self, reward_type: str) -> None:
        """
        Switch the reward signal between 'absolute_return' and 'log_return'.

        'absolute_return' — dollar P&L per step (recommended for DRL with reward_scaling=1e-4)
        'log_return'      — log(pv_after/pv_before) × reward_scaling (recommended for IL)
        """
        if reward_type in ("log_return", "absolute_return"):
            self._reward_type = reward_type

    def get_sb_env(self, env_number: int = 1):
        """Wrap self in a DummyVecEnv and return (vec_env, initial_obs)."""
        e = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs

    # ================================================================ #
    #  gymnasium.Env interface                                          #
    # ================================================================ #

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the start of the date window.

        Returns
        -------
        obs : np.ndarray of shape (state_dimension_number,)
        info : dict  (only when new_gym_api=True)
        """
        super().reset(seed=seed)

        self._time_index = 0
        self._holdings = np.zeros(self.portfolio_size, dtype=np.float64)
        self._balance = self._initial_amount
        self._portfolio_value = self._initial_amount
        self._terminal = False
        self._reward = 0
        self._turbulence = 0
        self._cost = 0
        self._trades = 0

        self._reset_memory()

        obs = self._get_obs()
        self._info = self._build_info()

        if self._new_gym_api:
            return obs, self._info
        return obs

    def step(self, action):
        """
        Execute one environment step.

        Parameters
        ----------
        action : np.ndarray of shape (portfolio_size,), values in [-1, 1]
            Each element scaled by K gives the signed share-count delta.

        Returns (new_gym_api=True)
        -------
        obs, reward, terminated, truncated, info
        """
        last_idx = len(self._sorted_times) - 1

        if self._time_index >= last_idx:
            self._terminal = True
            return self._handle_terminal()

        # Current prices before trading
        prices = self._get_prices(self._time_index)

        pv_before = self._balance + float(np.nansum(self._holdings * prices))

        # convert action to share-count
        share_deltas = action * self._K
        share_deltas = share_deltas.astype(int) # convert action into integer
        share_deltas = np.nan_to_num(share_deltas, nan=0)

        self._execute_trades(share_deltas, prices)

        # Advance time — use a partial step if a full step_size doesn't fit.
        # This matches the old env which processes every day up to end_date.
        old_time_index    = self._time_index
        self._time_index  = min(self._time_index + self._step_size, last_idx)

        # Record a daily portfolio value for every day in the window.
        # Holdings are fixed after the trade, so daily_pv = balance + sum(h * day_prices).
        for day in range(old_time_index + 1, self._time_index + 1):
            day_prices = self._get_prices(day)
            self._daily_portfolio_values.append(
                self._balance + float(np.sum(self._holdings * day_prices))
            )
            self._daily_dates.append(self._sorted_times[day])

        # Update turbulence indicator for new time index
        if self._turbulence_threshold is not None and self._turbulence_col is not None:
            t_idx = min(self._time_index, len(self._sorted_times) - 1)
            m_idx = self._time_to_matrix_idx[self._sorted_times[t_idx]]
            self._turbulence = self.data_matrix[0, m_idx, self.columns_map[self._turbulence_col]]

        # new prices afer advancing
        new_prices = self._get_prices(self._time_index)
        pv_after = self._balance + float(np.sum(self._holdings * new_prices))
        self._portfolio_value = pv_after

        # reward
        self._reward = self._reward_calculator(pv_before, pv_after)

        # Update step-level memory
        self._asset_memory["initial"].append(pv_before)
        self._asset_memory["final"].append(pv_after)
        self._portfolio_reward_memory.append(self._reward)
        self._actions_memory.append(np.array(action, dtype=np.float32).flatten()[:self.portfolio_size])
        self._step_dates.append(self._sorted_times[self._time_index])

        self._terminal = self._time_index >= last_idx
        if self._terminal:
            return self._handle_terminal()

        obs = self._get_obs()
        self._info = self._build_info()

        if self._new_gym_api:
            return obs, self._reward, False, False, self._info
        return obs, self._reward, False, self._info

    def _reward_calculator(self, pv_before: float, pv_after: float) -> float:
        """
        Compute the scalar reward for a step.

        'absolute_return': raw dollar P&L (pv_after - pv_before).
            Typically scaled by reward_scaling=1e-4 for DRL training stability.
        'log_return': log(pv_after / pv_before) × reward_scaling.
            Numerically stable; suitable for IL training.
        """
        if self._reward_type == "absolute_return":
            return pv_after - pv_before
        elif self._reward_type == "log_return":
            rate = pv_after / max(pv_before, 1e-10)
            return math.log(max(rate, 1e-10)) * self._reward_scaling

    def render(self, mode="human"):
        return self._get_obs()

    def close(self):
        pass

    # ================================================================ #
    #  Terminal handling                                                #
    # ================================================================ #

    def _handle_terminal(self):
        """
        Compute final metrics and save plots when the episode ends.

        Computes annual return, Sharpe ratio, and max drawdown from the
        daily equity series (independent of step_size). Saves four plots
        to self._results_file: portfolio_value.png, reward.png,
        actions.png, portfolio_summary.png.

        Returns the standard Gymnasium 5-tuple with info containing:
            'annual_return', 'sharpe_ratio', 'daily_pv_series'
        """
        final_values = self._asset_memory["final"]
        initial_pv = final_values[0]
        final_pv = self._portfolio_value

        n_days = max(len(self._daily_portfolio_values) - 1, 1)
        years = n_days / 252
        annual_return = (final_pv / max(initial_pv, 1e-10)) ** (1 / max(years, 1e-10)) - 1

        daily_pv_series = pd.Series(
            self._daily_portfolio_values,
            index=pd.DatetimeIndex(self._daily_dates),
            dtype=float,
        )

        daily_returns = daily_pv_series.pct_change().dropna()

        try:
            sharpe = float(qs.stats.sharpe(daily_returns, periods=252)) if len(daily_returns) > 1 else 0
            max_dd = float(qs.stats.max_drawdown(daily_returns))
        except Exception:
            sharpe, max_dd = 0, 0

        # Console output (mirrors PortfolioOptimizationEnv convention)
        print("=================================")
        print(f"Initial portfolio value: {initial_pv:.2f}")
        print(f"Final portfolio value:   {final_pv:.2f}")
        print(f"Accumulative return:     {final_pv / max(initial_pv, 1e-10):.4f}")
        print(f"Annual return:           {annual_return:.4f}")
        print(f"Maximum DrawDown:        {max_dd:.4f}")
        print(f"Sharpe ratio:            {sharpe:.4f}")
        print(f"Total cost of transactions: {self._cost:.4f}")
        print(f"Number of trades: {self._trades:.4f}")
        print("=================================")

        # Save plots
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

        info = self._build_info()
        info["annual_return"] = annual_return
        info["sharpe_ratio"] = sharpe
        info["daily_pv_series"] = daily_pv_series
        obs = self._get_obs()

        if self._new_gym_api:
            return obs, self._reward, True, False, info
        return obs, self._reward, True, info

    # ================================================================ #
    #  Data matrix construction                                         #
    # ================================================================ #

    def _build_data_matrix(self):
        """
        Pivot the long-format df into data_matrix of shape [N_stocks, T_all, N_features].

        All time steps in the df are included (not filtered to start/end date yet).
        _sorted_times is the active date window set separately by _update_sorted_times().

        Sets attributes:
            self._tic_list         — ordered array of ticker symbols
            self.portfolio_size    — number of stocks
            self._all_times        — sorted list of all dates in df
            self._time_to_matrix_idx — {date: column_index} mapping
            self.columns_map       — {feature_name: row_index} mapping
            self.data_matrix       — float64 array [N, T, F]
        """
        df = self._df.copy()
        df[self._time_column] = pd.to_datetime(df[self._time_column])

        # Preserve insertion order of tickers (matches load_data ordering)
        seen: dict = {}
        for tic in df[self._tic_column]:
            if tic not in seen:
                seen[tic] = None
        self._tic_list = np.array(list(seen.keys()))
        self.portfolio_size = len(self._tic_list)

        df = df.sort_values([self._tic_column, self._time_column])

        all_times = sorted(df[self._time_column].unique())
        # maps to interact with the data matrix
        self._all_times = all_times
        self._time_to_matrix_idx = {t: i for i, t in enumerate(all_times)} # Each date has it index

        feat_cols = [c for c in df.columns if c not in (self._time_column, self._tic_column)]
        self.columns_map = {col: i for i, col in enumerate(feat_cols)} # Each columns has a index

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
        """
        Filter _sorted_times to the inclusive [start_date, end_date] window.

        Raises ValueError if no data falls in the requested range.
        """
        start = pd.to_datetime(start_date) if start_date else self._all_times[0]
        end = pd.to_datetime(end_date) if end_date else self._all_times[-1]
        self._sorted_times = [t for t in self._all_times if start <= t <= end]
        if not self._sorted_times:
            raise ValueError(
                f"No data between {start_date} and {end_date}. "
                f"Available range: {self._all_times[0]} - {self._all_times[-1]}"
            )

    # ================================================================ #
    #  Observation, info, and memory helpers                           #
    # ================================================================ #

    def _build_spaces(self):
        """
        Define action and observation spaces.

        Action space: Box(-1, 1)^N — one element per stock.
        Observation space: Box(-inf, inf)^D where D = N × |features| + N + 1
            Layout: [feature_vals…, portfolio_weights (h), cash_ratio (b)]
        """
        n = self.portfolio_size

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

        feat_count = len([f for f in self._features if f in self.columns_map])
        self.state_dimension_number = n * feat_count + n + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dimension_number,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        """
        Build the flat 1D observation vector at the current time step.

        Layout:
            [feat_0 × N stocks, feat_1 × N stocks, …, weights h × N, cash_ratio b]
        All NaN values are replaced with 0. Weights and cash_ratio are
        normalized by current portfolio value; falls back to 0-weights + b=1
        when portfolio value is zero.
        """
        t_idx = min(self._time_index, len(self._sorted_times) - 1)
        m_idx = self._time_to_matrix_idx[self._sorted_times[t_idx]]

        obs = np.zeros(self.state_dimension_number, dtype=np.float32)
        offset = 0
        for feat in self._features:
            if feat in self.columns_map:
                vals = self.data_matrix[:, m_idx, self.columns_map[feat]]
                obs[offset:offset + self.portfolio_size] = np.nan_to_num(vals, nan=0).astype(np.float32)
                offset += self.portfolio_size

        prices = self.data_matrix[:, m_idx, self.columns_map[self._valuation_feature]]
        prices = np.nan_to_num(prices, nan=0)
        pv = self._balance + float(np.sum(self._holdings * prices))

        if pv > 0:
            weights = self._holdings * prices / pv
            obs[offset:offset + self.portfolio_size] = weights.astype(np.float32) # holdings
            obs[-1] = np.float32(self._balance / pv) # cash ratio
        else:
            obs[offset:offset + self.portfolio_size] = np.zeros(self.portfolio_size, dtype=np.float32)
            obs[-1] = np.float32(1.0)

        return obs

    def _build_info(self) -> dict:
        """
        Build the info dict for the current step.

        Keys (compatible with train.py and evaluate.py):
            tics, start_time, end_time, price_variation,
            portfolio_value, holdings, balance, prices, date,
            annual_return (0.0 placeholder), sharpe_ratio (0.0 placeholder)
        Terminal info adds: annual_return, sharpe_ratio, daily_pv_series.
        """
        t_idx = min(self._time_index, len(self._sorted_times) - 1)
        m_idx = self._time_to_matrix_idx[self._sorted_times[t_idx]]
        prices = self.data_matrix[:, m_idx, self.columns_map[self._valuation_feature]]
        prices = np.nan_to_num(prices, nan=0)
        pv = self._balance + float(np.sum(self._holdings * prices))
        date = self._sorted_times[t_idx]

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

    def _reset_memory(self):
        """
        Reset all episode memory buffers.

        Step-level buffers (one entry per rebalancing decision):
            _asset_memory, _portfolio_reward_memory, _actions_memory, _step_dates

        Daily buffers (one entry per trading day in the window):
            _daily_portfolio_values, _daily_dates
            Used for Sharpe ratio, max drawdown, and equity curve — independent of step_size.

        Also initialises _reward_type to 'absolute_return' on first call
        (subsequent calls preserve the type set via set_reward_type).
        """
        date_time = self._sorted_times[0] if self._sorted_times else None
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final":   [self._initial_amount],
        }
        self._portfolio_reward_memory = [0]
        self._actions_memory = [np.zeros(self.portfolio_size, dtype=np.float32)]
        self._step_dates = [date_time]

        self._daily_portfolio_values = [self._initial_amount]
        self._daily_dates = [date_time]

        if not hasattr(self, "_reward_type"):
            self._reward_type = "absolute_return"

    # ================================================================ #
    #  Trading mechanics                                                #
    # ================================================================ #

    def _get_prices(self, time_index: int) -> np.ndarray:
        """Close prices at time_index, clamped to valid range, NaN → 0."""
        t_idx = min(time_index, len(self._sorted_times) - 1)
        m_idx = self._time_to_matrix_idx[self._sorted_times[t_idx]]
        prices = self.data_matrix[:, m_idx, self.columns_map[self._valuation_feature]]
        return np.nan_to_num(prices, nan=0.0)

    def _execute_trades(self, share_deltas: np.ndarray, prices: np.ndarray) -> None:
        """
        Execute sells then buys with priority ordering and per-stock budget checks.

        Order: largest magnitude first (most negative sells, most positive buys).
        Turbulence override: panic-sell all holdings, block all buys.
        """
        # Turbulence override: panic-sell everything, block buys
        if self._turbulence_threshold is not None and self._turbulence >= self._turbulence_threshold:
            for i in range(self.portfolio_size):
                if prices[i] > 0 and self._holdings[i] > 0:
                    self._sell_stock(i, self._holdings[i], prices[i])
            return

        argsort = np.argsort(share_deltas)
        sell_indices = argsort[:np.sum(share_deltas < 0)]    # most negative first
        buy_indices  = argsort[::-1][:np.sum(share_deltas > 0)]  # most positive first

        for index in sell_indices:
            if prices[index] <= 0 or not self._is_tradeable(index):
                continue
            self._sell_stock(index, abs(share_deltas[index]), prices[index])

        for index in buy_indices:
            if prices[index] <= 0 or not self._is_tradeable(index):
                continue
            self._buy_stock(index, share_deltas[index], prices[index])

    def _sell_stock(self, index: int, amount: int, price: float) -> None:
        """
        Sell up to `amount` shares of stock `index` at `price`.

        Proceeds are net of sell_cost_pct[index]. Cannot sell more than held.
        """
        if self._holdings[index] <= 0:
            return

        sell_shares = min(amount, self._holdings[index])
        sell_proceeds = price * sell_shares * (1 - self._sell_cost_pct[index]) # minus the specified stock cost of selling the share

        self._holdings[index] -= sell_shares
        self._balance += sell_proceeds
        self._cost += price * sell_shares * self._sell_cost_pct[index]
        self._trades += 1

    def _buy_stock(self, index: int, amount: int, price: float) -> None:
        """
        Buy up to `amount` shares of stock `index` at `price`.

        Capped by available cash (including buy_cost_pct[index] per share).
        No-op if affordable shares == 0.
        """
        affordable = int(self._balance // (price * (1 + self._buy_cost_pct[index]))) # if i use all my cash, how many shares can i buy
        buy_shares = min(amount, affordable)

        if buy_shares <= 0:
            return

        buy_cost = price * buy_shares * (1 + self._buy_cost_pct[index])

        self._holdings[index] += buy_shares
        self._balance -= buy_cost
        self._cost += price * buy_shares * self._buy_cost_pct[index]
        self._trades += 1

    def _is_tradeable(self, stock_index: int) -> bool:
        """
        Return True if the stock is tradeable at the current time step.

        If tradeable_col is None (default), all stocks are always tradeable.
        A value of 1.0 in the tradeable column indicates the stock is halted.
        """
        if self._tradeable_col is None:
            return True
        t_idx = min(self._time_index, len(self._sorted_times) - 1)
        m_idx = self._time_to_matrix_idx[self._sorted_times[t_idx]]
        flag = self.data_matrix[stock_index, m_idx, self.columns_map[self._tradeable_col]]
        return flag != 1.0

    def _seed(self, seed=None):
        """Initialise the numpy random state (Gymnasium seeding convention)."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
