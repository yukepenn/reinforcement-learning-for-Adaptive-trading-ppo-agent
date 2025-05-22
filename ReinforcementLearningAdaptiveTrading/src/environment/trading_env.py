import gymnasium as gym # Changed from gym to gymnasium
from gymnasium import spaces # Changed from gym to gymnasium
import numpy as np
import pandas as pd
from collections import deque # For recent_returns with a fixed window

class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for simulating trading of 10Y Treasury futures.
    Adheres to the Gymnasium API.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4} # Gymnasium metadata

    def __init__(self, data_df, config, current_step_in_df=0, logger=None): # Added logger
        """
        Initializes the trading environment.

        Args:
            data_df (pd.DataFrame): DataFrame containing market data (prices and features).
            config (dict): Configuration dictionary.
            current_step_in_df (int): Starting step within data_df.
            logger (logging.Logger, optional): Logger instance.
        """
        super(TradingEnv, self).__init__()

        self.df = data_df.copy()
        self.config = config
        self.logger = logger if logger else SimpleLogger() # Use provided logger or a basic fallback

        # Data parameters
        self.price_column = self.config['data']['price_column']
        if self.price_column not in self.df.columns:
            self.logger.error(f"Price column '{self.price_column}' not found in DataFrame.")
            raise ValueError(f"Price column '{self.price_column}' not found in DataFrame.")
        
        self.feature_columns = self.config['data']['feature_columns']
        missing_features = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_features:
            self.logger.error(f"Missing feature columns in DataFrame: {missing_features}")
            raise ValueError(f"Missing feature columns in DataFrame: {missing_features}")

        # Environment parameters from config
        env_cfg = self.config['environment_params']
        self.initial_capital = env_cfg.get('initial_capital', 1000000.0)
        
        # Transaction costs: config provides bps, convert to percentage for calculation
        transaction_cost_bps = env_cfg.get('transaction_cost_bps', 0.5) 
        self.transaction_cost_pct = transaction_cost_bps / 10000.0 # Convert bps to decimal percentage

        self.max_drawdown_pct = env_cfg.get('max_drawdown_pct', 0.20)
        self.stop_loss_penalty = env_cfg.get('stop_loss_penalty', -1.0)
        self.include_position_in_state = env_cfg.get('include_position_in_state', True)
        self.log_portfolio_value = env_cfg.get('log_portfolio_value', True)
        self.episode_max_steps = env_cfg.get('episode_max_steps', None)

        # Reward shaping coefficients
        self.lambda_pnl = env_cfg.get('reward_lambda_pnl', 1.0)
        self.lambda_volatility = env_cfg.get('reward_lambda_volatility', 0.05) # volatility_penalty_factor
        self.lambda_drawdown = env_cfg.get('reward_lambda_drawdown', 0.1)   # drawdown_penalty_factor
        
        self.volatility_penalty_window = env_cfg.get('volatility_penalty_window', 20) # From config

        # Action space: 0 (Short), 1 (Flat), 2 (Long)
        self.action_space = spaces.Discrete(3)

        # Observation space
        obs_space_dim = len(self.feature_columns)
        if self.include_position_in_state:
            obs_space_dim += 1 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32
        )

        # Episode management
        self.current_step = 0 # This will be set correctly in reset() relative to start_step_in_df
        self.start_step_in_df = current_step_in_df # The starting point in the *provided* df slice
        self.max_steps_in_current_df = len(self.df) - 1

        # Portfolio and position tracking (initialized in reset)
        self.portfolio_value = 0.0
        self.peak_portfolio_value = 0.0
        self.current_position = 1 # 0:Short, 1:Flat, 2:Long (matches action space)
        self.current_pnl_pos_representation = 0 # -1:Short, 0:Flat, 1:Long (for P&L calc)
        self.recent_returns = deque(maxlen=self.volatility_penalty_window)
        self.trade_history = [] # For potential detailed logging or analysis

        if self.df.empty:
            self.logger.warning("TradingEnv initialized with empty DataFrame.")
        self.logger.info(f"TradingEnv initialized. Data length: {len(self.df)} steps. Start step in df: {self.start_step_in_df}")
        self.logger.info(f"Transaction cost: {self.transaction_cost_pct*100:.4f}% ({transaction_cost_bps} bps)")


    def _get_observation(self):
        """Constructs the observation vector for the current step."""
        actual_df_idx = self.start_step_in_df + self.current_step
        
        if actual_df_idx < 0 or actual_df_idx > self.max_steps_in_current_df:
            self.logger.warning(f"current_step {self.current_step} (actual_df_idx {actual_df_idx}) is out of bounds for df (0 to {self.max_steps_in_current_df}). Returning zero observation.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        features = self.df[self.feature_columns].iloc[actual_df_idx].values.astype(np.float32)
        
        if self.include_position_in_state:
            position_feature = np.array([self.current_position], dtype=np.float32)
            observation = np.concatenate((features, position_feature))
        else:
            observation = features
        return observation

    def reset(self, *, seed=None, options=None): # Gymnasium API
        """Resets the environment for a new episode."""
        super().reset(seed=seed) # Handles seeding for action_space.sample() etc.

        self.current_step = 0 # Relative to the start of the episode
        
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.current_position = 1  # Start Flat
        self.current_pnl_pos_representation = 0 

        self.recent_returns.clear()
        self.trade_history = []
        
        initial_obs = self._get_observation()
        info = {"message": "Environment reset", "initial_portfolio_value": self.portfolio_value}
        
        # self.logger.debug(f"Env Reset. Initial Obs Shape: {initial_obs.shape}. Initial Portfolio: {self.portfolio_value}")
        return initial_obs, info

    def step(self, action):
        """Advances the environment by one time step."""
        terminated = False
        truncated = False
        
        actual_df_idx = self.start_step_in_df + self.current_step

        if actual_df_idx > self.max_steps_in_current_df:
            # This should ideally be caught by termination/truncation logic before _get_observation is called for next step
            self.logger.error("Step called when already past end of data. This indicates a logic flaw.")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0, True, False, {"error": "Stepped past end of data"}


        prev_portfolio_value = self.portfolio_value
        prev_position_action_code = self.current_position 
        prev_pnl_pos = self.current_pnl_pos_representation

        # Update position based on action
        self.current_position = int(action) # Ensure action is int
        if self.current_position == 0: self.current_pnl_pos_representation = -1 # Short
        elif self.current_position == 1: self.current_pnl_pos_representation = 0  # Flat
        else: self.current_pnl_pos_representation = 1  # Long
            
        # P&L from market movement
        pnl_from_market = 0
        if (actual_df_idx + 1) <= self.max_steps_in_current_df: # Ensure there's a next price
            current_price = self.df[self.price_column].iloc[actual_df_idx]
            next_price = self.df[self.price_column].iloc[actual_df_idx + 1]
            price_change = next_price - current_price
            pnl_from_market = prev_pnl_pos * price_change
        else: # Last step in data, no P&L from market movement for this step's end
            price_change = 0 

        # Transaction costs
        transaction_cost = 0
        if self.current_position != prev_position_action_code:
            current_price_for_cost = self.df[self.price_column].iloc[actual_df_idx]
            cost_per_unit_trade = self.transaction_cost_pct * current_price_for_cost
            
            if prev_pnl_pos != 0 and self.current_pnl_pos_representation != 0 and prev_pnl_pos != self.current_pnl_pos_representation:
                transaction_cost = 2 * cost_per_unit_trade # Close and Open opposite
            elif prev_pnl_pos != self.current_pnl_pos_representation:
                transaction_cost = 1 * cost_per_unit_trade
            self.trade_history.append({'step': self.current_step, 'action': self.current_position, 'price': current_price_for_cost, 'cost': transaction_cost})

        # Update portfolio
        self.portfolio_value += pnl_from_market
        self.portfolio_value -= transaction_cost
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        # Reward components
        reward_pnl = self.lambda_pnl * (pnl_from_market - transaction_cost)

        daily_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value != 0 else 0.0
        self.recent_returns.append(daily_return)
        
        reward_volatility_penalty = 0
        if len(self.recent_returns) == self.volatility_penalty_window:
            # Using squared daily portfolio return as volatility penalty source
            reward_volatility_penalty = self.lambda_volatility * (daily_return ** 2) * 100 # Scaled up a bit
            # Could also use: self.lambda_volatility * np.std(self.recent_returns)

        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0.0
        reward_drawdown_penalty = self.lambda_drawdown * current_drawdown if current_drawdown > 0 else 0.0
        
        reward = reward_pnl - reward_volatility_penalty - reward_drawdown_penalty
        
        # Termination and Truncation
        stop_loss_triggered = False
        if current_drawdown > self.max_drawdown_pct:
            terminated = True
            stop_loss_triggered = True
            reward += self.stop_loss_penalty 
            self.logger.info(f"Stop-loss triggered at step {self.current_step}. Drawdown: {current_drawdown*100:.2f}%")

        self.current_step += 1 # Advance internal step counter for the episode

        if not terminated and (self.start_step_in_df + self.current_step) > self.max_steps_in_current_df:
            terminated = True # End of available data for this episode run

        if not terminated and self.episode_max_steps is not None and self.current_step >= self.episode_max_steps:
            truncated = True # Reached max configured steps for an episode
        
        # Get next observation
        observation = self._get_observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            'portfolio_value': self.portfolio_value if self.log_portfolio_value else 'not_logged',
            'pnl_pos_representation': self.current_pnl_pos_representation,
            'transaction_cost': transaction_cost,
            'pnl_from_market': pnl_from_market,
            'reward_pnl_component': reward_pnl,
            'reward_vol_penalty_component': -reward_volatility_penalty,
            'reward_dd_penalty_component': -reward_drawdown_penalty,
            'daily_return': daily_return,
            'drawdown_pct': current_drawdown * 100,
            'stop_loss_triggered': stop_loss_triggered
        }
        
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'): # Gymnasium API uses 'mode', not 'render_modes'
        """Renders the environment status."""
        if mode == 'human':
            actual_df_idx = self.start_step_in_df + self.current_step -1 # -1 because current_step was already incremented
            price_info = f"Price: {self.df[self.price_column].iloc[actual_df_idx]:.2f}" if actual_df_idx < len(self.df) else "Price: N/A"
            
            print(f"Step: {self.current_step}, DF Idx: {actual_df_idx}, {price_info}, "
                  f"Portfolio: {self.portfolio_value:.2f}, Position: {self.current_pnl_pos_representation}, "
                  f"Drawdown: {( (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0) * 100:.2f}%")

    def close(self):
        """Perform any necessary cleanup."""
        # self.logger.info("TradingEnv closed.")
        pass

# Dummy logger for standalone execution if actual logger isn't passed
class SimpleLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")

if __name__ == '__main__':
    print("Example Usage of TradingEnv (Gymnasium compliant):")
    dummy_df = pd.DataFrame({
        'Date': pd.to_datetime([f'2020-01-{i+1:02d}' for i in range(200)]),
        'price': np.random.randn(200).cumsum() + 100,
        'yield_curve_slope': np.random.rand(200) * 10,
        'volatility_20d': np.random.rand(200) * 5,
        'ma_crossover_signal': np.random.randint(-1, 2, size=200),
        'momentum_1m': np.random.randn(200)
    }).set_index('Date')

    dummy_yaml_config = {
        "data": {"price_column": "price", "feature_columns": ["yield_curve_slope", "volatility_20d", "ma_crossover_signal", "momentum_1m"]},
        "environment_params": {
            "initial_capital": 100000, "transaction_cost_bps": 1.0, # 1 bps
            "max_drawdown_pct": 0.15, "stop_loss_penalty": -100,
            "include_position_in_state": True, "log_portfolio_value": True,
            "reward_lambda_pnl": 1.0, "reward_lambda_volatility": 0.1, "reward_lambda_drawdown": 0.2,
            "volatility_penalty_window": 10, "episode_max_steps": 50 
        }
    }
    test_logger = SimpleLogger()
    env = TradingEnv(data_df=dummy_df, config=dummy_yaml_config, logger=test_logger)
    
    obs, info = env.reset(seed=42)
    test_logger.info(f"Initial Observation: {obs[:4]}... Shape: {obs.shape}")
    test_logger.info(f"Reset Info: {info}")

    total_reward_sum = 0
    for i in range(100): # Max 100 steps for this test
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_sum += reward
        if i % 10 == 0:
            # env.render() # Print step info using render
            test_logger.info(f"Step {i}: Action: {action}, Reward: {reward:.4f}, Term: {terminated}, Trunc: {truncated}, PV: {info.get('portfolio_value',0):.2f}")
        if terminated or truncated:
            test_logger.info(f"Episode finished after {i+1} steps. Final Info: {info}")
            break
    test_logger.info(f"Total reward for random actions: {total_reward_sum:.2f}")
    env.close()
