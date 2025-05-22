import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for simulating trading of 10Y Treasury futures.

    The environment state includes market features and optionally the current position.
    Actions are discrete: 0 (Short), 1 (Flat), 2 (Long).
    The reward function is shaped to promote risk-adjusted returns, penalizing
    volatility and drawdowns.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_df, config, current_step_in_df=0):
        """
        Initializes the trading environment.

        Args:
            data_df (pd.DataFrame): DataFrame containing market data (prices and features).
                                    Must have a 'price' column and feature columns as
                                    specified in config['data']['feature_columns'].
            config (dict): Configuration dictionary containing parameters for the environment,
                           such as initial capital, transaction costs, reward coefficients, etc.
            current_step_in_df (int): The starting step within the data_df. Useful if data_df is
                                      a slice of a larger dataset (e.g. for evaluation on test set)
        """
        super(TradingEnv, self).__init__()

        self.df = data_df.copy() # Store the data
        self.config = config
        
        # Ensure 'price' column exists, critical for P&L calculation
        if self.config['data']['price_column'] not in self.df.columns:
            raise ValueError(f"Price column '{self.config['data']['price_column']}' not found in DataFrame.")
        self.price_column = self.config['data']['price_column']

        # Define feature columns from config
        self.feature_columns = self.config['data']['feature_columns']
        missing_features = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_features}")

        # Environment parameters
        self.initial_capital = self.config['environment_params'].get('initial_capital', 1000000.0)
        self.transaction_cost_pct = self.config['environment_params'].get('transaction_cost_pct', 0.0001)
        self.max_drawdown_pct = self.config['environment_params'].get('max_drawdown_pct', 0.20)
        self.stop_loss_penalty = self.config['environment_params'].get('stop_loss_penalty', -1.0)
        self.include_position_in_state = self.config['environment_params'].get('include_position_in_state', True)
        self.log_portfolio_value = self.config['environment_params'].get('log_portfolio_value', True)
        
        # Reward shaping coefficients
        self.lambda_pnl = self.config['environment_params'].get('reward_lambda_pnl', 1.0)
        self.lambda_volatility = self.config['environment_params'].get('reward_lambda_volatility', 0.05)
        self.lambda_drawdown = self.config['environment_params'].get('reward_lambda_drawdown', 0.1)

        # Action space: 0 (Short), 1 (Flat), 2 (Long)
        self.action_space = spaces.Discrete(3) # Short, Flat, Long

        # Observation space: market features + current position (if included)
        # Normalize features beforehand if necessary.
        # The shape is number of features + 1 if position is included (as a single integer).
        # If position is one-hot encoded, it would be +3. Here, using a single integer.
        obs_space_dim = len(self.feature_columns)
        if self.include_position_in_state:
            obs_space_dim += 1
        
        # Assuming features are pre-normalized or will be handled by VecNormalize wrapper
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32
        )

        # Episode management
        self.current_step = current_step_in_df 
        self.start_step = current_step_in_df # To reset to the correct beginning of the provided df slice
        self.max_steps_in_df = len(self.df) -1 # Max index in the dataframe

        # Portfolio and position tracking
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.current_position = 1 # Start Flat (0: Short, 1: Flat, 2: Long internal representation for position)
                                  # This matches action space: 0=Short, 1=Flat, 2=Long for simplicity
                                  # But for P&L: -1 for short, 0 for flat, 1 for long is more natural
        self.current_pnl_pos_representation = 0 # -1 for short, 0 for flat, 1 for long

        # For volatility penalty, track recent returns
        self.recent_returns = [] # Could be a deque with maxlen
        self.volatility_penalty_window = 20 # Look at last 20 daily returns for vol penalty

        print(f"TradingEnv initialized. Data length: {len(self.df)} steps. Start step: {self.start_step}")
        if self.df.empty:
            print("Warning: TradingEnv initialized with empty DataFrame.")


    def _get_observation(self):
        """Constructs the observation vector for the current step."""
        if self.current_step < 0 or self.current_step > self.max_steps_in_df:
             # This case should ideally be prevented by done logic in step()
            print(f"Warning: current_step {self.current_step} is out of bounds for df (0 to {self.max_steps_in_df}). Returning zero observation.")
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32)

        features = self.df[self.feature_columns].iloc[self.current_step].values.astype(np.float32)
        
        if self.include_position_in_state:
            # Represent position as 0 (Short), 1 (Flat), 2 (Long) to match action space directly
            # Or normalize: (self.current_position - 1) maps to (-1, 0, 1)
            position_feature = np.array([self.current_position], dtype=np.float32) 
            observation = np.concatenate((features, position_feature))
        else:
            observation = features
            
        return observation

    def reset(self):
        """Resets the environment to the initial state for a new episode."""
        self.current_step = self.start_step # Reset to the beginning of the provided DataFrame slice
        
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.current_position = 1  # Start Flat (0:Short, 1:Flat, 2:Long)
        self.current_pnl_pos_representation = 0 # (Short:-1, Flat:0, Long:1)

        self.recent_returns = []
        
        # print(f"Environment Reset. Current Step: {self.current_step}, Portfolio: {self.portfolio_value}")
        return self._get_observation()

    def step(self, action):
        """
        Advances the environment by one time step based on the agent's action.

        Args:
            action (int): The agent's chosen action (0: Short, 1: Flat, 2: Long).

        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.df.empty or self.current_step > self.max_steps_in_df :
            # Should not happen if done is handled correctly
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32), 0, True, {}

        # Store previous state for reward calculation
        prev_portfolio_value = self.portfolio_value
        prev_position = self.current_position # Action space representation (0,1,2)
        prev_pnl_pos = self.current_pnl_pos_representation # P&L representation (-1,0,1)

        # Execute action: Update position
        # Action: 0 (Short), 1 (Flat), 2 (Long)
        # Position (current_position): 0 (Short), 1 (Flat), 2 (Long)
        # P&L Position (current_pnl_pos_representation): -1 (Short), 0 (Flat), 1 (Long)
        
        self.current_position = action 
        if action == 0: # Short
            self.current_pnl_pos_representation = -1
        elif action == 1: # Flat
            self.current_pnl_pos_representation = 0
        else: # Long (action == 2)
            self.current_pnl_pos_representation = 1
            
        # Calculate P&L from market movement
        # Price change from current_step to current_step + 1 affects P&L for position held AT current_step
        # So, we need price at current_step and current_step + 1
        # If current_step is the last step, P&L from market movement is 0 as episode ends.
        
        pnl_from_market = 0
        if self.current_step < self.max_steps_in_df:
            current_price = self.df[self.price_column].iloc[self.current_step]
            next_price = self.df[self.price_column].iloc[self.current_step + 1]
            price_change = next_price - current_price
            pnl_from_market = prev_pnl_pos * price_change # prev_pnl_pos is position taken *before* this step's action
                                                        # that is affected by price_change from t to t+1

        # Calculate transaction costs if position changed
        transaction_cost = 0
        if self.current_position != prev_position: # If target position changed
            # Simplified: cost applies per change of state (e.g. flat to long, or long to short)
            # A more complex model could charge for closing and opening if reversing.
            # For example, if prev_pos was Long (2) and current_pos is Short (0), it's one significant change.
            # Cost is based on current price at the time of transaction.
            current_price_for_cost = self.df[self.price_column].iloc[self.current_step]
            cost_per_unit_trade = self.transaction_cost_pct * current_price_for_cost
            
            # Number of "sides" traded. From Long to Short = 2 (close long, open short)
            # From Long to Flat = 1. From Flat to Long = 1.
            if prev_pnl_pos != 0 and self.current_pnl_pos_representation != 0 and prev_pnl_pos != self.current_pnl_pos_representation:
                transaction_cost = 2 * cost_per_unit_trade # Close and Open opposite
            elif prev_pnl_pos != self.current_pnl_pos_representation: # Flat to Long/Short, or Long/Short to Flat
                transaction_cost = 1 * cost_per_unit_trade
        
        # Update portfolio value
        self.portfolio_value += pnl_from_market
        self.portfolio_value -= transaction_cost

        # Update peak portfolio value for drawdown calculation
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        # Calculate reward components
        reward_pnl = self.lambda_pnl * (pnl_from_market - transaction_cost) # Net P&L after costs

        # Volatility penalty (based on recent portfolio returns)
        # Use change in portfolio value as a proxy for return if not holding cash aside
        daily_return = 0
        if prev_portfolio_value > 0: # Avoid division by zero if capital is somehow lost
             daily_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        self.recent_returns.append(daily_return)
        if len(self.recent_returns) > self.volatility_penalty_window:
            self.recent_returns.pop(0)
        
        reward_volatility_penalty = 0
        if len(self.recent_returns) == self.volatility_penalty_window:
            # Penalize squared returns or std dev of returns
            # Using squared daily return as a simple proxy for volatility contribution
            reward_volatility_penalty = self.lambda_volatility * (daily_return ** 2)
            # Alternatively, std of recent_returns: self.lambda_volatility * np.std(self.recent_returns)

        # Drawdown penalty
        current_drawdown = 0
        if self.peak_portfolio_value > 0: # Avoid division by zero
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        
        reward_drawdown_penalty = 0
        if current_drawdown > 0: # Only penalize if there is a drawdown
            # Penalize proportionally to drawdown size
            reward_drawdown_penalty = self.lambda_drawdown * current_drawdown
            # Could also penalize only if it's a *new* max drawdown, or if it exceeds a threshold.

        # Stop-loss logic
        done = False
        stop_loss_triggered = False
        if current_drawdown > self.max_drawdown_pct:
            done = True
            stop_loss_triggered = True
            reward_pnl += self.stop_loss_penalty # Apply large penalty for hitting stop-loss
            print(f"Stop-loss triggered at step {self.current_step}. Drawdown: {current_drawdown*100:.2f}% > {self.max_drawdown_pct*100:.2f}%")


        # Combine reward components
        reward = reward_pnl - reward_volatility_penalty - reward_drawdown_penalty
        
        # Advance time step
        self.current_step += 1

        # Check if episode is done
        if not done and self.current_step > self.max_steps_in_df : # End of data
            done = True
        
        # Max episode steps from config (if any)
        if self.config['environment_params'].get('episode_max_steps') is not None:
            if (self.current_step - self.start_step) >= self.config['environment_params']['episode_max_steps']:
                done = True
        
        # Information dictionary
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value if self.log_portfolio_value else 'not_logged',
            'current_position': self.current_position, # 0:S, 1:F, 2:L
            'pnl_pos_representation': self.current_pnl_pos_representation, # -1:S, 0:F, 1:L
            'transaction_cost': transaction_cost,
            'pnl_from_market': pnl_from_market,
            'drawdown_pct': current_drawdown * 100,
            'stop_loss_triggered': stop_loss_triggered,
            'reward_components': {
                'pnl_reward': reward_pnl,
                'vol_penalty': -reward_volatility_penalty,
                'dd_penalty': -reward_drawdown_penalty
            }
        }
        
        return self._get_observation(), reward, done, info

    def render(self, mode='human', close=False):
        """(Optional) Renders the environment, e.g., by printing current status."""
        if mode == 'human':
            print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, "
                  f"Position: {self.current_pnl_pos_representation} (-1S,0F,1L), "
                  f"Drawdown: {( (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0) * 100:.2f}%")

    def close(self):
        """(Optional) Perform any necessary cleanup."""
        # print("TradingEnv closed.")
        pass

if __name__ == '__main__':
    # Example Usage (requires a dummy config and data)
    print("Example Usage of TradingEnv:")

    # Create dummy data
    num_days = 200
    price_data = np.random.randn(num_days).cumsum() + 100
    feature_data_1 = np.random.rand(num_days) * 10
    feature_data_2 = np.random.rand(num_days) * 5
    
    dummy_df = pd.DataFrame({
        'price': price_data,
        'yield_curve_slope': feature_data_1,
        'volatility_20d': feature_data_2,
        'ma_crossover_signal': np.random.randint(-1, 2, size=num_days),
        'momentum_1m': np.random.randn(num_days)
    })

    # Dummy config (subset of what's in config.py)
    dummy_config = {
        "data": {
            "price_column": "price",
            "feature_columns": ["yield_curve_slope", "volatility_20d", "ma_crossover_signal", "momentum_1m"]
        },
        "environment_params": {
            "initial_capital": 100000,
            "transaction_cost_pct": 0.001, # Higher cost for testing
            "max_drawdown_pct": 0.15,
            "stop_loss_penalty": -100, # Large penalty
            "include_position_in_state": True,
            "log_portfolio_value": True,
            "reward_lambda_pnl": 1.0,
            "reward_lambda_volatility": 0.1,
            "reward_lambda_drawdown": 0.2,
            "episode_max_steps": None 
        }
    }

    env = TradingEnv(data_df=dummy_df, config=dummy_config)
    obs = env.reset()
    print(f"Initial Observation: {obs}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    total_reward_sum = 0
    for i in range(num_days - 5): # Run for a few steps less than total to avoid end-of-data issues in simple example
        action = env.action_space.sample()  # Sample random action
        obs, reward, done, info = env.step(action)
        total_reward_sum += reward
        # env.render() # Print step info
        if i % 50 == 0 :
            print(f"Step {i}: Action: {action}, Reward: {reward:.4f}, Portfolio: {info.get('portfolio_value',0):.2f}, Done: {done}")
            # print(f"   Info: {info}")
        if done:
            print(f"Episode finished after {i+1} steps. Final Portfolio: {info.get('portfolio_value',0):.2f}")
            break
    
    print(f"Total reward for random actions: {total_reward_sum:.2f}")
    env.close()
