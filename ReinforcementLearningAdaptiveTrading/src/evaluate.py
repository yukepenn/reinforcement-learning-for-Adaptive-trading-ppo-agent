import argparse
import os
import pandas as pd
import numpy as np
import yaml # Or use json/configparser if preferred for config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Assuming other necessary modules will be created:
from environment.trading_env import TradingEnv # Assuming TradingEnv is in environment/trading_env.py
from data.data_loader import load_data, split_data # Assuming these are in data/data_loader.py
# from data.feature_engineering import engineer_features # Assuming this function exists
from utils.metrics import calculate_sharpe_ratio, max_drawdown, annualized_volatility, total_return # Assuming these in utils/metrics.py
from config import DEFAULT_CONFIG # Assuming config.py will have this

def evaluate_agent(model, env, data_split_name="test"):
    """
    Evaluates the trained agent on a given environment.
    
    :param model: The trained PPO model.
    :param env: The Gym environment (preferably the test environment).
    :param data_split_name: Name of the data split (e.g., "test", "validation") for logging.
    :return: A dictionary of performance metrics.
    """
    print(f"Evaluating agent on {data_split_name} data...")
    obs = env.reset()
    done = False
    
    episode_rewards = []
    portfolio_values = [] # To store portfolio value at each step for equity curve
    all_actions = []
    
    # Assuming the environment has access to its underlying data's length
    # For a vectorized env, env.num_envs will be > 1. We evaluate on the first one.
    # This loop structure might need adjustment if using vectorized env directly for evaluation.
    # Typically, for evaluation, a single, non-vectorized env is preferred.
    
    current_portfolio_value = env.get_attr("initial_capital")[0] # Get from underlying env
    portfolio_values.append(current_portfolio_value)

    # Determine number of steps based on the environment's data
    # This is a heuristic; TradingEnv should ideally expose its data length
    num_steps = 0
    if hasattr(env.unwrapped.envs[0], 'df') and env.unwrapped.envs[0].df is not None:
        num_steps = len(env.unwrapped.envs[0].df) -1 # -1 because episode starts with reset
    else: # Fallback if df is not directly accessible
        print("Warning: Could not determine exact number of steps from env.df. Running for a default number of steps.")
        num_steps = 252 # Default to one year of trading days if length unknown
        if data_split_name == "test" and hasattr(env.unwrapped.envs[0], 'config'):
            # Try to get test period length if possible
            cfg = env.unwrapped.envs[0].config
            test_start = pd.to_datetime(cfg.get('data',{}).get('test_period',{}).get('start', '2016-01-01'))
            test_end = pd.to_datetime(cfg.get('data',{}).get('test_period',{}).get('end', '2023-12-31'))
            # This is a rough estimate, actual days depend on data
            num_steps = (test_end - test_start).days  * (252/365) # Approximate trading days


    for i in range(int(num_steps)): # Ensure num_steps is an integer
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        all_actions.append(action[0] if isinstance(action, list) or isinstance(action, np.ndarray) else action)
        episode_rewards.append(rewards[0] if isinstance(rewards, list) else rewards) # Assuming single env for eval
        
        # Portfolio value might be in info dict from TradingEnv
        # If using multiple envs, info will be a list of dicts
        current_portfolio_value = info[0].get('portfolio_value', portfolio_values[-1] + rewards[0])
        portfolio_values.append(current_portfolio_value)
        
        if dones[0] if isinstance(dones, list) else dones:
            print(f"Episode finished after {i+1} steps.")
            break
    
    if not episode_rewards: # handles case where num_steps is 0 or very small
        return {
            "total_return_pct": 0,
            "annualized_return_pct": 0,
            "annualized_volatility_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "average_reward": 0,
            "actions": [],
            "portfolio_values": portfolio_values
        }

    # Calculate metrics
    returns = pd.Series(episode_rewards) # Daily rewards can be treated as returns if reward = P&L
    
    # More accurately, use portfolio values for returns if reward is shaped
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

    metrics = {}
    metrics["total_return_pct"] = total_return(pd.Series(portfolio_values)) * 100
    metrics["annualized_return_pct"] = portfolio_returns.mean() * 252 * 100 # Assuming daily data
    metrics["annualized_volatility_pct"] = annualized_volatility(portfolio_returns) * 100
    metrics["sharpe_ratio"] = calculate_sharpe_ratio(portfolio_returns) # Assumes daily returns
    metrics["max_drawdown_pct"] = max_drawdown(pd.Series(portfolio_values)) * 100
    metrics["average_reward"] = np.mean(episode_rewards)
    metrics["actions"] = all_actions
    metrics["portfolio_values"] = portfolio_values
    
    print(f"--- Agent Performance on {data_split_name} ---")
    for key, value in metrics.items():
        if key not in ["actions", "portfolio_values"]:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
    
    return metrics

def run_baseline_strategy(data_df, strategy_name="buy_and_hold", config=None):
    """
    Simulates a baseline trading strategy.
    
    :param data_df: DataFrame with price data (at least a 'price' column).
    :param strategy_name: Name of the baseline strategy ("buy_and_hold", "ma_crossover").
    :param config: Configuration dictionary, may contain parameters for strategies.
    :return: A dictionary of performance metrics for the baseline.
    """
    print(f"Running baseline strategy: {strategy_name}...")
    if data_df.empty or 'price' not in data_df.columns:
        print(f"Price data is missing or empty for baseline: {strategy_name}.")
        return {}

    initial_capital = config.get('environment_params', {}).get('initial_capital', 1000000)
    portfolio_values = [initial_capital]
    actions = [] # 0: Short, 1: Flat, 2: Long

    if strategy_name == "buy_and_hold":
        # Assumes buying one unit whose value is the price. For futures, this needs adjustment.
        # More simply, portfolio follows price changes.
        price_at_start = data_df['price'].iloc[0]
        for i in range(len(data_df)):
            current_price = data_df['price'].iloc[i]
            # Portfolio value relative to start, scaled by initial capital
            pv = initial_capital * (current_price / price_at_start)
            portfolio_values.append(pv)
            actions.append(2) # Always Long
    
    elif strategy_name == "ma_crossover":
        if 'short_ma' not in data_df.columns or 'long_ma' not in data_df.columns:
            # Calculate MAs if not present (e.g., from feature_engineering.py)
            # This is a placeholder; actual MA calculation should be robust.
            ma_short_window = config.get('baseline_params',{}).get('ma_crossover',{}).get('short_window', 50)
            ma_long_window = config.get('baseline_params',{}).get('ma_crossover',{}).get('long_window', 200)
            data_df['short_ma'] = data_df['price'].rolling(window=ma_short_window).mean()
            data_df['long_ma'] = data_df['price'].rolling(window=ma_long_window).mean()
            data_df = data_df.dropna() # Drop NA results from rolling mean
            if data_df.empty:
                print("Not enough data for MA crossover after dropping NAs.")
                return {}


        position = 0 # Start flat
        current_capital = initial_capital
        
        # Need to re-iterate for MA crossover based on its own logic for PV
        portfolio_values = [initial_capital] # Reset for this strategy
        
        # Simplified P&L calculation for MA crossover (assumes 1 unit)
        # More accurate simulation would use TradingEnv logic if possible
        for i in range(1, len(data_df)): # Start from 1 because we look at previous day's signal for today's P&L
            prev_price = data_df['price'].iloc[i-1]
            current_price = data_df['price'].iloc[i]
            price_change = current_price - prev_price

            # Signal based on previous day's MAs
            signal_long = data_df['short_ma'].iloc[i-1] > data_df['long_ma'].iloc[i-1]
            signal_short = data_df['short_ma'].iloc[i-1] < data_df['long_ma'].iloc[i-1]

            daily_pnl = 0
            if position == 1: # Long
                daily_pnl = price_change
            elif position == -1: # Short
                daily_pnl = -price_change
            
            current_capital += daily_pnl # Ignores transaction costs for simplicity here
            portfolio_values.append(current_capital)

            if signal_long:
                position = 1 # Go Long
                actions.append(2)
            elif signal_short:
                position = -1 # Go Short
                actions.append(0)
            else: # MAs are equal or not enough data
                position = 0 # Go Flat
                actions.append(1)
    
    elif strategy_name == "always_flat":
        portfolio_values = [initial_capital] * (len(data_df) + 1) # +1 to match length if loop runs
        actions = [1] * len(data_df)

    else:
        print(f"Unknown baseline strategy: {strategy_name}")
        return {}

    if not portfolio_values or len(portfolio_values) <= 1: # Need at least 2 values for pct_change
         return {
            "total_return_pct": 0, "annualized_return_pct": 0, "annualized_volatility_pct": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0, "actions": actions, "portfolio_values": portfolio_values
        }

    # Calculate metrics for baseline
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    if portfolio_returns.empty: # Handle cases with no change or single value
        metrics = {
            "total_return_pct": 0, "annualized_return_pct": 0, "annualized_volatility_pct": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0
        }
    else:
        metrics = {}
        metrics["total_return_pct"] = total_return(pd.Series(portfolio_values)) * 100
        metrics["annualized_return_pct"] = portfolio_returns.mean() * 252 * 100
        metrics["annualized_volatility_pct"] = annualized_volatility(portfolio_returns) * 100
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(portfolio_returns)
        metrics["max_drawdown_pct"] = max_drawdown(pd.Series(portfolio_values)) * 100
    
    metrics["actions"] = actions
    metrics["portfolio_values"] = portfolio_values

    print(f"--- Baseline: {strategy_name} Performance ---")
    for key, value in metrics.items():
        if key not in ["actions", "portfolio_values"]:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO Agent and Baselines for Adaptive Trading")
    parser.add_argument("--config_path", type=str, default="src/config.py", help="Path to the configuration file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model (.zip file)")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "test", "validation"], help="Data split to evaluate on")
    parser.add_argument("--vec_normalize_stats_path", type=str, default=None, help="Path to VecNormalize statistics (.pkl file) if environment was normalized during training")

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config_path.endswith('.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = getattr(config_module, 'DEFAULT_CONFIG', {})
    elif args.config_path.endswith(('.yaml', '.yml')):
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Unsupported configuration file format: {args.config_path}. Using default config.")
        config = DEFAULT_CONFIG

    # Load data for the specified split
    # This is a placeholder - actual data loading will depend on data_loader.py
    print(f"Placeholder: Loading {args.data_split} data...")
    # Example:
    # raw_data = load_data(config['data']['raw_data_path']) 
    # _, eval_df = split_data(raw_data, config['data']['train_period'], config['data'][f'{args.data_split}_period'])
    # eval_features_df = engineer_features(eval_df) # From feature_engineering.py

    # Create dummy data for now
    dummy_dates = pd.to_datetime([f'2022-01-{i+1:02d}' for i in range(100)]) # Example test period
    dummy_data_dict = {'price': np.random.rand(100) * 100 + 1200}
    for feature_name in config.get('environment_params', {}).get('feature_columns', []):
        dummy_data_dict[feature_name] = np.random.rand(100)
    eval_df = pd.DataFrame(dummy_data_dict, index=dummy_dates)


    # Create evaluation environment
    env_kwargs = {'data': eval_df, 'config': config, 'current_step_in_df': 0} # current_step_in_df for TradingEnv
    
    # For evaluation, typically use a single environment
    # The make_vec_env or DummyVecEnv is often used for consistency with training if VecNormalize was used.
    eval_env = DummyVecEnv([lambda: TradingEnv(**env_kwargs)])

    if args.vec_normalize_stats_path and os.path.exists(args.vec_normalize_stats_path):
        print(f"Loading VecNormalize statistics from {args.vec_normalize_stats_path}")
        eval_env = VecNormalize.load(args.vec_normalize_stats_path, eval_env)
        eval_env.training = False # Important: set to False for evaluation
        eval_env.norm_reward = False # Usually, do not normalize rewards during evaluation
    elif config.get('training_params', {}).get('normalize_env', False) and not args.vec_normalize_stats_path:
        print("Warning: Environment was normalized during training, but no vec_normalize_stats_path provided for evaluation. Observations might be scaled differently.")


    # Load the trained model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    print(f"Loading trained model from {args.model_path}...")
    # Pass the eval_env if it's a VecNormalized environment to ensure consistency,
    # though SB3 PPO.load usually handles this if the original env was normalized.
    # For custom_objects, if any custom policies were used.
    model = PPO.load(args.model_path, env=eval_env if args.vec_normalize_stats_path else None)
    print("Model loaded successfully.")

    # Evaluate the agent
    agent_metrics = evaluate_agent(model, eval_env, data_split_name=args.data_split)

    # Run baseline strategies
    # Baselines need the raw DataFrame with prices and potentially features for MA crossover
    baseline_buy_and_hold_metrics = run_baseline_strategy(eval_df.copy(), "buy_and_hold", config) # Pass copy
    baseline_ma_crossover_metrics = run_baseline_strategy(eval_df.copy(), "ma_crossover", config) # Pass copy
    baseline_always_flat_metrics = run_baseline_strategy(eval_df.copy(), "always_flat", config)

    # Further: Compare metrics, generate plots (e.g., equity curves)
    # This could be done here or by saving results to a file for notebook analysis.
    print("\n--- Summary of Results ---")
    # A simple print, could be a table or CSV output
    if agent_metrics:
      print(f"Agent Sharpe Ratio: {agent_metrics.get('sharpe_ratio', 'N/A'):.2f}, Max Drawdown: {agent_metrics.get('max_drawdown_pct', 'N/A'):.2f}%")
    if baseline_buy_and_hold_metrics:
      print(f"Buy & Hold Sharpe Ratio: {baseline_buy_and_hold_metrics.get('sharpe_ratio', 'N/A'):.2f}, Max Drawdown: {baseline_buy_and_hold_metrics.get('max_drawdown_pct', 'N/A'):.2f}%")
    if baseline_ma_crossover_metrics:
      print(f"MA Crossover Sharpe Ratio: {baseline_ma_crossover_metrics.get('sharpe_ratio', 'N/A'):.2f}, Max Drawdown: {baseline_ma_crossover_metrics.get('max_drawdown_pct', 'N/A'):.2f}%")

    eval_env.close()
    print("\nEvaluation script completed.")

if __name__ == "__main__":
    main()
