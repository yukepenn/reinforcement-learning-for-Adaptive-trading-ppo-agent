import argparse
import os
import pandas as pd
import numpy as np
import yaml # For loading YAML config
import matplotlib.pyplot as plt # For basic plotting

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Adjusted import paths (similar to train.py)
# from environment.trading_env import TradingEnv
# from data.data_loader import load_data # We'll load processed data directly
# from utils.metrics import calculate_sharpe_ratio, max_drawdown, annualized_volatility, total_return # and others
# from utils.config_loader import load_config
# from utils.logger import setup_logger
# from utils.plotting import plot_equity_curves # Will be created later

# TEMPORARY: Direct imports assuming evaluate.py is in src/ for now
from environment.trading_env import TradingEnv
# We'll load processed data, so direct use of data_loader might not be needed here,
# but feature_engineer might be for baselines if features aren't in the loaded test_df.
from data_processing.feature_engineer import calculate_moving_average # For MA baseline
import utils.metrics as metrics_module # Use as a module to avoid naming conflicts

# Placeholder for logger
# logger = print # Temporary, replace with actual logger

# --- Agent Evaluation Function ---
def evaluate_agent(model, env, config, logger, data_split_name="test"): # Added logger
    """
    Evaluates the trained agent on a given environment.
    """
    logger.info(f"Evaluating agent on {data_split_name} data...")
    
    obs = env.reset()
    # done = False # Not needed as dones is a list from vecenv
    
    episode_rewards = []
    portfolio_values = [] 
    all_actions = []
    
    initial_capital = env.get_attr("initial_capital")[0]
    portfolio_values.append(initial_capital)

    num_steps = 0
    # Try to get actual length from the DataFrame used by the environment instance
    if hasattr(env.unwrapped.envs[0], 'df') and env.unwrapped.envs[0].df is not None:
        num_steps = len(env.unwrapped.envs[0].df) - 1 # df has dates, steps are transitions
    else:
        logger.warning("Could not determine exact number of steps from env.df. Evaluation might be incomplete.")
        # Fallback: use test period from config to estimate, though less reliable
        test_period_cfg = config.get('data', {}).get(f'{data_split_name}_period', {})
        if test_period_cfg.get('start') and test_period_cfg.get('end'):
            start_date = pd.to_datetime(test_period_cfg['start'])
            end_date = pd.to_datetime(test_period_cfg['end'])
            num_steps = (end_date - start_date).days # Approximation
            logger.info(f"Estimated num_steps from config period: {num_steps}")
        else:
            num_steps = 252 # Default if no other info
            logger.warning(f"Defaulting num_steps to {num_steps}")


    for i in range(int(num_steps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, step_rewards, dones, infos = env.step(action) # step_rewards for clarity
        
        all_actions.append(action[0]) # Assuming single env for eval
        episode_rewards.append(step_rewards[0])
        
        current_portfolio_value = infos[0].get('portfolio_value', portfolio_values[-1] + step_rewards[0])
        portfolio_values.append(current_portfolio_value)
        
        if dones[0]:
            logger.info(f"Agent evaluation episode finished after {i+1} steps.")
            break
    
    if not episode_rewards:
        logger.warning("No rewards collected during agent evaluation.")
        # Return structure with Nones or NaNs for metrics
        return {m: 0 for m in ["total_return_pct", "annualized_return_pct", "annualized_volatility_pct", "sharpe_ratio", "max_drawdown_pct", "average_reward"]} | {"actions": [], "portfolio_values": portfolio_values}


    portfolio_series = pd.Series(portfolio_values)
    daily_returns_series = portfolio_series.pct_change().dropna()

    eval_metrics = {}
    eval_metrics["total_return_pct"] = metrics_module.total_return(portfolio_series) * 100
    eval_metrics["annualized_return_pct"] = metrics_module.annualized_return(daily_returns_series) * 100
    eval_metrics["annualized_volatility_pct"] = metrics_module.annualized_volatility(daily_returns_series) * 100
    eval_metrics["sharpe_ratio"] = metrics_module.sharpe_ratio(daily_returns_series, risk_free_rate=config['evaluation_params'].get('risk_free_rate', 0.0))
    eval_metrics["max_drawdown_pct"] = metrics_module.max_drawdown(portfolio_series) * 100
    eval_metrics["sortino_ratio"] = metrics_module.sortino_ratio(daily_returns_series, required_return=config['evaluation_params'].get('required_return_for_sortino', 0.0))
    eval_metrics["calmar_ratio"] = metrics_module.calmar_ratio(portfolio_series)
    eval_metrics["average_reward"] = np.mean(episode_rewards) if episode_rewards else 0
    eval_metrics["actions"] = all_actions
    eval_metrics["portfolio_values"] = portfolio_values # For plotting
    
    logger.info(f"--- Agent Performance on {data_split_name} ---")
    for key, value in eval_metrics.items():
        if key not in ["actions", "portfolio_values"]:
            logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")
    
    return eval_metrics

# --- Baseline Strategy Simulation ---
def run_baseline_strategy(data_df, price_col, strategy_name, config, logger): # Added logger, price_col
    """
    Simulates a baseline trading strategy.
    `data_df` should be the same data the agent was evaluated on.
    `price_col` is the name of the column holding the price data.
    """
    logger.info(f"Running baseline strategy: {strategy_name}...")
    if data_df.empty or price_col not in data_df.columns:
        logger.error(f"Price column '{price_col}' not found or data is empty for baseline: {strategy_name}.")
        return {}

    initial_capital = config['environment_params'].get('initial_capital', 1000000)
    portfolio_values = [initial_capital]
    actions = [] # 0:S, 1:F, 2:L

    if strategy_name == "buy_and_hold":
        price_at_start = data_df[price_col].iloc[0]
        for i in range(len(data_df)):
            current_price = data_df[price_col].iloc[i]
            pv = initial_capital * (current_price / price_at_start) if price_at_start != 0 else initial_capital
            portfolio_values.append(pv)
            actions.append(2) # Always Long
    
    elif strategy_name == "ma_crossover":
        ma_params = config['evaluation_params'].get('ma_crossover_baseline', {})
        short_window = ma_params.get('short_window', 50)
        long_window = ma_params.get('long_window', 200)

        # Use pre-calculated MAs if available (e.g., from feature engineering step)
        # Otherwise, calculate them here.
        short_ma_col = f'sma_{short_window}' # Example naming convention
        long_ma_col = f'sma_{long_window}'   # Example naming convention

        temp_df = data_df.copy() # Avoid modifying original df view
        if short_ma_col not in temp_df.columns:
            logger.info(f"Calculating short MA ({short_window}d) for baseline...")
            temp_df[short_ma_col] = calculate_moving_average(temp_df[price_col], window=short_window)
        if long_ma_col not in temp_df.columns:
            logger.info(f"Calculating long MA ({long_window}d) for baseline...")
            temp_df[long_ma_col] = calculate_moving_average(temp_df[price_col], window=long_window)
        
        temp_df = temp_df.dropna(subset=[short_ma_col, long_ma_col])
        if temp_df.empty:
            logger.warning("Not enough data for MA crossover after dropping NAs from MAs.")
            return {}

        position = 0 
        current_capital = initial_capital
        portfolio_values = [initial_capital] # Reset for this strategy's PV calculation

        for i in range(1, len(temp_df)):
            prev_price = temp_df[price_col].iloc[i-1]
            current_price = temp_df[price_col].iloc[i]
            price_change = current_price - prev_price

            signal_long = temp_df[short_ma_col].iloc[i-1] > temp_df[long_ma_col].iloc[i-1]
            signal_short = temp_df[short_ma_col].iloc[i-1] < temp_df[long_ma_col].iloc[i-1]

            daily_pnl = 0
            if position == 1: daily_pnl = price_change    # Long
            elif position == -1: daily_pnl = -price_change # Short
            
            # Transaction costs for MA crossover (simplified)
            tc = 0
            new_position_target = position
            if signal_long: new_position_target = 1
            elif signal_short: new_position_target = -1
            else: new_position_target = 0 # Go flat if MAs are equal or signal unclear

            if new_position_target != position: # If position changes
                tc = config['environment_params'].get('transaction_cost_pct', 0.0001) * current_price # Simplified cost
            
            current_capital += (daily_pnl - tc)
            portfolio_values.append(current_capital)
            position = new_position_target
            actions.append(position + 1) # Convert -1,0,1 to 0,1,2 for actions list
    
    elif strategy_name == "always_flat":
        portfolio_values = [initial_capital] * (len(data_df) +1) # Match length convention
        actions = [1] * len(data_df)
    else:
        logger.warning(f"Unknown baseline strategy: {strategy_name}")
        return {}

    if not portfolio_values or len(portfolio_values) <= 1:
        return {m: 0 for m in ["total_return_pct", "annualized_return_pct", "annualized_volatility_pct", "sharpe_ratio", "max_drawdown_pct"]} | {"actions": actions, "portfolio_values": portfolio_values}

    portfolio_series = pd.Series(portfolio_values)
    daily_returns_series = portfolio_series.pct_change().dropna()

    baseline_metrics = {}
    if daily_returns_series.empty:
         for m in ["total_return_pct", "annualized_return_pct", "annualized_volatility_pct", "sharpe_ratio", "max_drawdown_pct", "sortino_ratio", "calmar_ratio"]:
             baseline_metrics[m] = 0.0
    else:
        baseline_metrics["total_return_pct"] = metrics_module.total_return(portfolio_series) * 100
        baseline_metrics["annualized_return_pct"] = metrics_module.annualized_return(daily_returns_series) * 100
        baseline_metrics["annualized_volatility_pct"] = metrics_module.annualized_volatility(daily_returns_series) * 100
        baseline_metrics["sharpe_ratio"] = metrics_module.sharpe_ratio(daily_returns_series, risk_free_rate=config['evaluation_params'].get('risk_free_rate', 0.0))
        baseline_metrics["max_drawdown_pct"] = metrics_module.max_drawdown(portfolio_series) * 100
        baseline_metrics["sortino_ratio"] = metrics_module.sortino_ratio(daily_returns_series, required_return=config['evaluation_params'].get('required_return_for_sortino', 0.0))
        baseline_metrics["calmar_ratio"] = metrics_module.calmar_ratio(portfolio_series)

    baseline_metrics["actions"] = actions
    baseline_metrics["portfolio_values"] = portfolio_values

    logger.info(f"--- Baseline: {strategy_name} Performance ---")
    for key, value in baseline_metrics.items():
        if key not in ["actions", "portfolio_values"]:
            logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")
            
    return baseline_metrics

# --- Plotting Function (Example) ---
def plot_equity_curves(results_dict, title="Equity Curves", save_path=None, logger=None):
    plt.figure(figsize=(12, 7))
    for strategy_name, metrics in results_dict.items():
        if "portfolio_values" in metrics and metrics["portfolio_values"]:
            # Ensure portfolio_values is a pandas Series for proper plotting if not already
            pv_series = pd.Series(metrics["portfolio_values"])
            plt.plot(pv_series.index, pv_series, label=strategy_name)
    
    plt.title(title)
    plt.xlabel("Time Steps (or Date if index is datetime)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        if logger: logger.info(f"Equity curve plot saved to {save_path}")
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Agent and Baselines")
    parser.add_argument("--config_path", type=str, default="configs/ppo_treasury_config.yaml", help="Path to YAML config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model (.zip file)")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "test", "validation"], help="Data split to evaluate on")
    # vec_normalize_stats_path can be part of config now
    # parser.add_argument("--vec_normalize_stats_path", type=str, default=None, help="Path to VecNormalize stats")

    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {args.config_path}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Could not load or parse configuration file {args.config_path}: {e}")
        exit(1)

    # --- Setup Logger (Placeholder) ---
    class SimpleLogger: # Replace with actual logger setup
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    logger = SimpleLogger()
    logger.info(f"Configuration loaded from {args.config_path}")
    
    # --- Load Data ---
    data_cfg = config['data']
    price_col = data_cfg['price_column']
    processed_data_file = os.path.join(data_cfg['processed_data_path'], data_cfg.get(f'{args.data_split}_processed_filename', f'{args.data_split}_processed.csv'))
    
    if not os.path.exists(processed_data_file):
        logger.error(f"Processed data file for split '{args.data_split}' not found at {processed_data_file}. Run preprocess_data.py first.")
        exit(1)
    
    logger.info(f"Loading {args.data_split} data from {processed_data_file}...")
    eval_df = pd.read_csv(processed_data_file, index_col='Date', parse_dates=True)
    if eval_df.empty:
        logger.error(f"Loaded data for {args.data_split} is empty.")
        exit(1)

    # --- Environment Setup ---
    env_kwargs = {'data_df': eval_df, 'config': config, 'current_step_in_df': 0}
    eval_env = DummyVecEnv([lambda: TradingEnv(**env_kwargs)]) # No make_vec_env needed for single

    vec_normalize_stats_path = config['training'].get('vec_normalize_stats_name', 'vec_normalize_stats.pkl')
    full_vec_stats_path = os.path.join(config['training'].get('save_path', 'models/'), vec_normalize_stats_path)

    if config['training'].get('normalize_env', False):
        if os.path.exists(full_vec_stats_path):
            logger.info(f"Loading VecNormalize statistics from {full_vec_stats_path}")
            eval_env = VecNormalize.load(full_vec_stats_path, eval_env)
            eval_env.training = False 
            eval_env.norm_reward = False
        else:
            logger.warning(f"VecNormalize stats file not found at {full_vec_stats_path} but normalization was enabled. Evaluating with unnormalized env.")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found at {args.model_path}")
        exit(1)
    logger.info(f"Loading trained model from {args.model_path}...")
    model = PPO.load(args.model_path, env=eval_env) # SB3 handles VecNormalize if env is wrapped
    logger.info("Model loaded successfully.")

    # --- Run Evaluations ---
    all_results = {}
    agent_metrics = evaluate_agent(model, eval_env, config, logger, data_split_name=args.data_split)
    all_results["RL_Agent"] = agent_metrics

    for baseline_name in config['evaluation_params'].get('baseline_strategies', []):
        baseline_metrics = run_baseline_strategy(eval_df.copy(), price_col, baseline_name, config, logger)
        all_results[baseline_name.replace("_", " ").title()] = baseline_metrics
    
    # --- Output Results ---
    logger.info("--- Evaluation Summary ---")
    # Create a DataFrame for summary table
    summary_data = []
    for strategy, mets in all_results.items():
        if mets: # Check if metrics dict is not empty
            summary_data.append({
                "Strategy": strategy,
                "Total Return %": mets.get("total_return_pct", 0),
                "Annualized Return %": mets.get("annualized_return_pct", 0),
                "Annualized Volatility %": mets.get("annualized_volatility_pct", 0),
                "Sharpe Ratio": mets.get("sharpe_ratio", 0),
                "Max Drawdown %": mets.get("max_drawdown_pct", 0)
            })
    summary_df = pd.DataFrame(summary_data)
    logger.info("
" + summary_df.to_string(index=False))

    # Save summary to CSV
    results_path = config['training'].get('log_path', 'logs/') # Use log_path for results too
    os.makedirs(results_path, exist_ok=True)
    summary_filename = os.path.join(results_path, f"evaluation_summary_{args.data_split}.csv")
    summary_df.to_csv(summary_filename, index=False)
    logger.info(f"Evaluation summary saved to {summary_filename}")

    # Plotting
    plot_save_path = os.path.join(results_path, f"equity_curve_{args.data_split}.png")
    plot_equity_curves(all_results, title=f"Equity Curves - {args.data_split.title()} Data", save_path=plot_save_path, logger=logger)

    eval_env.close()
    logger.info("
Evaluation script completed.")
