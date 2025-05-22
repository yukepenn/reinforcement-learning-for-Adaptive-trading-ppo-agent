# ReinforcementLearningAdaptiveTrading/src/config.py

# This file centralizes configuration settings and hyperparameters for the project.
# By adjusting values here, one can easily tune experiments without modifying core code.

DEFAULT_CONFIG = {
    "project_name": "ReinforcementLearningAdaptiveTrading",
    "random_seed": 42, # For reproducibility

    "data": {
        "raw_data_path": "data/raw/your_raw_data.csv", # Placeholder - replace with actual path or loading mechanism
        "processed_data_path": "data/processed/", # Path to save/load processed data
        "feature_data_path": "data/features/",   # Path to save/load feature data
        
        # Define periods for splitting data
        "train_period": {"start": "2000-01-01", "end": "2015-12-31"},
        "validation_period": {"start": "2014-01-01", "end": "2015-12-31"}, # Example validation period, can overlap with train
        "test_period": {"start": "2016-01-01", "end": "2023-12-31"},
        
        # List of columns to be used as features from the feature-engineered dataframe
        # These should match the output of feature_engineering.py
        "feature_columns": [
            "yield_curve_slope", 
            "volatility_20d", 
            "ma_crossover_signal", # e.g., 50d_ma - 200d_ma
            "momentum_1m",
            # "rsi_14d", # Example additional feature
            # "current_position" # Will be added by the environment if include_position_in_state=True
        ],
        "price_column": "price" # Name of the column representing the futures price
    },

    "environment_params": {
        "initial_capital": 1000000.0, # Starting portfolio value
        "transaction_cost_pct": 0.0001, # Transaction cost as a percentage of trade value (0.01%)
        "max_drawdown_pct": 0.20, # Maximum allowable drawdown (e.g., 20%) for stop-loss
        "stop_loss_penalty": -1.0, # Large negative reward if stop-loss is hit
        
        # Reward shaping coefficients (lambda values)
        "reward_lambda_pnl": 1.0,          # Weight for profit/loss component
        "reward_lambda_volatility": 0.05,  # Penalty weight for volatility
        "reward_lambda_drawdown": 0.1,   # Penalty weight for drawdown
        
        "include_position_in_state": True, # Whether to include current position as a feature
        "episode_max_steps": None, # Max steps per episode. If None, runs through the whole dataset.
                                   # Can be set to e.g., 252 for yearly episodes.
        "log_portfolio_value": True # Whether the env should log portfolio value in info dict
    },

    "ppo_params": { # Hyperparameters for the PPO algorithm from Stable Baselines3
        "n_envs": 4, # Number of parallel environments for training
        "total_timesteps": 1_000_000, # Total timesteps for training (e.g., 1e6)
        "learning_rate": 3e-4, # Learning rate (can be a float or a schedule)
        "n_steps": 2048, # Number of steps to run for each environment per update
        "batch_size": 64, # Minibatch size for PPO updates
        "n_epochs": 10, # Number of epochs when optimizing the surrogate loss
        "gamma": 0.99, # Discount factor for future rewards
        "gae_lambda": 0.95, # Factor for trade-off of bias vs variance for GAE
        "clip_range": 0.2, # Clipping parameter for PPO (can be a float or a schedule)
        "ent_coef": 0.01, # Entropy coefficient for exploration
        "vf_coef": 0.5, # Value function coefficient for the loss calculation
        "max_grad_norm": 0.5, # Maximum norm for gradient clipping
        
        # For MlpPolicy, network architecture can be specified if needed
        # "policy_kwargs": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]), # Example: 2 layers of 64 units
    },

    "training_params": {
        "verbose": 1, # Verbosity level (0: no output, 1: info, 2: debug)
        "seed": 42, # Seed for reproducibility during training
        "tensorboard_log_path": "logs/tensorboard/", # Path to save TensorBoard logs
        "tb_log_name": "PPO_TradingAgent_v1", # Name for the TensorBoard log
        "model_save_path": "models/", # Directory to save trained models and checkpoints
        "final_model_name": "ppo_trading_agent_final.zip",
        "best_model_name": "ppo_trading_agent_best.zip",
        "checkpoint_save_freq": 100000, # Save a checkpoint every N timesteps (across all envs)
        "normalize_env": True, # Whether to use VecNormalize for observations and rewards
        "vec_normalize_stats_name": "vec_normalize_stats.pkl", # Filename for VecNormalize stats

        # Early stopping parameters (if using a callback for it)
        # "early_stopping_patience": 10, # Number of evaluations to wait for improvement
        # "early_stopping_metric": "eval/mean_reward", # Metric to monitor for early stopping
        
        # Evaluation parameters for callbacks (e.g., EvalCallback)
        "eval_freq": 20000, # Evaluate the agent every N timesteps (across all envs)
        "eval_n_episodes": 5, # Number of episodes to run for evaluation
        # "eval_deterministic": True, # Whether to use deterministic actions for evaluation
    },
    
    "evaluation_params": {
        "data_split_to_evaluate": "test", # "train", "validation", or "test"
        # Baselines can have their own params if needed
        "baseline_strategies": ["buy_and_hold", "ma_crossover", "always_flat"],
        "ma_crossover_baseline": { # Parameters for the MA crossover baseline
            "short_window": 50,
            "long_window": 200
        }
    },

    "explainability_params": {
        "shap_nsamples": 100, # Number of samples for SHAP background dataset / explanations
        "shap_plot_path": "figures/shap_summary.png"
    },
    
    "logging": {
        "log_level": "INFO", # "DEBUG", "INFO", "WARNING", "ERROR"
        "log_file": "logs/app.log"
    }
}

# Example of how to add comments for hyperparameter choices:
# PPO Hyperparameters Justification:
# - n_envs=4: Utilizes multiple CPU cores for faster data collection.
# - total_timesteps=1e6: A common starting point for moderately complex RL tasks.
# - learning_rate=3e-4: Default Adam optimizer learning rate, often works well for PPO.
# - n_steps=2048: PPO default, provides a good batch of experiences for updates.
# - batch_size=64: Smaller batches for more frequent updates within an epoch.
# - n_epochs=10: PPO default, multiple passes over collected data to improve policy.
# - gamma=0.99: Standard discount factor, considers long-term rewards (appropriate for daily trading).
# - gae_lambda=0.95: Standard GAE parameter, balances bias and variance in advantage estimation.
# - clip_range=0.2: PPO default, prevents excessively large policy updates, ensuring stability.
# - ent_coef=0.01: Small entropy bonus to encourage exploration, preventing premature convergence.

if __name__ == '__main__':
    # Example of accessing configuration values
    print(f"Training timesteps: {DEFAULT_CONFIG['ppo_params']['total_timesteps']}")
    print(f"Initial capital for environment: {DEFAULT_CONFIG['environment_params']['initial_capital']}")
    print(f"Feature columns: {DEFAULT_CONFIG['data']['feature_columns']}")
    print(f"MA Crossover Short Window (Baseline): {DEFAULT_CONFIG['evaluation_params']['ma_crossover_baseline']['short_window']}")
