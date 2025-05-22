import argparse
import os
import yaml # For loading YAML config
import pandas as pd # Added for placeholder data
import numpy as np  # Added for placeholder data

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback # Using SB3's EvalCallback

# Adjusted import paths assuming train.py might be in 'scripts/' later,
# and 'src' is the root for these modules.
# This might require PYTHONPATH setup or installing the src package.
# For now, if train.py remains in src/, these are:
# from environment.trading_env import TradingEnv
# from data.data_loader import load_data, split_data
# from data.feature_engineering import engineer_features 
# from utils.config_loader import load_config # Will be created later
# from utils.logger import setup_logger # Will be created later

# TEMPORARY: Direct imports assuming train.py is in src/ for now
# These will be updated once directory structure is finalized and utils are created.
from environment.trading_env import TradingEnv
from data_processing.data_loader import load_data, split_data # Assuming move to data_processing
from data_processing.feature_engineer import engineer_features # Assuming move to data_processing
# from utils.callbacks import LoggingCallback # Assuming this custom one might still be used

# Placeholder for logger - will be replaced by proper logger from utils.logger
# logger = print # Temporary redirect, replace with actual logger object

# --- Main Training Function ---
def train_agent(config, logger): # Added logger as argument
    """
    Main function to train the RL agent.
    """
    logger.info("Starting training process...")

    # --- Data Loading ---
    # This section will use data_loader.py and feature_engineer.py
    # It should load preprocessed data if available, or run preprocessing.
    # For now, using placeholder logic.
    processed_train_path = os.path.join(config['data']['processed_data_path'], config['data'].get('train_processed_filename', 'train_processed.csv'))
    processed_val_path = os.path.join(config['data']['processed_data_path'], config['data'].get('val_processed_filename', 'val_processed.csv'))

    if os.path.exists(processed_train_path) and os.path.exists(processed_val_path):
        logger.info(f"Loading preprocessed training data from: {processed_train_path}")
        train_df = pd.read_csv(processed_train_path, index_col='Date', parse_dates=True)
        logger.info(f"Loading preprocessed validation data from: {processed_val_path}")
        val_df = pd.read_csv(processed_val_path, index_col='Date', parse_dates=True)
    else:
        logger.warning("Preprocessed data not found. Using dummy data for training. Run preprocess_data.py first.")
        # Create dummy data if preprocessed files don't exist
        dummy_dates_train = pd.to_datetime([f'2000-01-{i+1:02d}' for i in range(200)])
        dummy_data_train = {'price': np.random.rand(200) * 100 + 1000}
        for feature_name in config['data'].get('feature_columns', []):
            dummy_data_train[feature_name] = np.random.rand(200)
        train_df = pd.DataFrame(dummy_data_train, index=dummy_dates_train)

        dummy_dates_val = pd.to_datetime([f'2001-01-{i+1:02d}' for i in range(50)])
        dummy_data_val = {'price': np.random.rand(50) * 100 + 1100}
        for feature_name in config['data'].get('feature_columns', []):
            dummy_data_val[feature_name] = np.random.rand(50)
        val_df = pd.DataFrame(dummy_data_val, index=dummy_dates_val)
        logger.info(f"Using dummy training data ({len(train_df)} rows) and validation data ({len(val_df)} rows).")


    # --- Environment Setup ---
    logger.info(f"Using {config['agent'].get('n_envs', 1)} parallel environments for training.")
    
    train_env_kwargs = {'data_df': train_df, 'config': config, 'current_step_in_df': 0}
    train_env = make_vec_env(
        TradingEnv,
        n_envs=config['agent'].get('n_envs', 1),
        env_kwargs=train_env_kwargs,
        vec_env_cls=DummyVecEnv
    )

    eval_env_kwargs = {'data_df': val_df, 'config': config, 'current_step_in_df': 0}
    eval_env = make_vec_env(
        TradingEnv, # Create a single env for evaluation
        n_envs=1,
        env_kwargs=eval_env_kwargs,
        vec_env_cls=DummyVecEnv
    )

    if config['training'].get('normalize_env', False):
        logger.info("Normalizing training and evaluation environments.")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=config['agent'].get('gamma', 0.99))
        # Important: Save the VecNormalize statistics from train_env
        # Load them onto eval_env but do not update them, and do not normalize reward
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, gamma=config['agent'].get('gamma', 0.99))


    # --- Agent Initialization ---
    # PPO hyperparameters are taken from the 'agent' section of the config
    ppo_hyperparams = config['agent'].copy()
    # Remove non-PPO specific keys if any, or ensure PPO constructor handles extra keys
    ppo_hyperparams.pop('n_envs', None) # n_envs is for make_vec_env

    model = PPO(
        policy=ppo_hyperparams.pop('policy', 'MlpPolicy'), # Get policy, remove from dict
        env=train_env,
        tensorboard_log=os.path.join(config['training'].get('log_path', 'logs/'), 'tensorboard'),
        verbose=config['training'].get('verbose', 1),
        seed=config['training'].get('seed', None),
        **ppo_hyperparams # Pass the rest of PPO params
    )
    logger.info("PPO Model initialized.")
    logger.info(f"Policy network architecture: {model.policy}")

    # --- Callbacks ---
    callbacks_list = []
    
    # Checkpoint callback
    checkpoint_save_path = os.path.join(config['training'].get('save_path', 'models/'), 'checkpoints')
    os.makedirs(checkpoint_save_path, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(config['training'].get('checkpoint_save_freq', 100000) // config['agent'].get('n_envs',1), 1),
        save_path=checkpoint_save_path,
        name_prefix="ppo_trading_checkpoint"
    )
    callbacks_list.append(checkpoint_cb)

    # EvalCallback for saving best model and early stopping (optional)
    best_model_save_path = os.path.join(config['training'].get('save_path', 'models/'), 'best_model')
    os.makedirs(best_model_save_path, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=os.path.join(config['training'].get('log_path', 'logs/'), 'eval_results'),
        eval_freq=max(config['training'].get('eval_freq', 50000) // config['agent'].get('n_envs',1), 1),
        n_eval_episodes=config['training'].get('eval_episodes', 5),
        deterministic=config['training'].get('eval_deterministic', True),
        render=False,
        # callback_on_new_best= (optional: another callback here if best model found)
        # stop_train_callback for early stopping (optional)
    )
    callbacks_list.append(eval_callback)
    
    # Add custom logging callback if defined and needed
    # custom_log_cb = LoggingCallback(log_freq=1000)
    # callbacks_list.append(custom_log_cb)

    # --- Training ---
    total_timesteps = int(config['training'].get('total_timesteps', 1e6))
    logger.info(f"Starting training for {total_timesteps} total timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks_list if callbacks_list else None,
            tb_log_name=config['training'].get('tb_log_name', "PPO_TradingAgent")
        )
        logger.info("Training finished.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Potentially save the model even if training is interrupted
        interrupted_model_path = os.path.join(config['training'].get('save_path', 'models/'), "ppo_trading_agent_interrupted.zip")
        model.save(interrupted_model_path)
        logger.info(f"Interrupted model saved to {interrupted_model_path}")
        if config['training'].get('normalize_env', False) and hasattr(train_env, 'save'):
            interrupted_stats_path = os.path.join(config['training'].get('save_path', 'models/'), "vec_normalize_stats_interrupted.pkl")
            train_env.save(interrupted_stats_path)
            logger.info(f"VecNormalize stats for interrupted model saved to {interrupted_stats_path}")
        raise

    # --- Save Final Model and Normalization Stats ---
    final_model_path = os.path.join(config['training'].get('save_path', 'models/'), config['training'].get('final_model_name', "ppo_trading_agent_final.zip"))
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    if config['training'].get('normalize_env', False) and hasattr(train_env, 'save'):
        normalize_stats_path = os.path.join(config['training'].get('save_path', 'models/'), config['training'].get('vec_normalize_stats_name', "vec_normalize_stats.pkl"))
        train_env.save(normalize_stats_path) # Save stats from the training environment
        logger.info(f"VecNormalize statistics saved to {normalize_stats_path}")

    train_env.close()
    eval_env.close()
    logger.info("Training script completed.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent for Adaptive Trading")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/ppo_treasury_config.yaml", # Default to YAML in configs/
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    # This will be replaced by: from utils.config_loader import load_config
    # config = load_config(args.config_path) 
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {args.config_path}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Could not load or parse configuration file {args.config_path}: {e}")
        exit(1)

    # --- Setup Logger ---
    # This will be replaced by: from utils.logger import setup_logger
    # logger = setup_logger(config['logging'])
    # For now, using print as a basic logger.
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    logger = SimpleLogger()
    logger.info(f"Configuration loaded from {args.config_path}")


    # --- Ensure essential paths from config exist ---
    os.makedirs(config['training'].get('save_path', 'models/'), exist_ok=True)
    os.makedirs(os.path.join(config['training'].get('log_path', 'logs/'), 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(config['training'].get('log_path', 'logs/'), 'eval_results'), exist_ok=True)
    os.makedirs(config['data'].get('processed_data_path', 'data/processed/'), exist_ok=True)


    train_agent(config, logger)
