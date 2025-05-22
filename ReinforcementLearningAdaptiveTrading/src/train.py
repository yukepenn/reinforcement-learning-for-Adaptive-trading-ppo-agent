import argparse
import os
import yaml # Or use json/configparser if preferred for config
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback # Import necessary callbacks

# Assuming other necessary modules will be created:
from environment.trading_env import TradingEnv
from data.data_loader import load_data, split_data
from utils.callbacks import SaveBestModelCallback # Assuming this will be a custom callback
from config import DEFAULT_CONFIG # Assuming config.py will have this

def train_agent(config):
    """
    Main function to train the RL agent.
    Sets up the environment, model, and initiates training.
    """
    print("Starting training process...")

    # Load data
    # This is a placeholder. Actual data loading will depend on data_loader.py implementation
    # For example:
    # raw_data = load_data(config['data']['raw_data_path'])
    # train_df, _ = split_data(raw_data, config['data']['train_period'], config['data']['test_period'])
    # features_df = some_feature_engineering_function(train_df) # From feature_engineering.py
    
    # Placeholder for data - replace with actual data loading and feature engineering
    # For the environment to initialize, it needs data.
    # This part will need to be more robust once data_loader and feature_engineering are implemented.
    print("Placeholder: Data loading and feature engineering would happen here.")
    # Example: create dummy data for now if TradingEnv requires it for initialization
    import pandas as pd
    import numpy as np
    dummy_dates = pd.to_datetime([f'2000-01-{i+1:02d}' for i in range(100)])
    dummy_data_dict = {'price': np.random.rand(100) * 100 + 1000}
    # Add dummy features that TradingEnv might expect based on config.py
    for feature_name in config.get('environment_params', {}).get('feature_columns', []):
        dummy_data_dict[feature_name] = np.random.rand(100)
    
    train_df = pd.DataFrame(dummy_data_dict, index=dummy_dates)
    
    print(f"Using {config.get('ppo_params', {}).get('n_envs', 1)} parallel environments.")

    # Create vectorized environments
    # env_kwargs will pass data and config to each TradingEnv instance
    env_kwargs = {'data': train_df, 'config': config}
    env = make_vec_env(
        TradingEnv,
        n_envs=config.get('ppo_params', {}).get('n_envs', 1),
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv # Use DummyVecEnv for single-process vectorization
    )

    # Optional: Wrap with VecNormalize for observation and reward normalization
    if config.get('training_params', {}).get('normalize_env', False):
        print("Normalizing environment observations and rewards.")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=config.get('ppo_params',{}).get('gamma', 0.99))
        # Important: Save the VecNormalize statistics when saving the model
        # And load them when evaluating or resuming training

    # Define the PPO model
    # policy_kwargs can be used for custom network architectures if needed
    # For now, using default 'MlpPolicy'
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.get('ppo_params', {}).get('learning_rate', 3e-4),
        n_steps=config.get('ppo_params', {}).get('n_steps', 2048),
        batch_size=config.get('ppo_params', {}).get('batch_size', 64),
        n_epochs=config.get('ppo_params', {}).get('n_epochs', 10),
        gamma=config.get('ppo_params', {}).get('gamma', 0.99),
        gae_lambda=config.get('ppo_params', {}).get('gae_lambda', 0.95),
        clip_range=config.get('ppo_params', {}).get('clip_range', 0.2),
        ent_coef=config.get('ppo_params', {}).get('ent_coef', 0.0),
        vf_coef=config.get('ppo_params', {}).get('vf_coef', 0.5),
        max_grad_norm=config.get('ppo_params', {}).get('max_grad_norm', 0.5),
        tensorboard_log=config.get('training_params', {}).get('tensorboard_log_path', None),
        verbose=config.get('training_params', {}).get('verbose', 1),
        seed=config.get('training_params', {}).get('seed', None),
        # policy_kwargs=config.get('policy_kwargs', None) # For custom network architecture
    )

    print("PPO Model initialized.")
    print(f"Policy network architecture: {model.policy}")

    # Callbacks
    callbacks = []
    
    # Checkpoint callback to save model periodically
    checkpoint_cb = CheckpointCallback(
        save_freq=max(config.get('training_params',{}).get('save_freq', 100000) // config.get('ppo_params',{}).get('n_envs',1), 1),
        save_path=config.get('training_params',{}).get('model_save_path', './models/checkpoints/'),
        name_prefix="ppo_trading_agent_checkpoint"
    )
    callbacks.append(checkpoint_cb)

    # Custom callback for saving the best model (based on evaluation)
    # This requires a proper evaluation environment setup.
    # For now, this is a placeholder.
    # eval_env = TradingEnv(some_validation_data, config) # Needs validation data
    # eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model/',
    #                              log_path='./logs/results/', eval_freq=500,
    #                              deterministic=True, render=False)
    # callbacks.append(eval_callback)

    # Example of a custom SaveBestModelCallback (if defined in utils.callbacks)
    # save_best_model_cb = SaveBestModelCallback(check_freq=..., log_dir=..., model_save_path=...)
    # callbacks.append(save_best_model_cb)


    # Train the agent
    print(f"Training for {config.get('ppo_params', {}).get('total_timesteps', 1e6)} total timesteps...")
    try:
        model.learn(
            total_timesteps=int(config.get('ppo_params', {}).get('total_timesteps', 1e6)),
            callback=callbacks if callbacks else None,
            tb_log_name=config.get('training_params', {}).get('tb_log_name', "PPO_TradingAgent")
        )
        print("Training finished.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Potentially save the model even if training is interrupted
        # save_model(model, os.path.join(config.get('training_params',{}).get('model_save_path', './models/'), "ppo_trading_agent_interrupted.zip"), env if config.get('training_params',{}).get('normalize_env', False) else None)
        raise

    # Save the final model
    final_model_path = os.path.join(config.get('training_params',{}).get('model_save_path', './models/'), config.get('training_params',{}).get('final_model_name', "ppo_trading_agent_final.zip"))
    save_model(model, final_model_path, env if config.get('training_params',{}).get('normalize_env', False) else None)
    
    # If using VecNormalize, save the running averages
    if config.get('training_params', {}).get('normalize_env', False) and hasattr(env, 'save'):
        normalize_stats_path = os.path.join(config.get('training_params',{}).get('model_save_path', './models/'), "vec_normalize_stats.pkl")
        env.save(normalize_stats_path)
        print(f"VecNormalize statistics saved to {normalize_stats_path}")

    env.close()
    print("Training script completed.")


def validate_agent(model, env):
    """
    (Optional) Validate the agent on a validation set during training.
    This function would be called by a callback.
    """
    # This is a conceptual placeholder.
    # Actual validation would involve running the model on a validation dataset
    # and computing metrics like Sharpe ratio or cumulative return.
    # Stable Baselines3 EvalCallback can handle this more systematically.
    print("Validating agent (placeholder)...")
    # Example:
    # obs = env.reset()
    # total_reward = 0
    # for _ in range(len(env.df) -1): # Assuming env.df is validation data
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = env.step(action)
    #     total_reward += rewards
    #     if dones:
    #         break
    # print(f"Validation episode reward: {total_reward}")
    # return total_reward # Or other relevant metric


def save_model(model, path, vec_normalize_env=None):
    """
    Save the trained model.
    If VecNormalize was used, its statistics should also be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")

    # If VecNormalize is used, it's recommended to save its statistics separately
    # as SB3's model.save() doesn't always handle it perfectly for continued training/evaluation.
    # However, for PPO, SB3 typically bundles it if the environment passed to model.save() is the VecNormalized one.
    # The explicit save in train_agent is a good practice.


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Adaptive Trading")
    parser.add_argument("--config_path", type=str, default="src/config.py", help="Path to the configuration file")
    # Allow overriding specific config values via CLI if needed, e.g.:
    # parser.add_argument("--learning_rate", type=float, help="Override learning rate")

    args = parser.parse_args()

    # Load configuration
    # This is a simplified way to load config. If config.py defines a dictionary,
    # we can import it directly. If it's YAML/JSON, we parse it.
    
    config = {}
    if args.config_path.endswith('.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = getattr(config_module, 'DEFAULT_CONFIG', {}) # Assuming config.py has DEFAULT_CONFIG
        # Potentially update config with CLI args here if needed
    elif args.config_path.endswith(('.yaml', '.yml')):
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Unsupported configuration file format: {args.config_path}. Using default config.")
        config = DEFAULT_CONFIG # Fallback

    # Ensure essential paths exist
    if 'model_save_path' in config.get('training_params', {}):
        os.makedirs(config['training_params']['model_save_path'], exist_ok=True)
    if 'tensorboard_log_path' in config.get('training_params', {}):
        os.makedirs(config['training_params']['tensorboard_log_path'], exist_ok=True)


    train_agent(config)

if __name__ == "__main__":
    main()
