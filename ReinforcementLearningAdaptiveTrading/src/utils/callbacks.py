import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback
# from stable_baselines3.common.evaluation import evaluate_policy # For custom eval within callback

# Note: For SB3 v2.x, some imports and superclass might differ slightly if using older versions.
# This is generally for SB3 v1.x+ and Gymnasium. For SB3 < v1.0 (legacy), syntax would vary more.

class SaveBestModelCallback(EventCallback):
    """
    Custom callback for saving the best model during training.
    This is an example and can be extended or used in conjunction with SB3's EvalCallback.
    EvalCallback itself provides functionality for saving the best model based on evaluation
    on a separate environment. This custom callback might be useful if you have
    a very specific way of determining the "best" model or want more control.

    If using SB3's EvalCallback, it's often preferred for its robustness and integration.
    This example shows how one might build a custom one if needed.
    """
    def __init__(self, check_env, log_dir: str, model_save_path: str, 
                 check_freq: int = 10000, n_eval_episodes: int = 5,
                 deterministic: bool = True, verbose: int = 1):
        super(SaveBestModelCallback, self).__init__(callback=None, verbose=verbose) # BaseCallback then EventCallback
        
        self.check_env = check_env # The environment to use for evaluation
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        
        self.best_mean_reward = -np.inf # Initialize with a very low value

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.model_save_path is not None:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an EventCallback), this will be called
        when the event triggers.
        """
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print(f"Running evaluation at training step {self.num_timesteps}...")
            
            # Using a separate evaluation environment (self.check_env)
            # This is a simplified evaluation. SB3's evaluate_policy is more robust.
            # For simplicity, mimicking a basic evaluation loop here.
            
            current_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.check_env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, done, _ = self.check_env.step(action)
                    episode_reward += reward
                current_rewards.append(episode_reward)
            
            mean_reward = np.mean(current_rewards)
            
            if self.verbose > 0:
                print(f"Evaluation: Mean reward over {self.n_eval_episodes} episodes: {mean_reward:.2f}")
            
            # Log this mean reward (e.g., to TensorBoard)
            if self.logger is not None: # self.logger comes from BaseCallback
                self.logger.record("eval/custom_mean_reward", mean_reward)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.model_save_path is not None:
                    save_path = os.path.join(self.model_save_path, f"best_model_step_{self.num_timesteps}.zip")
                    self.model.save(save_path)
                    if self.verbose > 0:
                        print(f"New best model saved to {save_path} with mean reward: {mean_reward:.2f}")
            else:
                if self.verbose > 0:
                    print(f"Current mean reward {mean_reward:.2f} did not beat best {self.best_mean_reward:.2f}")

        return True # Continue training

# Other potential custom callbacks:
# - EarlyStoppingCallback: Stop training if performance on a validation set plateaus.
#   (SB3 has a built-in EarlyStopping callback that can be used with EvalCallback)
# - LoggingCallback: For logging custom metrics or environment-specific information
#   to TensorBoard or other logging frameworks.
# - RewardShapingCallback: Dynamically adjust reward shaping parameters during training. (Advanced)


class LoggingCallback(BaseCallback):
    """
    A custom callback to log additional information during training.
    For example, log portfolio value from the info dict if the environment provides it.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # `self.training_env` is available. If it's a VecEnv, `buf_infos` stores info dicts.
            if self.training_env is not None and hasattr(self.training_env, 'buf_infos'):
                # Example: Log portfolio value if it's in the info dict
                # buf_infos is a list of dicts, one for each parallel environment
                # We can log the mean, or for the first env, etc.
                
                portfolio_values = [info.get('portfolio_value') for info in self.training_env.buf_infos if 'portfolio_value' in info and info.get('portfolio_value') != 'not_logged']
                if portfolio_values:
                    mean_portfolio_value = np.mean(portfolio_values)
                    self.logger.record("custom/mean_portfolio_value", mean_portfolio_value)

                # Example: Log drawdown
                drawdowns = [info.get('drawdown_pct') for info in self.training_env.buf_infos if 'drawdown_pct' in info]
                if drawdowns:
                    mean_drawdown = np.mean(drawdowns)
                    self.logger.record("custom/mean_drawdown_pct", mean_drawdown)
            
            # Log other things from self.locals or self.globals if needed
            # self.logger.record("custom/learning_rate", self.model.learning_rate) # If lr is static attribute
            # Note: Accessing learning rate might need a schedule check if it's dynamic.


        return True


if __name__ == '__main__':
    print("This is src/utils/callbacks.py")
    print("This file is intended for custom Stable Baselines3 callbacks.")
    print("Examples included: SaveBestModelCallback (conceptual), LoggingCallback.")
    print("\nNote: For saving the best model, SB3's `EvalCallback` is often recommended and provides robust functionality.")
    
    # To use EvalCallback (recommended way for saving best model):
    # from stable_baselines3.common.callbacks import EvalCallback
    # eval_env = YourTradingEnv(...) # A separate instance for evaluation
    # eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_sb3/',
    #                              log_path='./logs/sb3_eval_results/', eval_freq=5000,
    #                              deterministic=True, render=False)
    # model.learn(total_timesteps=100000, callback=[eval_callback, other_custom_callbacks])
