# Reinforcement Learning for Adaptive Trading (PPO Agent)

## Project Overview

**Reinforcement Learning for Adaptive Trading (PPO Agent)** is a solo project that trains a reinforcement learning agent to trade a 10-year Treasury futures contract by going **long**, **short**, or **flat** (no position) at each time step. The objective is to maximize **risk-adjusted returns** (e.g. high Sharpe ratio) rather than just raw profit. The agent is trained with the Proximal Policy Optimization (**PPO**) algorithm, a stable actor-critic RL method known for clipping policy updates to ensure training stability.

This project demonstrates:
-   **Deep Reinforcement Learning Expertise:** Custom `gymnasium.Env` environment, policy/value network understanding (even if using SB3 defaults), vectorized training concepts.
-   **Financial Domain Knowledge:** Use of relevant features like yield curve slope, volatility, and trend indicators for trading financial instruments.
-   **Software Engineering Best Practices:** Modular design, clear code, version control, dependency management, configuration management, and comprehensive documentation.
-   **Explainability:** Techniques like SHAP for interpreting policy decisions.

The agent is trained on historical treasury market data (e.g., 2000–2015) and evaluated on out-of-sample data (e.g., 2016–2023) against baseline strategies.

## Repository Structure

The project follows a professional repository structure:

```
ReinforcementLearningAdaptiveTrading/
├── README.md
├── requirements.txt
├── configs/
│   └── ppo_treasury_config.yaml
├── data/
│   ├── raw/              # Raw, untouched data (e.g., 10Y_future_prices.csv, yield_data.csv)
│   │   └── .gitkeep
│   ├── processed/        # Cleaned and feature-engineered data for training/evaluation
│   │   └── .gitkeep
│   └── features/         # Optional: Intermediate feature storage or scalers
│       └── .gitkeep
├── notebooks/
│   ├── 01_DataExploration.ipynb
│   ├── 02_FeatureEngineeringExperiments.ipynb
│   ├── 03_TrainingExperiments.ipynb
│   └── 04_PerformanceEvaluation.ipynb
├── scripts/
│   ├── preprocess_data.py # Script to process raw data into training/test sets
│   ├── train.py           # Main script to train the RL agent
│   ├── evaluate.py        # Script to evaluate a trained agent and baselines
│   └── tune_hyperparameters.py # Script for Optuna hyperparameter tuning (optional)
├── src/
│   ├── __init__.py
│   ├── data_processing/    # Modules for data loading and feature engineering
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── feature_engineer.py
│   ├── environments/       # Custom Gym trading environment
│   │   ├── __init__.py
│   │   └── trading_env.py
│   ├── agents/             # RL agent model definitions or wrappers
│   │   ├── __init__.py
│   │   └── ppo_model.py    # Wrapper for SB3 PPO or custom PPO implementation
│   ├── utils/              # Utility modules
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── callbacks.py
│   │   └── plotting.py     # Optional: for standardized plots
│   └── evaluation/         # Modules for backtesting and performance analysis (optional)
│       ├── __init__.py
│       └── performance_analyzer.py
├── models/                 # Saved model artifacts (e.g., best_model.zip)
│   └── .gitkeep
├── logs/                   # Training logs, application logs, TensorBoard logs
│   └── .gitkeep
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_trading_env.py
│   ├── test_data_loader.py
│   ├── test_feature_engineer.py
│   └── test_baselines.py   # Or test_performance_analyzer.py
└── .gitignore
```

## Setup Instructions

### 1. Python Version
This project is developed with **Python 3.9+**. Please ensure you have a compatible Python version installed.

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

**Using `venv`:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using `conda`:**
```bash
conda create -n rl_trading_env python=3.9
conda activate rl_trading_env
```

### 3. Install Dependencies
Once your virtual environment is activated, install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Note: `requirements.txt` will be created in a subsequent step).*

## Data Acquisition and Preparation

### Data Sources
The primary instrument for this project is the **10-year Treasury futures contract**. You will also need historical data for features, such as:
-   **10-Year Treasury Futures Prices:** Daily Open, High, Low, Close, Volume. Possible sources include:
    -   CME DataMine (paid)
    -   Quandl/Nasdaq Data Link (some free, some paid)
    -   Third-party financial data providers (e.g., Alpha Vantage, IEX Cloud, Polygon.io - check their terms for futures data).
    -   For academic purposes, historical data might be available through university libraries or specific datasets.
-   **US Treasury Yields:** Daily yields for various maturities (e.g., 2-year and 10-year) to calculate the yield curve slope. Sources:
    -   Federal Reserve Economic Data (FRED)
    -   U.S. Department of the Treasury website

### Expected Format
-   Data should ideally be in **CSV format**.
-   Required columns for price data (example): `Date, Open, High, Low, Close, Volume`.
-   Required columns for yield data (example): `Date, 2Y_Yield, 10Y_Yield`.

### Preparation Steps
1.  Place your raw CSV files into the `data/raw/` directory. For example:
    -   `data/raw/10Y_futures_prices.csv`
    -   `data/raw/treasury_yields.csv`
2.  Update the file paths and relevant column names in the configuration file (`configs/ppo_treasury_config.yaml`) under the `data` section.
3.  Run the preprocessing script to clean the data, merge sources (if necessary), engineer features, and save the processed dataset to `data/processed/`:
    ```bash
    python scripts/preprocess_data.py --config configs/ppo_treasury_config.yaml
    ```
    This script will use `src.data_processing.data_loader` and `src.data_processing.feature_engineer`.

## Configuration Management

Project parameters are managed through YAML configuration files located in the `configs/` directory. The primary configuration file is `configs/ppo_treasury_config.yaml`.

This file allows you to adjust parameters related to:
-   **Data:** File paths, date ranges for train/validation/test splits, feature names, price column.
-   **Environment:** Initial capital, transaction costs, reward shaping coefficients, stop-loss parameters, observation window size.
-   **Agent (PPO):** Learning rate, batch size, PPO-specific hyperparameters (`n_steps`, `gamma`, `gae_lambda`, `clip_range`, etc.).
-   **Training:** Total timesteps, logging intervals, evaluation frequency, model save paths, TensorBoard log paths.

To modify experiment settings, edit the `ppo_treasury_config.yaml` file. The scripts load their configurations from this file.

## How to Run

### 1. Preprocess Data
(If not already done, see "Data Acquisition and Preparation")
```bash
python scripts/preprocess_data.py --config configs/ppo_treasury_config.yaml
```

### 2. Train the Agent
To train the PPO agent:
```bash
python scripts/train.py --config configs/ppo_treasury_config.yaml
```
-   This script will load the processed data, set up the `TradingEnv`, initialize the PPO agent (from `src.agents.ppo_model`), and start the training process.
-   Progress will be logged to the console and TensorBoard (if configured).
-   The best model (based on evaluation against a validation set/period) and the final model will be saved in the `models/` directory.

### 3. Evaluate a Trained Agent
To evaluate a trained model (e.g., `models/best_model.zip`):
```bash
python scripts/evaluate.py --model_path models/best_model.zip --config configs/ppo_treasury_config.yaml
```
-   This script loads the specified model and evaluates it on the test dataset (defined in the config).
-   It calculates and prints performance metrics (e.g., Sharpe ratio, max drawdown).
-   It may generate plots such as equity curves comparing the agent to baseline strategies.

### 4. Hyperparameter Tuning (Optional)
If you wish to perform hyperparameter optimization using Optuna:
```bash
python scripts/tune_hyperparameters.py --config configs/ppo_treasury_config.yaml --n_trials 100
```
-   This script will search for optimal hyperparameters for the PPO agent based on performance on a validation set.

## Expected Output
-   **Models:** Trained models are saved in the `models/` directory (e.g., `best_model.zip`, `final_model.zip`).
-   **Logs:**
    -   Console output during training and evaluation.
    -   Application logs in `logs/app.log` (or similar, configured in `logger.py`).
    -   TensorBoard logs in `logs/tensorboard/` for visualizing training progress.
-   **Results/Plots:** Evaluation scripts may save plots (e.g., equity curves, feature importance) to a `results/` or `figures/` directory (this needs to be implemented in `evaluate.py` or notebooks).
-   **Processed Data:** The `scripts/preprocess_data.py` script saves feature-engineered datasets to `data/processed/`.

## Detailed Project Components

*(This section will be populated with the detailed descriptions for each module from the original project brief, slightly adapted to the new structure where necessary.)*

### Source Code (`src/`)

#### `src/data_processing/`
-   **`data_loader.py`**: Contains functions to load raw market data (prices, yields) from CSV files or other sources. Handles date parsing and initial data cleaning.
-   **`feature_engineer.py`**: Defines functions to compute financial features (yield curve slope, volatility, trend indicators, etc.) from the raw data. Includes logic for scaling/normalizing features.

#### `src/environments/`
-   **`trading_env.py`**: Defines the custom `TradingEnv` class (subclass of `gymnasium.Env`). This class simulates the trading of 10Y Treasury futures, including:
    -   State representation (market features, current position).
    -   Action space (Short, Flat, Long).
    -   Step logic (trade execution, transaction costs).
    -   Custom reward function (P&L, penalties for volatility/drawdown, stop-loss).
    -   Episode management.

#### `src/agents/`
-   **`ppo_model.py`**: Contains a wrapper class for the Stable Baselines3 PPO model or a custom PPO implementation. Handles model initialization, training, prediction, saving, and loading.

#### `src/utils/`
-   **`config_loader.py`**: Utility to load configurations from YAML files (e.g., `configs/ppo_treasury_config.yaml`).
-   **`logger.py`**: Configures and provides a project-wide logger instance.
-   **`metrics.py`**: Functions to calculate performance metrics (Sharpe ratio, max drawdown, annualized return, etc.).
-   **`callbacks.py`**: Custom callbacks for Stable Baselines3 training (e.g., for specialized logging, or if `EvalCallback` is not sufficient).
-   **`plotting.py`**: (Optional) Utility functions for generating standardized plots (e.g., equity curves).

#### `src/evaluation/`
-   **`performance_analyzer.py`**: (Optional, can be part of `scripts/evaluate.py`) Contains logic for detailed analysis of agent performance, comparison against baselines, and potentially generating reports or visualizations.

### Scripts (`scripts/`)
-   **`preprocess_data.py`**: Orchestrates data loading, cleaning, feature engineering, and saving of processed data.
-   **`train.py`**: Main entry point for training the RL agent. Parses config, sets up environment and model, and runs the training loop.
-   **`evaluate.py`**: Main entry point for evaluating a trained agent and comparing it to baselines.
-   **`tune_hyperparameters.py`**: Script for hyperparameter optimization using Optuna.

### Configuration (`configs/`)
-   **`ppo_treasury_config.yaml`**: Central YAML file for all project configurations.

---

## Data and Feature Engineering
A critical aspect of this project is the **financial data** used to train the agent and the features extracted from it. We use historical daily data for 10-year Treasury futures (or an equivalent instrument, possibly 10-year Treasury note continuous futures prices). Each data point consists of the date, the price or return of the futures, and possibly volume or open interest. We augment this market data with additional features to provide context to the RL agent:

*   **Yield Curve Slope:** The slope of the yield curve at a given time, computed as the difference between long-term and short-term interest rates (for example, 10-year Treasury yield minus 2-year Treasury yield). A positive slope (steep curve) often indicates expectations of economic growth/inflation, whereas an inverted or flat curve can precede recessions. This feature gives the agent a sense of the macroeconomic regime, which can influence bond prices.
*   **Volatility Regime Indicator:** A measure of how volatile the market is. We compute, for instance, the 20-day rolling volatility of the 10Y futures’ daily returns. High values indicate turbulent market conditions. Alternatively or additionally, we could include the **MOVE index** (an analog of the VIX for bond markets) if available. This feature helps the agent adjust its risk-taking; for example, it might learn to take smaller or no positions during high volatility periods to avoid large losses.
*   **Trend and Momentum Features:** Indicators capturing price trends. We include moving averages such as a 50-day moving average and a 200-day moving average of the futures price, and perhaps their crossover. We can also include momentum (the percent price change over the last 1 month or 3 months) and oscillators like RSI. These features help the agent identify persistent trends or mean-reversion opportunities. For instance, a simple trend feature might be the difference (in percentage) between the current price and the 100-day moving average, or a binary feature indicating if short-term trend is up or down.
*   **Technical Indicators:** Other domain-specific signals like **carry** (for bonds, the yield difference could act as carry when holding futures vs cash bonds) or **seasonality flags** (though seasonality is weaker in rates, it could be month-of-year dummy if any pattern exists). We could also input **economic calendar events** or **Fed policy indicators** as dummy features if we wanted to capture those effects (e.g., a feature that signals days of FOMC meetings, since rate announcements can cause big moves). These are optional and can be added as extra columns in the feature set if data is available.

All features are engineered in `src/data_processing/feature_engineer.py` and combined into a feature matrix. We typically standardize these features (subtract mean, divide std) based on the training set statistics so that the neural network doesn’t have to deal with disparate scales. The **table below** summarizes some key features:

| **Feature**                | **Description**                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| *Yield Curve Slope*        | Difference between 10-year and 2-year Treasury yields (in %). Indicates the steepness of the yield curve (macro indicator).                                                    |
| *20-Day Volatility*        | Rolling 20-day standard deviation of daily returns of the futures. High values signify a volatile market regime.                                                               |
| *50d vs 200d MA Crossover* | Difference between 50-day and 200-day moving average of price. Positive if uptrend (golden cross), negative if downtrend (death cross).                                        |
| *Momentum (1M)*            | 21-day price momentum (% change over the last month). Captures short-term trend or mean reversion signals.                                                                     |
| *Position Indicator*       | (Derived internally by the environment) The current position of the agent (long/short/flat). This can be fed back into the state (e.g., one-hot encoded) so the network knows its current stance. |

*Table: Key input features for the trading agent and their financial interpretation.*

The training data spans from 2000–2015 (configurable in `configs/ppo_treasury_config.yaml`). We chose this period to include multiple interest rate cycles (e.g., early 2000s recession, 2008 crisis, etc.) so the agent can learn in varied conditions. The **validation** approach during training is to use a portion of the training data or a rolling validation: for instance, train on 2000–2010 and validate on 2011–2012, then extend, etc., or use cross-validation on time series (walk-forward validation). Primarily, we hold out 2016–2023 as an out-of-sample **test set** to evaluate final performance. The data module (`src.data_processing.data_loader`) ensures no future data leaks into training (we are careful with any feature that is forward-looking and only use lagged information).

By engineering these features, we inject domain knowledge into the agent’s observation space, which can make learning more sample-efficient. For example, by seeing the yield curve slope, the agent might implicitly learn different behavior in steep vs flat curve environments (which correspond to different market dynamics). The rich feature set also aids interpretability – we can later ask which features were most important for the agent’s decisions using techniques like SHAP.

## Custom Trading Environment Design
The **`TradingEnv`** (defined in `src/environments/trading_env.py`) is designed to realistically simulate trading in the futures market while being compatible with `gymnasium` (the successor to OpenAI Gym) and reinforcement learning frameworks like Stable Baselines3. Key aspects of this environment include the state representation, action effects, reward calculation, and episode management:

*   **State (Observation):** At each time step, the environment’s state is the feature vector for that day (as described under "Data and Feature Engineering"). For example, on day `t`, the observation might be: `[slope_t, volatility_t, trend_signal_t, momentum_t, position_{t-1}]`. The position at the previous step is included if `include_position_in_state` is true in the configuration, allowing the agent to be aware of its current holding. The observation is typically a numpy array of floats, and `observation_space` is defined as `gymnasium.spaces.Box` with appropriate dimensions and bounds (often normalized).

*   **Action Effects:** The action the agent takes is a target position. We have three discrete actions: **Short (0)**, **Flat (1)**, and **Long (2)**, defined in `action_space = gymnasium.spaces.Discrete(3)`.
    *   If the agent is currently flat and takes *Long*, it opens a long position (+1 unit). If it takes *Short*, it opens a short position (-1 unit).
    *   If the agent is already long and takes *Long* again, it maintains the long position (assuming a simple one-unit position size). Similar logic applies for short positions.
    *   If the agent switches (e.g., long to short), it first closes the existing position and then opens the opposite position. This incurs transaction costs for both closing and opening if simulated.
    *   Going to *Flat* from Long or Short closes the open position.
    *   **Transaction costs** are simulated each time a position is changed, based on `transaction_cost_bps` in the config. This discourages over-trading.

*   **Reward Function:** The reward is designed to reflect **risk-adjusted profit**. A basic reward would be the daily profit or loss (P&L). We improve this by adding penalties:
    *   **Volatility Penalty:** Subtracts a term proportional to the squared daily return or a similar volatility measure (controlled by `volatility_penalty_factor`). This encourages smoother returns.
    *   **Drawdown Penalty:** The environment tracks the running maximum of the portfolio value. If current equity falls below this peak, a penalty proportional to the drawdown is applied (controlled by `drawdown_penalty_factor`).
    *   The overall reward is `reward = P&L_step - volatility_penalty - drawdown_penalty`. Coefficients for each component are configurable.
    *   **Stop-Loss Penalty:** If an episode ends due to hitting the maximum drawdown (`max_drawdown_pct` from config), an additional large negative reward (`stop_loss_penalty`) can be applied.

*   **Episode Termination (`terminated` and `truncated` flags):**
    1.  **End of Data:** `terminated` is true when the agent reaches the end of the provided dataset.
    2.  **Stop-Loss:** `terminated` is true if the portfolio drawdown exceeds `max_drawdown_pct`.
    3.  **Max Episode Steps:** `truncated` can be true if an optional `episode_max_steps` is defined in config and reached (useful for training on smaller chunks of data).

*   **Integration with Stable Baselines3:** `TradingEnv` adheres to the `gymnasium.Env` interface. For training with SB3, it's typically wrapped in `DummyVecEnv` for vectorization (allowing multiple parallel environments) and potentially `VecNormalize` for observation and reward normalization, configured via `ppo_treasury_config.yaml`. Example:
    ```python
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    # env = DummyVecEnv([lambda: TradingEnv(train_data, config) for _ in range(config['agent']['n_envs'])])
    # if config['training']['normalize_env']:
    #   env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=config['agent']['gamma'])
    ```

The environment's design is documented with docstrings in `trading_env.py`, explaining the logic for reward calculation, state changes, and any assumptions made.

## Reinforcement Learning Setup (PPO Agent)
For training the agent, we use the **Proximal Policy Optimization (PPO)** algorithm implemented in **Stable Baselines3**. PPO is chosen for its reliability, sample efficiency, and stable learning dynamics. Stable Baselines3 provides a well-tested PyTorch-based implementation. The core RL setup is managed via `scripts/train.py` and configured in `configs/ppo_treasury_config.yaml`.

Key aspects of the RL training setup:

*   **Policy and Value Networks:** PPO is an actor-critic method.
    -   The **policy network (actor)** learns the trading strategy (mapping state to action probabilities).
    -   The **value network (critic)** estimates the expected return from a given state, helping to reduce variance in advantage estimation.
    -   By default, we use SB3’s `MlpPolicy`, which creates separate Multi-Layer Perceptrons (MLPs) for the actor and critic. The architecture (e.g., number of layers, neurons per layer) can be configured in `ppo_treasury_config.yaml` via `policy_kwargs` if customization is needed (e.g., `net_arch=[dict(pi=[64, 64], vf=[64, 64])]`).
    -   Custom policies can be defined in `src/agents/ppo_model.py` if more complex architectures (e.g., LSTMs for time-series memory via `RecurrentPPO`) are explored.

*   **Hyperparameters:** All PPO hyperparameters are centralized in `configs/ppo_treasury_config.yaml` under the `agent` section. Important parameters include:
    -   `learning_rate`: Typically 1e-4 to 5e-4; can be a constant or a schedule.
    -   `n_steps`: Number of steps to run for each environment per update.
    -   `batch_size`: Mini-batch size for policy updates.
    -   `n_epochs`: Number of optimization epochs per PPO update.
    -   `gamma`: Discount factor for future rewards (e.g., 0.99).
    -   `gae_lambda`: Lambda for Generalized Advantage Estimation (e.g., 0.95).
    -   `clip_range`: PPO clipping parameter to stabilize updates (e.g., 0.2).
    -   `ent_coef`: Entropy coefficient to encourage exploration.
    -   `vf_coef`: Value function coefficient in the loss calculation.
    -   `seed`: For reproducibility.

*   **Vectorized Environments:** To speed up training, PPO utilizes multiple parallel instances of `TradingEnv` (number specified by `n_envs` in config). This is handled by SB3's `DummyVecEnv` (for single-process) or `SubprocVecEnv` (for multi-process).
    -   `VecNormalize`: Optionally, the vectorized environment can be wrapped with `VecNormalize` to normalize observations and/or rewards on-the-fly, which can aid training stability. If used, the normalization statistics must be saved alongside the model and loaded during evaluation.

*   **Training Procedure (`scripts/train.py`):**
    1.  Loads configuration and preprocessed data.
    2.  Initializes the vectorized `TradingEnv`.
    3.  Instantiates the PPO agent (from `src.agents.ppo_model` which wraps SB3 PPO).
    4.  Sets up callbacks:
        -   `EvalCallback`: Periodically evaluates the agent on a separate validation environment and saves the best-performing model.
        -   `CheckpointCallback`: Saves model checkpoints at regular intervals.
        -   Custom callbacks (from `src.utils.callbacks`) for additional logging if needed.
    5.  Calls `model.learn()` to start training for `total_timesteps`.
    6.  Monitors training via console logs and TensorBoard (logs saved to `logs/tensorboard/`). Metrics like episode reward, policy loss, value loss, and entropy are typically logged.
    7.  Saves the final trained model and VecNormalize statistics (if used).

*   **Stable Baselines3 (SB3):** We leverage SB3's robust PPO implementation, allowing focus on environment design, feature engineering, and reward shaping rather than re-implementing the core RL algorithm.

*   **Saving and Versioning Models:**
    -   Models are saved to the `models/` directory (e.g., `best_model.zip`, `ppo_final.zip`).
    -   It's good practice to accompany saved models with their corresponding configuration file or metadata for traceability.

## Model Explainability and Interpretability
Explaining the behavior of a trained trading agent is crucial for building trust and for deriving insights. This project emphasizes interpretability through several methods, primarily documented and implemented within the `04_PerformanceEvaluation.ipynb` notebook:

*   **SHAP (SHapley Additive exPlanations) Analysis:**
    -   After training, the PPO agent's policy network (actor) can be treated as a function mapping state features to action logits or probabilities.
    -   We use the SHAP library to compute feature importances. For a set of test states, SHAP values explain how much each feature contributed to the agent's decision (e.g., push towards Long, Short, or Flat).
    -   Outputs include:
        -   **Summary Plots:** Bar charts showing mean absolute SHAP values, indicating overall feature importance.
        -   **Force Plots:** For individual predictions, showing how features push the output for specific instances.
    -   This helps identify which market signals (e.g., yield curve slope, volatility) most influence the agent's trading decisions, providing a sanity check against financial intuition.

*   **Action Breakdown and Scenario Analysis:**
    -   **Action Frequency:** Analyze the distribution of actions (Long, Short, Flat) taken by the agent over the test period. This reveals if the agent has a bias or how often it chooses to stay out of the market.
    -   **Trade Statistics:** Calculate metrics like average holding period, number of trades, win/loss ratio of trades.
    -   **Scenario Testing:** Feed handcrafted market scenarios (e.g., high volatility + flat yield curve) into the trained model to observe its decisions, helping to infer learned "rules-of-thumb."

*   **Trade Analysis and Visualization:**
    -   **Equity Curve Plotting:** The `04_PerformanceEvaluation.ipynb` notebook (and `scripts/evaluate.py`) should plot the agent's equity curve over time, often compared against baselines.
    -   **Position Visualization:** Plot the agent's position (Long/Short/Flat) overlaid on the price chart of the 10Y Treasury futures. This visually demonstrates how the agent reacts to market movements.
    -   **Case Studies:** Analyze agent behavior during specific, significant market events in the test period (e.g., periods of crisis, rallies, or sharp reversals) to understand its response under stress.

*   **Documentation and Code Clarity:**
    -   Well-commented code, clear docstrings, and descriptive variable names contribute to understanding the intended logic.
    -   The `TradingEnv` in particular documents the reward function and state/action mechanics.
    -   This README itself aims to explain design choices.

*   **Configuration Transparency:**
    -   Hyperparameters and environment settings are clearly defined in `configs/ppo_treasury_config.yaml`, with comments explaining key choices.

By combining quantitative methods like SHAP with qualitative analysis of trades and behavior, we aim to make the RL agent's strategy more transparent and understandable, moving beyond a "black-box" perception.

## Evaluation and Baseline Comparison
After training, the agent's performance is rigorously evaluated on the out-of-sample test dataset (e.g., 2016–2023, defined in `configs/ppo_treasury_config.yaml`). This evaluation is crucial for demonstrating the agent's effectiveness compared to simpler strategies. The evaluation process is primarily handled by `scripts/evaluate.py` and can be interactively explored in `notebooks/04_PerformanceEvaluation.ipynb`.

*   **Evaluation Procedure:**
    1.  Load the trained PPO model (e.g., `models/best_model.zip`).
    2.  Set up the `TradingEnv` with the test dataset.
    3.  Run the agent in deterministic mode (no exploration) through the test period, recording actions, rewards, and portfolio values.
    4.  Simulate baseline strategies on the same test data.

*   **Baseline Strategies:**
    -   **Buy-and-Hold:** A passive strategy that buys the asset at the beginning of the test period and holds it until the end.
    -   **Moving Average (MA) Crossover:** A simple technical strategy (e.g., go long when 50-day MA > 200-day MA, go short or flat otherwise). Parameters for MAs are configurable.
    -   **Always Flat:** A zero-risk, zero-return baseline representing no trading activity.

*   **Performance Metrics:** Calculated using `src/utils/metrics.py` for the agent and each baseline:
    -   **Total Return:** Percentage change in portfolio value over the test period.
    -   **Annualized Return:** Average yearly return.
    -   **Annualized Volatility:** Standard deviation of daily returns, annualized.
    -   **Sharpe Ratio:** Risk-adjusted return (Annualized Return / Annualized Volatility, considering a risk-free rate if specified).
    -   **Maximum Drawdown (Max DD):** Largest peak-to-trough decline in portfolio value.
    -   **Sortino Ratio:** Similar to Sharpe, but only penalizes downside volatility.
    -   **Calmar Ratio:** Annualized Return / Max Drawdown.
    -   **Trade Statistics:** Win rate, average profit/loss per trade, number of trades (for agent and active baselines).

*   **Results Presentation:**
    -   Performance metrics are typically presented in a table for easy comparison:
      | Strategy           | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown |
      | ------------------ | ------------- | ----------------- | ------------ | ------------ |
      | **RL PPO Agent**   | (e.g., 12.5%) | (e.g., 8.0%)      | (e.g., 1.56) | (e.g., -10.2%)|
      | Buy & Hold         | (e.g., 8.0%)  | (e.g., 6.5%)      | (e.g., 1.23) | (e.g., -25.0%)|
      | MA Crossover       | (e.g., 5.0%)  | (e.g., 5.5%)      | (e.g., 0.90) | (e.g., -15.0%)|
      *(Note: Numbers are illustrative.)*
    -   Equity curves are plotted to visualize portfolio growth over time for the agent and baselines.

*   **Discussion of Results:** The evaluation aims to determine if the RL agent can achieve superior risk-adjusted returns compared to baselines, evidenced by higher Sharpe/Sortino ratios and lower drawdowns for a given level of return. The analysis also considers transaction costs, which are factored into the agent's P&L within the environment.

*   **Robustness Checks (Future Work/Considerations):**
    -   Testing on different time periods or related instruments.
    -   Sensitivity analysis of hyperparameters.
    -   Statistical significance of performance differences (e.g., using t-tests or bootstrap methods on returns).

## Testing and Code Quality
Ensuring code reliability and maintainability is a key focus.

*   **Unit Tests (`tests/` directory):**
    -   We use `pytest` for writing and running unit tests.
    -   **`test_trading_env.py`**: Tests for the `TradingEnv`, covering core logic like action execution, P&L calculation, reward shaping, transaction costs, and episode termination (stop-loss, end-of-data).
    -   **`test_data_loader.py`**: Tests for data loading functions in `src.data_processing.data_loader`, including handling different file formats (if applicable), date parsing, and data splitting.
    -   **`test_feature_engineer.py`**: Tests for feature calculation functions in `src.data_processing.feature_engineer`, ensuring features like volatility, MAs, RSI, etc., are computed correctly.
    -   **`test_baselines.py` / `test_performance_analyzer.py`**: Tests for baseline strategy logic and metric calculations.
    -   Tests are run via the command line: `pytest tests/`

*   **Code Style and Linting:**
    -   The codebase aims to adhere to PEP 8 style guidelines.
    -   Linters like Flake8 and formatters like Black may be used to ensure consistency and readability. (Configuration for these tools, e.g., `pyproject.toml`, would be added if formally integrated).

*   **Modularity and Reusability:**
    -   The project is structured into distinct modules (data processing, environment, agent, utilities) to promote separation of concerns and allow components to be updated or reused independently.

*   **Documentation:**
    -   **Docstrings:** All major classes and functions include docstrings explaining their purpose, arguments, and return values.
    -   **README.md:** This file serves as comprehensive project documentation.
    -   **Jupyter Notebooks:** Notebooks in `notebooks/` are written with explanatory Markdown cells to document analyses and experiments.

*   **Dependency Management:**
    -   Project dependencies are listed in `requirements.txt`.

*   **Configuration Management:**
    -   Experiment parameters are managed externally via YAML files in the `configs/` directory, promoting reproducibility and easy modification of settings.

## Key Design Choices & Discussion Points (Q&A Style)

*   **Q: Why use PPO for this trading problem?**
    *   A: PPO offers a good balance of performance, stability, and sample efficiency. Its clipped objective function prevents overly large policy updates, which is beneficial in noisy financial market environments. Stable Baselines3 provides a reliable and well-tested implementation.
*   **Q: How is the reward function designed for risk-adjusted returns?**
    *   A: The reward is not just raw P&L. It's shaped to include penalties for volatility (e.g., squared daily returns) and drawdowns from the peak portfolio value. This encourages the agent to find strategies that yield smoother equity curves and optimize metrics like the Sharpe ratio, rather than just maximizing profit. Configurable coefficients (`volatility_penalty_factor`, `drawdown_penalty_factor`) allow tuning this balance.
*   **Q: Why a custom Gym environment (`TradingEnv`)?**
    *   A: A custom environment built with `gymnasium` allows precise modeling of the 10Y Treasury futures market. This includes accurate simulation of discrete actions (Long/Short/Flat), position sizing (even if simplified to 1 unit), transaction costs, and the implementation of our specific risk-adjusted reward function. It gives full control over the market dynamics presented to the agent.
*   **Q: What financial features are used and why?**
    *   A: The agent uses features like yield curve slope (a macroeconomic indicator), market volatility (e.g., rolling standard deviation of returns as a risk measure), and trend/momentum indicators (e.g., MA crossovers, RSI, price momentum). These are chosen because they are commonly used financial indicators that provide context about market conditions and potential future price movements. Details are in the "Data and Feature Engineering" section.
*   **Q: How do you handle data splitting and avoid lookahead bias?**
    *   A: Data is strictly split into training, validation (optional, can be a segment of training or a separate holdout), and testing periods using date ranges defined in the configuration. All feature calculations (e.g., rolling means, volatility) are performed using only past data (lagged information) available up to that point in time to prevent lookahead bias. Scalers for normalization are fit *only* on the training set.
*   **Q: How is model interpretability addressed?**
    *   A: We use SHAP (SHapley Additive exPlanations) to analyze feature importance for the trained policy, helping to understand which factors most influence its decisions. Additionally, we analyze action distributions, trade statistics, and agent behavior in specific market scenarios. Visualizations like position overlays on price charts also aid interpretation.
*   **Q: Why YAML for configuration over `config.py`?**
    *   A: YAML files (in `configs/`) provide a more flexible and human-readable way to manage experiment parameters compared to a Python file. It's easier to define multiple configurations, track changes with version control, and load them without altering Python code.
*   **Q: How could the agent's memory be improved?**
    *   A: Currently, the agent relies on a Markovian state (current observation). To incorporate longer-term memory, one could use features that summarize longer history (e.g., longer MA windows) or implement a recurrent policy network (e.g., using LSTMs with SB3's `RecurrentPPO`).
*   **Q: How are transaction costs handled?**
    *   A: Transaction costs are modeled as a percentage of the trade value (configurable `transaction_cost_bps`) and are deducted from the portfolio value each time the agent changes its position. This realistically penalizes frequent trading.

## Next Steps / Future Work

*   **Advanced Feature Engineering:** Explore more sophisticated features (e.g., GARCH for volatility, features from order book data if available, alternative representations of yield curve).
*   **Recurrent Policies:** Implement and test recurrent neural networks (LSTMs, GRUs) within the PPO framework (e.g., using `RecurrentPPO` from `sb3-contrib`) to provide the agent with longer-term memory.
*   **Different RL Algorithms:** Experiment with other RL algorithms suitable for financial markets (e.g., A2C, DDPG if action space were continuous, or more advanced policy gradient methods).
*   **Hyperparameter Optimization:** Conduct more extensive hyperparameter tuning using Optuna (`scripts/tune_hyperparameters.py`) across a wider range of parameters and for more trials.
*   **Walk-Forward Validation:** Implement a more robust walk-forward validation scheme for backtesting, which better simulates how models would be retrained and deployed over time.
*   **Risk Management Enhancements:** Explore dynamic position sizing based on volatility or conviction, more sophisticated stop-loss mechanisms (e.g., trailing stops).
*   **Live Paper Trading:** Integrate with a brokerage API or a simulated exchange to paper trade the trained agent in (near) real-time.
*   **Ensemble Methods:** Experiment with ensembling multiple trained agents or models.
*   **Enhanced Interpretability:** Dive deeper with SHAP analysis (e.g., interaction plots) or explore other XAI techniques.
*   **Portfolio Context:** Extend to manage a portfolio of assets rather than a single instrument.

---

*This README will be further updated as the project progresses and specific implementation details are finalized.*
