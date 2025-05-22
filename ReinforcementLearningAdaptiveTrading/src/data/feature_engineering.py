import pandas as pd
import numpy as np

# Helper function for rolling calculations, handling potential edge cases
def rolling_apply_safe(series, window, func, min_periods=None):
    if series.empty or len(series) < window:
        return pd.Series(index=series.index, dtype=float) # Return empty or NaN series matching index
    if min_periods is None:
        min_periods = window
    return series.rolling(window=window, min_periods=min_periods).apply(func, raw=True)


def calculate_yield_curve_slope(df, long_term_yield_col, short_term_yield_col):
    """
    Calculates the yield curve slope.

    Args:
        df (pd.DataFrame): DataFrame containing yield data.
        long_term_yield_col (str): Column name for the long-term yield (e.g., '10Y_Yield').
        short_term_yield_col (str): Column name for the short-term yield (e.g., '2Y_Yield').

    Returns:
        pd.Series: Series containing the yield curve slope.
                   Returns an empty Series if input columns are missing.
    """
    if long_term_yield_col not in df.columns or short_term_yield_col not in df.columns:
        print("Warning: Yield columns for slope calculation not found in DataFrame.")
        return pd.Series(dtype=float) # Or pd.Series(index=df.index, dtype=float) if index is known
    return df[long_term_yield_col] - df[short_term_yield_col]


def calculate_historical_volatility(price_series, window=20):
    """
    Calculates historical rolling volatility of price returns.

    Args:
        price_series (pd.Series): Series of prices.
        window (int): Rolling window period (e.g., 20 days).

    Returns:
        pd.Series: Series containing the historical volatility.
    """
    if price_series.empty or len(price_series) < 2: # Need at least 2 prices for one return
        return pd.Series(index=price_series.index, dtype=float)
        
    daily_returns = price_series.pct_change()
    # Standard deviation of daily returns, annualized by sqrt(252) if desired,
    # but for features often raw rolling std is used. Here, just rolling std.
    volatility = daily_returns.rolling(window=window, min_periods=window).std()
    return volatility


def calculate_moving_average(price_series, window=50):
    """
    Calculates the simple moving average (SMA) of a price series.

    Args:
        price_series (pd.Series): Series of prices.
        window (int): Rolling window period for the MA.

    Returns:
        pd.Series: Series containing the moving average.
    """
    if price_series.empty or len(price_series) < window:
        return pd.Series(index=price_series.index, dtype=float)
    return price_series.rolling(window=window, min_periods=window).mean()


def calculate_ma_crossover_signal(price_series, short_window=50, long_window=200):
    """
    Calculates the difference between a short-term and long-term moving average.
    A positive value indicates short MA is above long MA (bullish), negative is bearish.

    Args:
        price_series (pd.Series): Series of prices.
        short_window (int): Window for the short-term MA.
        long_window (int): Window for the long-term MA.

    Returns:
        pd.Series: Series containing the MA crossover signal (difference).
    """
    sma_short = calculate_moving_average(price_series, short_window)
    sma_long = calculate_moving_average(price_series, long_window)
    if sma_short.empty or sma_long.empty:
        return pd.Series(index=price_series.index, dtype=float)
    return sma_short - sma_long


def calculate_rsi(price_series, window=14):
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        price_series (pd.Series): Series of prices.
        window (int): Window period for RSI calculation.

    Returns:
        pd.Series: Series containing the RSI values (0-100).
    """
    if price_series.empty or len(price_series) < window +1: # Need enough data for initial calculations
        return pd.Series(index=price_series.index, dtype=float)

    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean() # Use min_periods=1 for EWMA-like start
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean() # Use min_periods=1 for EWMA-like start
    
    # Using simple moving average for initial RS, then can use EWMA for smoother RSI
    # For simplicity here, using rolling mean for gain/loss.
    # More standard RSI uses Wilder's smoothing (an EWMA).
    # gain = delta.clip(lower=0).rolling(window=window).mean()
    # loss = -delta.clip(upper=0).rolling(window=window).mean()


    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle cases where loss is zero (rs is inf, rsi is 100) or gain is zero (rs is 0, rsi is 0)
    rsi[loss == 0] = 100
    rsi[gain == 0] = 0 # If gain is 0 and loss is also 0, then rs is NaN, rsi is NaN. This is fine.
    
    return rsi


def calculate_momentum(price_series, window=21):
    """
    Calculates N-day price momentum (percentage change).

    Args:
        price_series (pd.Series): Series of prices.
        window (int): Window period for momentum (e.g., 21 days for 1-month momentum).

    Returns:
        pd.Series: Series containing the momentum values.
    """
    if price_series.empty or len(price_series) < window:
         return pd.Series(index=price_series.index, dtype=float)
    # M_t = (P_t / P_{t-N}) - 1 or P_t - P_{t-N}
    # Using (P_t - P_{t-N}) / P_{t-N} which is pct_change over N periods
    momentum = price_series.pct_change(periods=window) 
    return momentum


def engineer_features(df, config):
    """
    Main function to orchestrate feature engineering.
    It takes a DataFrame (typically raw or processed data from data_loader)
    and adds new feature columns based on the provided configuration.

    Args:
        df (pd.DataFrame): Input DataFrame, must contain a 'price' column,
                           and other columns if needed for specific features (e.g., yields).
                           Assumes df index is DatetimeIndex.
        config (dict): Configuration dictionary, specifically config['data'] for feature params.

    Returns:
        pd.DataFrame: DataFrame with added feature columns.
                      All engineered features will be prefixed with 'feat_'.
    """
    if df.empty:
        print("Warning: Input DataFrame for feature engineering is empty.")
        return df

    price_col = config['data'].get('price_column', 'price')
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found for feature engineering.")

    features_df = pd.DataFrame(index=df.index)

    # 1. Yield Curve Slope (Example - requires specific yield columns in input df)
    # This feature's actual calculation depends on availability of columns like '10Y_Yield', '2Y_Yield'
    # These would typically be loaded by data_loader and merged if they are in separate files.
    # For this example, let's assume they might be present or skip if not.
    if 'yield_curve_slope' in config['data'].get('feature_columns', []):
        # These column names would come from config or be hardcoded if standard
        long_yield_col = config['data'].get('long_term_yield_column_name', '10Y_Yield') 
        short_yield_col = config['data'].get('short_term_yield_column_name', '2Y_Yield')
        if long_yield_col in df.columns and short_yield_col in df.columns:
            features_df['feat_yield_curve_slope'] = calculate_yield_curve_slope(df, long_yield_col, short_yield_col)
        else:
            print(f"Warning: Columns for yield curve slope ({long_yield_col}, {short_yield_col}) not found. Skipping feature.")
            features_df['feat_yield_curve_slope'] = np.nan # Fill with NaNs if cannot compute


    # 2. Volatility
    if 'volatility_20d' in config['data'].get('feature_columns', []): # Example name
        vol_window = config['data'].get('feature_params', {}).get('volatility_window', 20)
        features_df['feat_volatility_20d'] = calculate_historical_volatility(df[price_col], window=vol_window)

    # 3. Trend/Momentum Indicators - Moving Average Crossover
    if 'ma_crossover_signal' in config['data'].get('feature_columns', []): # Example name
        short_ma_win = config['data'].get('feature_params', {}).get('ma_short_window', 50)
        long_ma_win = config['data'].get('feature_params', {}).get('ma_long_window', 200)
        features_df['feat_ma_crossover_signal'] = calculate_ma_crossover_signal(df[price_col], short_window=short_ma_win, long_window=long_ma_win)
        # Also add individual MAs if they are listed in feature_columns
        if 'sma_short' in config['data'].get('feature_columns', []): # Assuming generic name
             features_df['feat_sma_short'] = calculate_moving_average(df[price_col], window=short_ma_win)
        if 'sma_long' in config['data'].get('feature_columns', []): # Assuming generic name
             features_df['feat_sma_long'] = calculate_moving_average(df[price_col], window=long_ma_win)


    # 4. Momentum
    if 'momentum_1m' in config['data'].get('feature_columns', []): # Example name for 1-month momentum
        momentum_window = config['data'].get('feature_params', {}).get('momentum_window_1m', 21) # Approx 21 trading days in a month
        features_df['feat_momentum_1m'] = calculate_momentum(df[price_col], window=momentum_window)

    # 5. RSI (Relative Strength Index)
    if 'rsi_14d' in config['data'].get('feature_columns', []): # Example name
        rsi_window = config['data'].get('feature_params', {}).get('rsi_window', 14)
        features_df['feat_rsi_14d'] = calculate_rsi(df[price_col], window=rsi_window)
        
    # Add other features as needed based on config['data']['feature_columns']
    # E.g., Carry, Seasonality flags, Volume-based indicators etc.

    # Normalization (Example: z-score normalization)
    # This should ideally be done carefully: fit scaler on training data ONLY, then transform train/val/test.
    # For simplicity here, it's commented out. VecNormalize in SB3 can also handle this for env observations.
    # if config['data'].get('normalize_features', False):
    #     for col in features_df.columns:
    #         if features_df[col].dtype in [np.float64, np.float32]:
    #             mean = features_df[col].mean() # Should be mean from training set
    #             std = features_df[col].std()   # Should be std from training set
    #             if std > 1e-6: # Avoid division by zero for constant columns
    #                 features_df[col] = (features_df[col] - mean) / std
    #             else:
    #                 features_df[col] = 0 # Or handle as per strategy for constant features

    # Combine with original df or return only features?
    # Typically, you'd merge these features back into the main df that TradingEnv will use.
    # The TradingEnv expects a single df with price and all specified feature columns.
    
    # Ensure all requested features are present, fill with NaN if not computed
    for feat_name in config['data'].get('feature_columns', []):
        internal_feat_name = 'feat_' + feat_name # Assuming we prefix all engineered feats
        if internal_feat_name not in features_df.columns:
            # This happens if a feature in config isn't implemented or skipped above
            print(f"Warning: Feature '{feat_name}' (expected as '{internal_feat_name}') was requested but not generated. Filling with NaN.")
            features_df[internal_feat_name] = np.nan


    # Rename features to match exactly what's in config['data']['feature_columns']
    # if they were prefixed with 'feat_'
    rename_map = {col: col.replace('feat_', '') for col in features_df.columns if col.startswith('feat_')}
    features_df = features_df.rename(columns=rename_map)
    
    # Select only the features listed in config, plus the price column from original df
    final_df_cols = [price_col] + config['data'].get('feature_columns', [])
    # Merge features with original data that contains the price
    # Keep original df's columns that are not features (e.g. if it has 'volume' not used as feature)
    output_df = df.copy()
    for col in features_df.columns:
        if col in config['data'].get('feature_columns', []): # Only add if it's a requested feature
            output_df[col] = features_df[col]
    
    # Fill NaNs resulting from rolling calculations (e.g., at the beginning of the series)
    # Common strategies: forward fill, or drop rows with NaNs.
    # Forward fill is often preferred in time series to avoid lookahead bias from backfill.
    # However, for RL, agent needs valid features. Dropping NaNs might be necessary if ffill isn't appropriate.
    # The amount of data lost depends on the longest rolling window.
    if config['data'].get('feature_fillna_method', 'ffill') == 'ffill':
        output_df = output_df.fillna(method='ffill')
    
    # Crucially, after ffill, there might still be NaNs at the very beginning if windows are large.
    # These rows are unusable by the agent and should be dropped.
    output_df = output_df.dropna()

    print(f"Feature engineering complete. Output data shape: {output_df.shape}")
    if output_df.empty and not df.empty:
        print("Warning: Output DataFrame is empty after feature engineering and dropna. Check window sizes and data length.")
        
    return output_df


if __name__ == '__main__':
    print("\nExample Usage of feature_engineering.py:")

    # Create dummy data
    num_days = 300
    price = np.random.randn(num_days).cumsum() + 100
    idx = pd.to_datetime([pd.Timestamp('2020-01-01') + pd.Timedelta(days=i) for i in range(num_days)])
    dummy_raw_df = pd.DataFrame({'price': price, 
                                 '10Y_Yield': np.random.rand(num_days) * 2 + 1, # 1% to 3%
                                 '2Y_Yield': np.random.rand(num_days) * 1 + 0.5   # 0.5% to 1.5%
                                 }, index=idx)
    print(f"Initial dummy data shape: {dummy_raw_df.shape}")

    # Dummy config (subset of what's in config.py)
    dummy_fe_config = {
        "data": {
            "price_column": "price",
            "feature_columns": [ # These are the *target* feature names in the final DataFrame for the env
                "yield_curve_slope", 
                "volatility_20d", 
                "ma_crossover_signal",
                "momentum_1m",
                "rsi_14d",
                "sma_short", # Also request individual MAs
                "sma_long"
            ],
            # Feature specific parameters (optional, provides defaults in functions)
            "feature_params": { 
                "volatility_window": 20,
                "ma_short_window": 10, # Shorter for more signals in small dummy data
                "ma_long_window": 30,  # Shorter for more signals
                "momentum_window_1m": 21,
                "rsi_window": 14
            },
            "long_term_yield_column_name": "10Y_Yield", # For yield_curve_slope
            "short_term_yield_column_name": "2Y_Yield", # For yield_curve_slope
            "feature_fillna_method": "ffill", # or None or 'dropna' (handled by final dropna)
            # "normalize_features": False # Normalization example
        }
    }

    # Engineer features
    features_output_df = engineer_features(dummy_raw_df.copy(), dummy_fe_config)
    
    print("\nEngineered Features DataFrame head:")
    print(features_output_df.head())
    print("\nEngineered Features DataFrame tail:")
    print(features_output_df.tail())
    print(f"Shape after feature engineering: {features_output_df.shape}")
    print("\nNaN counts per column in engineered data:")
    print(features_output_df.isnull().sum())

    # Test with a feature name in config that isn't implemented/calculated
    dummy_fe_config_extended = dummy_fe_config.copy()
    dummy_fe_config_extended['data']['feature_columns'].append("non_existent_feature")
    features_output_df_extended = engineer_features(dummy_raw_df.copy(), dummy_fe_config_extended)
    print(f"\nShape with non_existent_feature: {features_output_df_extended.shape}")
    if 'non_existent_feature' in features_output_df_extended.columns:
        print("NaNs in 'non_existent_feature':", features_output_df_extended['non_existent_feature'].isnull().sum())
