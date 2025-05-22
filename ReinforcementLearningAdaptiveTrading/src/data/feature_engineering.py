import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib # For saving/loading scaler
import os

# Dummy logger for standalone execution
class SimpleLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")

# Helper function for rolling calculations
def rolling_apply_safe(series, window, func, min_periods=None, logger=None):
    if logger is None: logger = SimpleLogger()
    if series.empty or len(series) < window:
        logger.debug(f"Series too short for rolling window {window}, returning NaN series.")
        return pd.Series(index=series.index, dtype=float)
    if min_periods is None: min_periods = window
    return series.rolling(window=window, min_periods=min_periods).apply(func, raw=True)

def calculate_yield_curve_slope(df, long_term_yield_col, short_term_yield_col, logger=None):
    if logger is None: logger = SimpleLogger()
    if long_term_yield_col not in df.columns or short_term_yield_col not in df.columns:
        logger.warning(f"Yield columns for slope calc not found: {long_term_yield_col} or {short_term_yield_col}. Skipping.")
        return pd.Series(index=df.index, dtype=float)
    return df[long_term_yield_col] - df[short_term_yield_col]

def calculate_historical_volatility(price_series, window=20, logger=None):
    if logger is None: logger = SimpleLogger()
    if price_series.empty or len(price_series) < 2:
        logger.debug(f"Price series too short for volatility (window {window}), returning NaN series.")
        return pd.Series(index=price_series.index, dtype=float)
    daily_returns = price_series.pct_change()
    return daily_returns.rolling(window=window, min_periods=window).std()

def calculate_moving_average(price_series, window=50, logger=None):
    if logger is None: logger = SimpleLogger()
    if price_series.empty or len(price_series) < window:
        logger.debug(f"Price series too short for MA (window {window}), returning NaN series.")
        return pd.Series(index=price_series.index, dtype=float)
    return price_series.rolling(window=window, min_periods=window).mean()

def calculate_ma_crossover_signal(price_series, short_window=50, long_window=200, logger=None):
    if logger is None: logger = SimpleLogger()
    sma_short = calculate_moving_average(price_series, short_window, logger=logger)
    sma_long = calculate_moving_average(price_series, long_window, logger=logger)
    if sma_short.empty or sma_long.empty: # Check if MAs could be computed
        return pd.Series(index=price_series.index, dtype=float)
    return sma_short - sma_long

def calculate_rsi(price_series, window=14, logger=None):
    if logger is None: logger = SimpleLogger()
    if price_series.empty or len(price_series) < window + 1 :
        logger.debug(f"Price series too short for RSI (window {window}), returning NaN series.")
        return pd.Series(index=price_series.index, dtype=float)
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi[loss == 0] = 100
    rsi[gain == 0] = 0 
    return rsi

def calculate_momentum(price_series, window=21, logger=None):
    if logger is None: logger = SimpleLogger()
    if price_series.empty or len(price_series) < window : # pct_change needs at least `periods` items
         logger.debug(f"Price series too short for momentum (window {window}), returning NaN series.")
         return pd.Series(index=price_series.index, dtype=float)
    return price_series.pct_change(periods=window)

def engineer_features(df, config, data_type='train', logger=None): # Removed scaler_path from args, get from config
    """
    Main function to orchestrate feature engineering and scaling.
    Args:
        df (pd.DataFrame): Input DataFrame.
        config (dict): Configuration dictionary.
        data_type (str): Type of data ('train', 'validation', 'test'), for scaler handling.
        logger (logging.Logger, optional): Logger instance.
    """
    if logger is None: logger = SimpleLogger()

    if df.empty:
        logger.warning("Input DataFrame for feature engineering is empty.")
        return df

    cfg_data = config['data']
    price_col = cfg_data['price_column']
    scaler_path = cfg_data.get('scaler_path') # Get scaler_path from config

    if price_col not in df.columns:
        logger.error(f"Price column '{price_col}' not found for feature engineering.")
        raise ValueError(f"Price column '{price_col}' not found for feature engineering.")

    features_df = pd.DataFrame(index=df.index)
    cfg_feature_params = cfg_data.get('feature_params', {})

    for feature_name in cfg_data.get('feature_columns', []):
        col_name_in_features_df = feature_name
        
        if feature_name == 'yield_curve_slope':
            long_yield_col = cfg_data.get('long_term_yield_column_name', '10Y_Yield')
            short_yield_col = cfg_data.get('short_term_yield_column_name', '2Y_Yield')
            if long_yield_col in df.columns and short_yield_col in df.columns:
                features_df[col_name_in_features_df] = calculate_yield_curve_slope(df, long_yield_col, short_yield_col, logger=logger)
            else:
                logger.warning(f"Yield columns for slope not found. Skipping '{feature_name}'.")
                features_df[col_name_in_features_df] = np.nan
        
        elif feature_name == 'volatility_20d':
            vol_window = cfg_feature_params.get('volatility_window', 20)
            features_df[col_name_in_features_df] = calculate_historical_volatility(df[price_col], vol_window, logger=logger)
        
        elif feature_name == 'ma_crossover_signal':
            short_win = cfg_feature_params.get('ma_short_window', 50)
            long_win = cfg_feature_params.get('ma_long_window', 200)
            features_df[col_name_in_features_df] = calculate_ma_crossover_signal(df[price_col], short_win, long_win, logger=logger)
        
        elif feature_name == 'momentum_1m':
            mom_win = cfg_feature_params.get('momentum_window_1m', 21)
            features_df[col_name_in_features_df] = calculate_momentum(df[price_col], mom_win, logger=logger)

        elif feature_name == 'rsi_14d':
            rsi_win = cfg_feature_params.get('rsi_window', 14)
            features_df[col_name_in_features_df] = calculate_rsi(df[price_col], rsi_win, logger=logger)
            
        elif feature_name.startswith('sma_'):
            try: # Example: sma_50
                window_str = feature_name.split('_')[-1] # "50"
                # Try to find a specific config like "ma_50_window" or use the number directly
                window_key = cfg_feature_params.get(f'ma_{window_str}_window', int(window_str))
                sma_win = int(window_key) # Ensure it's an int
                features_df[col_name_in_features_df] = calculate_moving_average(df[price_col], sma_win, logger=logger)
            except Exception as e:
                logger.warning(f"Could not parse/calculate SMA for '{feature_name}': {e}. Skipping.")
                features_df[col_name_in_features_df] = np.nan
        else:
            if col_name_in_features_df not in features_df.columns:
                 logger.warning(f"Feature '{feature_name}' is listed in config but not implemented in engineer_features. Filling with NaN.")
                 features_df[col_name_in_features_df] = np.nan

    output_df = df[[price_col]].copy()
    for col in features_df.columns:
        if col in cfg_data.get('feature_columns', []): # Only add if it's a requested feature by final name
             output_df[col] = features_df[col]
    
    feature_cols_to_scale = [f_col for f_col in cfg_data.get('feature_columns', []) if f_col in output_df.columns]

    if not feature_cols_to_scale:
        logger.info("No features specified or available for scaling.")
    elif scaler_path:
        scaler_dir = os.path.dirname(scaler_path)
        if scaler_dir and not os.path.exists(scaler_dir): # Ensure directory for scaler exists
            os.makedirs(scaler_dir, exist_ok=True)
            logger.info(f"Created directory for scaler: {scaler_dir}")

        if data_type == 'train':
            logger.info(f"Fitting scaler on training data features: {feature_cols_to_scale} and saving to {scaler_path}")
            scaler = StandardScaler()
            # Ensure only valid data (no NaNs from feature calculation before dropna) is used to fit scaler
            # However, typical workflow is: calc features -> handle NaNs from rolling windows -> then scale.
            # If NaNs are still present in feature_cols_to_scale here, fit_transform will fail or give warning.
            # Assuming feature_cols_to_scale are ready for scaling.
            temp_scaled_data = output_df[feature_cols_to_scale].dropna() # Drop rows with NaN in features before fitting
            if not temp_scaled_data.empty:
                scaler.fit(temp_scaled_data)
                output_df[feature_cols_to_scale] = scaler.transform(output_df[feature_cols_to_scale])
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}.")
            else:
                logger.warning("No valid data to fit the scaler after dropping NaNs from features. Scaling skipped.")

        else: 
            if os.path.exists(scaler_path):
                logger.info(f"Loading scaler from {scaler_path} and transforming {data_type} data features.")
                scaler = joblib.load(scaler_path)
                output_df[feature_cols_to_scale] = scaler.transform(output_df[feature_cols_to_scale])
            else:
                logger.error(f"Scaler file not found at {scaler_path} for {data_type} data. Features will not be scaled.")
    else:
        logger.warning("Config 'data.scaler_path' not provided. Features will not be scaled by engineer_features.")

    initial_rows = len(output_df)
    if cfg_data.get('feature_fillna_method', 'ffill') == 'ffill':
        output_df = output_df.fillna(method='ffill')
    
    final_dropna_rows_before = len(output_df)
    output_df = output_df.dropna()
    rows_dropped_final = final_dropna_rows_before - len(output_df)
    if rows_dropped_final > 0:
        logger.info(f"Dropped {rows_dropped_final} rows due to remaining NaNs after ffill (or if ffill was not used). Initial rows: {initial_rows}, final rows: {len(output_df)}")
    
    if output_df.empty and not df.empty:
        logger.warning("Output DataFrame is empty after feature engineering and dropna. Check window sizes, data length, and NaN handling.")
    logger.info(f"Feature engineering for {data_type} complete. Output data shape: {output_df.shape}")
    return output_df


if __name__ == '__main__':
    test_logger = SimpleLogger()
    test_logger.info("Example Usage of feature_engineering.py:")
    num_days = 300
    price_data = np.arange(num_days, dtype=float) * 0.5 + 100 + np.random.randn(num_days) * 2 # More realistic price
    idx = pd.to_datetime([pd.Timestamp('2020-01-01') + pd.Timedelta(days=i) for i in range(num_days)])
    dummy_raw_df = pd.DataFrame({
        'price': price_data, 
        '10Y_Yield': np.random.rand(num_days) * 0.02 + 0.01, # Yields as decimals
        '2Y_Yield': np.random.rand(num_days) * 0.01 + 0.005
    }, index=idx)

    dummy_fe_config = {
        "data": {
            "price_column": "price",
            "feature_columns": ["yield_curve_slope", "volatility_20d", "ma_crossover_signal", "momentum_1m", "rsi_14d", "sma_50"],
            "feature_params": { 
                "volatility_window": 20, "ma_short_window": 10, "ma_long_window": 30,
                "momentum_window_1m": 21, "rsi_window": 14, "ma_50_window": 50
            },
            "long_term_yield_column_name": "10Y_Yield", 
            "short_term_yield_column_name": "2Y_Yield",
            "feature_fillna_method": "ffill",
            "scaler_path": "temp_test_scaler.joblib" # Temporary path for testing
        }
    }
    
    # Ensure scaler directory exists if scaler_path includes a directory
    scaler_output_path = dummy_fe_config['data']['scaler_path']
    scaler_output_dir = os.path.dirname(scaler_output_path)
    if scaler_output_dir and not os.path.exists(scaler_output_dir):
        os.makedirs(scaler_output_dir)

    features_train_df = engineer_features(dummy_raw_df.copy(), dummy_fe_config, data_type='train', logger=test_logger)
    test_logger.info("Engineered Train Features DataFrame head:\n" + features_train_df.head().to_string())
    
    if os.path.exists(scaler_output_path):
        features_test_df = engineer_features(dummy_raw_df.copy().iloc[100:], dummy_fe_config, data_type='test', logger=test_logger)
        test_logger.info("Engineered Test Features DataFrame head (scaled with loaded scaler):\n" + features_test_df.head().to_string())
        
        # Check if scaling was applied (means should be close to 0, std close to 1 for scaled columns in train)
        # For test, they won't be exactly 0 and 1, but transformed by train's scaler.
        test_logger.info("Scaled train features description:\n" + features_train_df[dummy_fe_config['data']['feature_columns']].describe().to_string())
        test_logger.info("Scaled test features description:\n" + features_test_df[dummy_fe_config['data']['feature_columns']].describe().to_string())

        os.remove(scaler_output_path)
        test_logger.info(f"Cleaned up dummy scaler: {scaler_output_path}")
        if scaler_output_dir and not os.listdir(scaler_output_dir): # Remove dir if empty
             os.rmdir(scaler_output_dir)
             test_logger.info(f"Cleaned up dummy scaler directory: {scaler_output_dir}")

    else:
        test_logger.error("Dummy scaler was not saved, cannot test loading path.")
