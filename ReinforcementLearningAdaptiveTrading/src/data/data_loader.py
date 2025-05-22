import pandas as pd
import os

# Dummy logger for standalone execution if actual logger isn't passed
class SimpleLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")

def load_data(file_path, logger=None, date_column=None, expected_columns=None): # Added logger
    """
    Loads data from a CSV file.
    """
    if logger is None: logger = SimpleLogger()

    if not os.path.exists(file_path):
        logger.error(f"Data file not found at {file_path}")
        return pd.DataFrame()

    try:
        if date_column:
            df = pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
        else:
            df = pd.read_csv(file_path)
            potential_date_cols = ['Date', 'date', 'Time', 'time', 'Timestamp', 'timestamp']
            inferred_date_col = None
            for col in potential_date_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        inferred_date_col = col
                        logger.info(f"Inferred and set '{inferred_date_col}' as DateTimeIndex for {file_path}.")
                        break
                    except Exception as e:
                        logger.warning(f"Could not parse column '{col}' as datetime in {file_path}: {e}")
            if not inferred_date_col and not isinstance(df.index, pd.DatetimeIndex):
                 logger.warning(f"Could not automatically infer a date column or set a DatetimeIndex for {file_path}.")
        
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")

        if expected_columns:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing expected columns in loaded data from {file_path}: {missing_cols}")
                raise ValueError(f"Missing expected columns in loaded data: {missing_cols}")
        
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            
        return df

    except FileNotFoundError: # Should be caught by os.path.exists, but good for robustness
        logger.error(f"Data file not found at {file_path} (exception catch).")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return pd.DataFrame()


def split_data(df, train_period_config, test_period_config, validation_period_config=None, logger=None): # Added logger
    """
    Splits the DataFrame into training, testing, and optionally validation sets.
    """
    if logger is None: logger = SimpleLogger()

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame must have a DatetimeIndex for date-based splitting.")
        raise ValueError("DataFrame must have a DatetimeIndex for date-based splitting.")

    try:
        train_start = pd.to_datetime(train_period_config['start'])
        train_end = pd.to_datetime(train_period_config['end'])
        test_start = pd.to_datetime(test_period_config['start'])
        test_end = pd.to_datetime(test_period_config['end'])

        train_df = df.loc[train_start:train_end].copy()
        test_df = df.loc[test_start:test_end].copy()
        
        validation_df = pd.DataFrame()
        if validation_period_config and validation_period_config.get('start') and validation_period_config.get('end'):
            val_start = pd.to_datetime(validation_period_config['start'])
            val_end = pd.to_datetime(validation_period_config['end'])
            validation_df = df.loc[val_start:val_end].copy()
            logger.info(f"Validation data: {len(validation_df)} rows from {val_start.date()} to {val_end.date()}")
        else:
            logger.info("No valid validation period configuration provided, validation_df will be empty.")


        logger.info(f"Training data: {len(train_df)} rows from {train_start.date()} to {train_end.date()}")
        logger.info(f"Test data: {len(test_df)} rows from {test_start.date()} to {test_end.date()}")
        
        if not train_df.empty and not test_df.empty:
            if train_df.index.max() >= test_df.index.min():
                logger.warning("Training data might overlap with or succeed test data. Ensure periods are distinct and correctly ordered.")
        if not validation_df.empty and not test_df.empty:
             if validation_df.index.max() >= test_df.index.min() and validation_df.index.min() < test_df.index.max():
                logger.warning("Validation data might overlap with test data. Ensure periods are distinct if strict separation is needed.")

        return train_df, validation_df, test_df

    except KeyError as e:
        logger.error(f"Missing 'start' or 'end' key in period configuration: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logger.error(f"Error splitting data: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def add_external_data(df, external_data_sources_config, logger=None): # Added logger, changed arg name
    """
    Merges external data sources specified in the configuration.
    Example: external_data_sources_config = [{'name': 'yields', 'path': 'path/to/yields.csv', 'date_col': 'Date'}]
    """
    if logger is None: logger = SimpleLogger()
    
    if not external_data_sources_config:
        logger.info("No external data sources configured to add.")
        return df

    merged_df = df.copy()
    for source in external_data_sources_config:
        name = source.get('name')
        path = source.get('path')
        date_col = source.get('date_col', 'Date') # Default date column name for external sources
        
        if not name or not path:
            logger.warning(f"Skipping external data source due to missing name or path: {source}")
            continue
            
        logger.info(f"Loading external data source '{name}' from '{path}'...")
        ext_df = load_data(path, logger=logger, date_column=date_col) # Pass logger
        
        if not ext_df.empty:
            # Ensure no duplicate columns other than index before merge, suffix if needed
            common_cols = merged_df.columns.intersection(ext_df.columns)
            if not common_cols.empty:
                logger.warning(f"External data '{name}' has common columns with main data: {common_cols.tolist()}. Suffixing external columns with '_{name}'.")
                ext_df = ext_df.rename(columns={c: f"{c}_{name}" for c in common_cols})

            merged_df = pd.merge(merged_df, ext_df, left_index=True, right_index=True, how='left')
            # Consider ffill carefully, it might introduce lookahead if external data is sparse and not aligned
            # merged_df = merged_df.fillna(method='ffill') 
            logger.info(f"Successfully merged external data '{name}'.")
        else:
            logger.warning(f"External data source '{name}' from '{path}' was empty or failed to load.")
            
    return merged_df


if __name__ == '__main__':
    test_logger = SimpleLogger() # Use SimpleLogger for standalone test
    test_logger.info("Example Usage of data_loader.py:")

    dummy_data_file = "dummy_market_data.csv"
    dates = pd.to_datetime(['2010-01-01', '2010-01-02', '2010-01-03', '2010-01-04', '2010-01-05',
                            '2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-05'])
    dummy_df_content = pd.DataFrame({
        'Date': dates,
        'price': [100, 101, 102, 103, 104, 110, 111, 112, 113, 114],
        'volume': [10, 12, 11, 13, 10, 15, 16, 14, 17, 13]
    })
    dummy_df_content.to_csv(dummy_data_file, index=False)
    test_logger.info(f"Created dummy data file: {dummy_data_file}")

    loaded_df = load_data(dummy_data_file, logger=test_logger, date_column='Date', expected_columns=['price', 'volume'])
    if not loaded_df.empty:
        test_logger.info("Loaded DataFrame head:")
        test_logger.info(loaded_df.head().to_string())

        train_p = {'start': '2010-01-01', 'end': '2010-12-31'}
        val_p = {'start': '2010-01-03', 'end': '2010-01-05'} 
        test_p = {'start': '2011-01-01', 'end': '2011-12-31'}

        train_data, val_data, test_data = split_data(loaded_df, train_p, test_p, val_p, logger=test_logger)

        test_logger.info(f"Train data shape: {train_data.shape}")
        if not train_data.empty: test_logger.info(train_data.head(2).to_string())
        test_logger.info(f"Validation data shape: {val_data.shape}")
        if not val_data.empty: test_logger.info(val_data.head(2).to_string())
        test_logger.info(f"Test data shape: {test_data.shape}")
        if not test_data.empty: test_logger.info(test_data.head(2).to_string())
    else:
        test_logger.error("Failed to load dummy data.")

    if os.path.exists(dummy_data_file):
        os.remove(dummy_data_file)
        test_logger.info(f"Removed dummy data file: {dummy_data_file}")
