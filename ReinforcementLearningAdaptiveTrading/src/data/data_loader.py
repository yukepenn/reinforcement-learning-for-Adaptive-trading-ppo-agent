import pandas as pd
import os

def load_data(file_path, date_column=None, expected_columns=None):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): Path to the CSV data file.
        date_column (str, optional): Name of the column to parse as dates.
                                     If None, tries to infer.
        expected_columns (list, optional): A list of column names expected to be in the CSV.
                                           Raises ValueError if any are missing.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
                      Returns an empty DataFrame if loading fails or file not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()

    try:
        if date_column:
            df = pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
        else:
            # Try to infer date column if not specified, by attempting to parse common date column names
            # This is a basic attempt; more robust date parsing might be needed.
            df = pd.read_csv(file_path)
            potential_date_cols = ['Date', 'date', 'Time', 'time', 'Timestamp', 'timestamp']
            inferred_date_col = None
            for col in potential_date_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        inferred_date_col = col
                        print(f"Inferred and set '{inferred_date_col}' as DateTimeIndex.")
                        break
                    except Exception as e:
                        print(f"Could not parse column '{col}' as datetime: {e}")
            if not inferred_date_col and not isinstance(df.index, pd.DatetimeIndex):
                 print("Warning: Could not automatically infer a date column or set a DatetimeIndex.")


        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")

        if expected_columns:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing expected columns in loaded data: {missing_cols}")
        
        # Optional: sort by date index if it's a DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            
        return df

    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()


def split_data(df, train_period_config, test_period_config, validation_period_config=None):
    """
    Splits the DataFrame into training, testing, and optionally validation sets
    based on date ranges specified in configuration dictionaries.

    Args:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex.
        train_period_config (dict): Dict with 'start' and 'end' keys for training period.
                                    Example: {'start': '2000-01-01', 'end': '2015-12-31'}
        test_period_config (dict): Dict with 'start' and 'end' keys for testing period.
        validation_period_config (dict, optional): Dict for validation period.

    Returns:
        tuple: (train_df, validation_df, test_df)
               validation_df will be an empty DataFrame if validation_period_config is None.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for date-based splitting.")

    try:
        train_start = pd.to_datetime(train_period_config['start'])
        train_end = pd.to_datetime(train_period_config['end'])
        test_start = pd.to_datetime(test_period_config['start'])
        test_end = pd.to_datetime(test_period_config['end'])

        train_df = df.loc[train_start:train_end].copy()
        test_df = df.loc[test_start:test_end].copy()
        
        validation_df = pd.DataFrame()
        if validation_period_config:
            val_start = pd.to_datetime(validation_period_config['start'])
            val_end = pd.to_datetime(validation_period_config['end'])
            validation_df = df.loc[val_start:val_end].copy()
            print(f"Validation data: {len(validation_df)} rows from {val_start.date()} to {val_end.date()}")

        print(f"Training data: {len(train_df)} rows from {train_start.date()} to {train_end.date()}")
        print(f"Test data: {len(test_df)} rows from {test_start.date()} to {test_end.date()}")
        
        # Check for data leakage (overlap between train and test)
        if not train_df.empty and not test_df.empty:
            if train_df.index.max() >= test_df.index.min():
                print("Warning: Training data might overlap with or succeed test data. Ensure periods are distinct.")
        if not validation_df.empty and not test_df.empty:
             if validation_df.index.max() >= test_df.index.min() and validation_df.index.min() < test_df.index.max() : # more robust overlap check
                print("Warning: Validation data might overlap with test data. Ensure periods are distinct if strict separation is needed.")


        return train_df, validation_df, test_df

    except KeyError as e:
        print(f"Error: Missing 'start' or 'end' key in period configuration: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Error splitting data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Placeholder for adding external data if needed
def add_external_data(df, external_data_path_dict):
    """
    Example function to merge external data sources.
    
    Args:
        df (pd.DataFrame): Main DataFrame.
        external_data_path_dict (dict): Dict where keys are names (e.g., "yield_curve")
                                        and values are paths to CSV files.
    Returns:
        pd.DataFrame: DataFrame merged with external data.
    """
    # Example: Merge yield curve data
    # if "yields" in external_data_path_dict:
    #     yield_df = load_data(external_data_path_dict["yields"], date_column='Date')
    #     df = pd.merge(df, yield_df, left_index=True, right_index=True, how='left')
    #     df = df.fillna(method='ffill') # Forward fill NaNs after merge
    print("Placeholder: add_external_data function called. Implement actual merging logic.")
    return df


if __name__ == '__main__':
    # Example Usage:
    print("\nExample Usage of data_loader.py:")

    # Create a dummy CSV file for testing
    dummy_data_file = "dummy_market_data.csv"
    dates = pd.to_datetime(['2010-01-01', '2010-01-02', '2010-01-03', '2010-01-04', '2010-01-05',
                            '2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-05'])
    dummy_df_content = pd.DataFrame({
        'Date': dates,
        'price': [100, 101, 102, 103, 104, 110, 111, 112, 113, 114],
        'volume': [10, 12, 11, 13, 10, 15, 16, 14, 17, 13]
    })
    dummy_df_content.to_csv(dummy_data_file, index=False)
    print(f"Created dummy data file: {dummy_data_file}")

    # 1. Load data
    loaded_df = load_data(dummy_data_file, date_column='Date', expected_columns=['price', 'volume'])
    if not loaded_df.empty:
        print("\nLoaded DataFrame head:")
        print(loaded_df.head())

        # 2. Define split periods
        train_p = {'start': '2010-01-01', 'end': '2010-12-31'}
        val_p = {'start': '2010-01-03', 'end': '2010-01-05'} # Example validation within train for testing split
        test_p = {'start': '2011-01-01', 'end': '2011-12-31'}

        # 3. Split data
        train_data, val_data, test_data = split_data(loaded_df, train_p, test_p, val_p)

        print(f"\nTrain data shape: {train_data.shape}")
        if not train_data.empty: print(train_data.head(2))
        print(f"Validation data shape: {val_data.shape}")
        if not val_data.empty: print(val_data.head(2))
        print(f"Test data shape: {test_data.shape}")
        if not test_data.empty: print(test_data.head(2))
        
        # Test splitting with no validation
        train_data_no_val, _, test_data_no_val = split_data(loaded_df, train_p, test_p)
        print(f"\nTrain data (no val specified) shape: {train_data_no_val.shape}")
        
    else:
        print("Failed to load dummy data.")

    # Clean up dummy file
    if os.path.exists(dummy_data_file):
        os.remove(dummy_data_file)
        print(f"Removed dummy data file: {dummy_data_file}")
