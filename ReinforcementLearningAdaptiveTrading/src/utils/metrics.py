import numpy as np
import pandas as pd

def total_return(portfolio_values_series):
    """
    Calculates the total return from a series of portfolio values.

    Args:
        portfolio_values_series (pd.Series): Time series of portfolio/equity values.

    Returns:
        float: Total return as a decimal (e.g., 0.1 for 10%).
               Returns 0.0 if series is too short or has no change.
    """
    if portfolio_values_series.empty or len(portfolio_values_series) < 2:
        return 0.0
    
    start_value = portfolio_values_series.iloc[0]
    end_value = portfolio_values_series.iloc[-1]
    
    if start_value == 0: # Avoid division by zero if starting capital is zero
        return 0.0 if end_value == 0 else np.inf # Or handle as appropriate
        
    return (end_value - start_value) / start_value


def annualized_return(returns_series, trading_days_per_year=252):
    """
    Calculates the annualized return from a series of periodic returns (e.g., daily).

    Args:
        returns_series (pd.Series): Time series of periodic returns.
                                    Assumes returns are in decimal form (e.g., 0.01 for 1%).
        trading_days_per_year (int): Number of trading days in a year.

    Returns:
        float: Annualized return as a decimal.
    """
    if returns_series.empty:
        return 0.0
    
    mean_daily_return = returns_series.mean()
    # Geometric mean is more accurate for compounding, but arithmetic mean is common for Sharpe.
    # For simplicity, using arithmetic mean here.
    # (1 + returns_series).prod()**(trading_days_per_year / returns_series.count()) - 1
    return mean_daily_return * trading_days_per_year


def annualized_volatility(returns_series, trading_days_per_year=252):
    """
    Calculates the annualized volatility from a series of periodic returns.

    Args:
        returns_series (pd.Series): Time series of periodic returns.
        trading_days_per_year (int): Number of trading days in a year.

    Returns:
        float: Annualized volatility (standard deviation of returns).
    """
    if returns_series.empty or len(returns_series) < 2 : # Need at least 2 returns for std dev
        return 0.0
        
    return returns_series.std() * np.sqrt(trading_days_per_year)


def sharpe_ratio(returns_series, risk_free_rate=0.0, trading_days_per_year=252):
    """
    Calculates the Sharpe ratio from a series of periodic returns.

    Args:
        returns_series (pd.Series): Time series of periodic returns.
        risk_free_rate (float): Annual risk-free rate (decimal form).
        trading_days_per_year (int): Number of trading days in a year.

    Returns:
        float: Sharpe ratio. Returns np.nan if volatility is zero and excess return is non-zero.
               Returns 0.0 if volatility is zero and excess return is also zero.
    """
    if returns_series.empty or len(returns_series) < 2:
        return 0.0 # Or np.nan, depending on desired behavior for insufficient data

    ann_return = annualized_return(returns_series, trading_days_per_year)
    ann_vol = annualized_volatility(returns_series, trading_days_per_year)
    
    # Daily risk-free rate (assuming risk_free_rate is annual)
    # Not directly subtracting daily risk-free rate from daily returns here,
    # instead using annualized excess return.
    excess_return = ann_return - risk_free_rate
    
    if ann_vol == 0:
        if excess_return == 0:
            return 0.0 # No risk, no excess return
        else:
            # Positive excess return with zero volatility -> infinite Sharpe (practically)
            # Negative excess return with zero volatility -> negative infinite Sharpe
            # Can return np.nan or a large number, or handle as error.
            return np.nan # Or np.inf * np.sign(excess_return)
            
    return excess_return / ann_vol


def max_drawdown(portfolio_values_series):
    """
    Calculates the maximum drawdown from a series of portfolio values.

    Args:
        portfolio_values_series (pd.Series): Time series of portfolio/equity values.

    Returns:
        float: Maximum drawdown as a positive decimal (e.g., 0.2 for 20% drawdown).
               Returns 0.0 if series is empty or has no drawdowns.
    """
    if portfolio_values_series.empty:
        return 0.0

    # Calculate the running maximum
    running_max = portfolio_values_series.cummax()
    # Calculate drawdown series (difference between running max and current value)
    drawdown_series = running_max - portfolio_values_series
    # Calculate drawdown percentage series
    drawdown_pct_series = drawdown_series / running_max # Divide by zero if running_max is 0
    
    # Handle cases where running_max is 0 to avoid NaN/inf
    drawdown_pct_series[running_max == 0] = 0.0 

    # Get the maximum of the drawdown percentage series
    max_dd = drawdown_pct_series.max()
    
    return max_dd if pd.notnull(max_dd) else 0.0


def calmar_ratio(portfolio_values_series, risk_free_rate=0.0, trading_days_per_year=252):
    """
    Calculates the Calmar ratio.
    Calmar Ratio = Annualized Return / Maximum Drawdown.

    Args:
        portfolio_values_series (pd.Series): Time series of portfolio values.
        risk_free_rate (float): Annual risk-free rate for calculating excess annualized return.
        trading_days_per_year (int): Number of trading days in a year.

    Returns:
        float: Calmar ratio.
    """
    if portfolio_values_series.empty or len(portfolio_values_series) < 2:
        return 0.0

    daily_returns = portfolio_values_series.pct_change().dropna()
    if daily_returns.empty:
        return 0.0

    ann_return = annualized_return(daily_returns, trading_days_per_year)
    # excess_ann_return = ann_return - risk_free_rate # Some definitions use excess return
    
    max_dd = max_drawdown(portfolio_values_series)
    
    if max_dd == 0:
        if ann_return > 0: return np.inf # Positive return with no drawdown
        return 0.0 # Zero return with no drawdown
        
    return ann_return / max_dd


def sortino_ratio(returns_series, required_return=0.0, trading_days_per_year=252):
    """
    Calculates the Sortino ratio.
    Sortino Ratio = (Annualized Return - Annualized Required Return) / Annualized Downside Deviation.

    Args:
        returns_series (pd.Series): Time series of periodic returns.
        required_return (float): Annual required rate of return or MAR (Minimum Acceptable Return).
        trading_days_per_year (int): Number of trading days in a year.

    Returns:
        float: Sortino ratio.
    """
    if returns_series.empty or len(returns_series) < 2:
        return 0.0

    ann_return = annualized_return(returns_series, trading_days_per_year)
    
    # Calculate downside deviation
    # First, filter for returns below the target (daily equivalent of required_return)
    target_daily_return = (1 + required_return)**(1/trading_days_per_year) - 1
    downside_returns = returns_series[returns_series < target_daily_return]
    
    if downside_returns.empty: # No returns below target, so downside deviation is 0
        if (ann_return - required_return) > 0: return np.inf
        return 0.0 # Or np.nan if preferred when no downside risk is observed

    # Calculate variance of downside returns relative to target
    downside_deviation_sq = (downside_returns - target_daily_return).pow(2).sum() / len(returns_series) # Using full length for average
    # downside_deviation_sq = (downside_returns - target_daily_return).pow(2).mean() # Using only downside length
    
    annualized_downside_deviation = np.sqrt(downside_deviation_sq) * np.sqrt(trading_days_per_year)

    if annualized_downside_deviation == 0:
        if (ann_return - required_return) > 0: return np.inf
        return 0.0 # Or np.nan
        
    return (ann_return - required_return) / annualized_downside_deviation


if __name__ == '__main__':
    print("Example Usage of metrics.py:")

    # Create dummy data: 1 year of daily returns, 10% annualized return, 15% annualized vol
    np.random.seed(42)
    days = 252
    daily_mean_ret = 0.10 / days
    daily_std_dev = 0.15 / np.sqrt(days)
    
    sample_returns = pd.Series(np.random.normal(loc=daily_mean_ret, scale=daily_std_dev, size=days))
    
    # Create portfolio values from returns (starting at $1000)
    initial_pv = 1000
    sample_portfolio_values = pd.Series([initial_pv] * (days + 1))
    for i in range(days):
        sample_portfolio_values[i+1] = sample_portfolio_values[i] * (1 + sample_returns[i])

    print(f"\nSample Daily Returns (first 5):\n{sample_returns.head()}")
    print(f"Sample Portfolio Values (first 5):\n{sample_portfolio_values.head()}")

    # Test metrics
    tr = total_return(sample_portfolio_values)
    ar = annualized_return(sample_returns)
    av = annualized_volatility(sample_returns)
    sr = sharpe_ratio(sample_returns, risk_free_rate=0.01) # Assume 1% risk-free rate
    mdd = max_drawdown(sample_portfolio_values)
    cr = calmar_ratio(sample_portfolio_values)
    sor = sortino_ratio(sample_returns, required_return=0.02) # Assume 2% MAR

    print(f"\nCalculated Metrics:")
    print(f"Total Return: {tr*100:.2f}%")
    print(f"Annualized Return: {ar*100:.2f}%")
    print(f"Annualized Volatility: {av*100:.2f}%")
    print(f"Sharpe Ratio (vs 1% RF): {sr:.2f}")
    print(f"Max Drawdown: {mdd*100:.2f}%")
    print(f"Calmar Ratio: {cr:.2f}")
    print(f"Sortino Ratio (vs 2% MAR): {sor:.2f}")

    # Test edge cases for max_drawdown
    no_drawdown_pv = pd.Series([100, 110, 120, 130])
    print(f"Max Drawdown (no drawdown): {max_drawdown(no_drawdown_pv)*100:.2f}%")
    drawdown_at_end_pv = pd.Series([100, 110, 90, 95])
    print(f"Max Drawdown (drawdown at end): {max_drawdown(drawdown_at_end_pv)*100:.2f}%")
    empty_pv = pd.Series(dtype=float)
    print(f"Max Drawdown (empty series): {max_drawdown(empty_pv)*100:.2f}%")
    
    # Test Sharpe with zero volatility
    zero_vol_returns_positive = pd.Series([0.001, 0.001, 0.001]) # Positive return, zero vol
    print(f"Sharpe (zero vol, positive return): {sharpe_ratio(zero_vol_returns_positive)}") 
    zero_vol_returns_zero = pd.Series([0.0, 0.0, 0.0]) # Zero return, zero vol
    print(f"Sharpe (zero vol, zero return): {sharpe_ratio(zero_vol_returns_zero)}")
