"""
Pattern Recognition tools for the Pipeline data analysis agent.
Handles clustering, correlations, trends, and temporal pattern analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .utils.data_access import get_dataset

def analyze_seasonal_patterns(tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze seasonal patterns in gas flows (always deduplicated for accuracy)."""
    df = get_dataset(remove_duplicates=True)

    # Add seasonal columns
    df_temp = df.copy()
    df_temp['month'] = df_temp['eff_gas_day'].dt.month
    df_temp['season'] = df_temp['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    # Calculate seasonal volumes
    seasonal_volumes = df_temp.groupby('season')['scheduled_quantity'].agg(['sum', 'mean', 'count']).round(2)

    # Calculate by transaction type
    receipts_seasonal = df_temp[df_temp['rec_del_sign'] == 1].groupby('season')['scheduled_quantity'].sum()
    deliveries_seasonal = df_temp[df_temp['rec_del_sign'] == -1].groupby('season')['scheduled_quantity'].sum()

    # Format results
    seasonal_summary = {}
    for season in seasonal_volumes.index:
        seasonal_summary[season] = {
            'total_volume': float(seasonal_volumes.loc[season, 'sum']),
            'average_transaction': float(seasonal_volumes.loc[season, 'mean']),
            'transaction_count': int(seasonal_volumes.loc[season, 'count']),
            'receipts_volume': float(receipts_seasonal.get(season, 0)),
            'deliveries_volume': float(abs(deliveries_seasonal.get(season, 0)))
        }

    # Find peak season
    peak_season = seasonal_volumes['sum'].idxmax()

    return {
        "data": {
            "seasonal_patterns": seasonal_summary,
            "peak_season": peak_season,
            "peak_volume": float(seasonal_volumes.loc[peak_season, 'sum'])
        },
        "metadata": {
            "method_used": "Seasonal aggregation by calendar months",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity", "rec_del_sign"],
            "total_rows_before_filter": len(df),
            "seasonal_breakdown": {season: int(seasonal_volumes.loc[season, 'count']) for season in seasonal_volumes.index},
            "duplicates_removed": True
        }
    }

def analyze_monthly_trends(tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze monthly trends in pipeline operations (always deduplicated for accuracy)."""
    df = get_dataset(remove_duplicates=True)

    # Create monthly aggregates
    df_temp = df.copy()
    df_temp['year_month'] = df_temp['eff_gas_day'].dt.to_period('M')

    monthly_stats = df_temp.groupby('year_month').agg({
        'scheduled_quantity': ['sum', 'mean', 'count'],
        'loc_name': 'nunique',
        'pipeline_name': 'nunique',
        'connecting_entity': 'nunique'
    }).round(2)

    # Flatten column names
    monthly_stats.columns = ['total_volume', 'avg_transaction', 'transaction_count', 'unique_locations', 'unique_pipelines', 'unique_entities']

    # Calculate month-over-month growth
    monthly_stats['volume_growth'] = monthly_stats['total_volume'].pct_change() * 100

    # Get recent trends (last 12 months if available)
    recent_months = monthly_stats.tail(min(12, len(monthly_stats)))

    # Find highest and lowest months
    peak_month = monthly_stats['total_volume'].idxmax()
    low_month = monthly_stats['total_volume'].idxmin()

    return {
        "data": {
            "monthly_trends": recent_months.to_dict('index'),
            "peak_month": str(peak_month),
            "peak_volume": float(monthly_stats.loc[peak_month, 'total_volume']),
            "lowest_month": str(low_month),
            "lowest_volume": float(monthly_stats.loc[low_month, 'total_volume']),
            "average_monthly_growth": float(recent_months['volume_growth'].mean()) if len(recent_months) > 1 else 0
        },
        "metadata": {
            "method_used": "Monthly time series aggregation",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity", "loc_name", "pipeline_name", "connecting_entity"],
            "total_rows_before_filter": len(df),
            "months_analyzed": len(monthly_stats),
            "recent_months_shown": len(recent_months),
            "duplicates_removed": True
        }
    }

def find_consistent_pipelines(tool_context: ToolContext) -> Dict[str, Any]:
    """Find pipelines with most consistent volumes (always deduplicated for accuracy)."""

    df = get_dataset(remove_duplicates=True)

    # Calculate monthly volumes by pipeline
    df_temp = df.copy()
    df_temp['year_month'] = df_temp['eff_gas_day'].dt.to_period('M')

    pipeline_monthly = df_temp.groupby(['pipeline_name', 'year_month'])['scheduled_quantity'].sum().reset_index()

    # Calculate consistency metrics for each pipeline
    consistency_metrics = []
    for pipeline in pipeline_monthly['pipeline_name'].unique():
        pipeline_data = pipeline_monthly[pipeline_monthly['pipeline_name'] == pipeline]['scheduled_quantity']

        if len(pipeline_data) >= 3:  # Need at least 3 months for meaningful consistency
            mean_vol = pipeline_data.mean()
            std_vol = pipeline_data.std()
            cv = (std_vol / mean_vol) * 100 if mean_vol > 0 else float('inf')  # Coefficient of variation

            consistency_metrics.append({
                'pipeline_name': pipeline,
                'mean_monthly_volume': float(mean_vol),
                'std_monthly_volume': float(std_vol),
                'coefficient_of_variation': float(cv),
                'months_active': len(pipeline_data),
                'consistency_score': float(1 / (1 + cv/100))  # Higher score = more consistent
            })

    # Sort by consistency score (lower CV = more consistent)
    consistency_df = pd.DataFrame(consistency_metrics)
    if len(consistency_df) > 0:
        top_consistent = consistency_df.nsmallest(10, 'coefficient_of_variation')
    else:
        top_consistent = pd.DataFrame()

    return {
        "data": {
            "most_consistent_pipelines": top_consistent.to_dict('records'),
            "total_pipelines_analyzed": len(consistency_metrics)
        },
        "metadata": {
            "method_used": "Coefficient of variation analysis on monthly pipeline volumes",
            "columns_accessed": ["pipeline_name", "eff_gas_day", "scheduled_quantity"],
            "total_rows_before_filter": len(df),
            "consistency_metric": "Lower coefficient of variation indicates higher consistency",
            "duplicates_removed": True
        }
    }

def analyze_receipts_deliveries_patterns(tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze if receipts and deliveries follow similar patterns (always deduplicated for accuracy)."""

    df = get_dataset(remove_duplicates=True)

    # Create monthly time series for receipts and deliveries
    df_temp = df.copy()
    df_temp['year_month'] = df_temp['eff_gas_day'].dt.to_period('M')

    monthly_patterns = df_temp.groupby(['year_month', 'rec_del_sign'])['scheduled_quantity'].sum().unstack(fill_value=0)
    monthly_patterns.columns = ['Deliveries', 'Receipts']
    monthly_patterns['Deliveries'] = abs(monthly_patterns['Deliveries'])  # Make deliveries positive for comparison

    # Calculate correlation
    correlation = monthly_patterns['Receipts'].corr(monthly_patterns['Deliveries'])

    # Calculate balance (net flow) over time
    monthly_patterns['Net_Flow'] = monthly_patterns['Receipts'] - monthly_patterns['Deliveries']

    # Get recent patterns (last 12 months)
    recent_patterns = monthly_patterns.tail(min(12, len(monthly_patterns)))

    return {
        "data": {
            "correlation_coefficient": float(correlation),
            "correlation_strength": "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak",
            "recent_patterns": recent_patterns.to_dict('index'),
            "average_net_flow": float(monthly_patterns['Net_Flow'].mean()),
            "net_flow_volatility": float(monthly_patterns['Net_Flow'].std())
        },
        "metadata": {
            "method_used": "Monthly correlation analysis between receipts and deliveries",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity", "rec_del_sign"],
            "total_rows_before_filter": len(df),
            "months_analyzed": len(monthly_patterns),
            "correlation_interpretation": f"r={correlation:.3f}, indicating {'positive' if correlation > 0 else 'negative'} relationship",
            "duplicates_removed": True
        }
    }

def analyze_activity_over_time(tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze how pipeline activity has changed over time (always deduplicated for accuracy)."""

    df = get_dataset(remove_duplicates=True)

    # Create quarterly activity metrics
    df_temp = df.copy()
    df_temp['quarter'] = df_temp['eff_gas_day'].dt.to_period('Q')

    quarterly_activity = df_temp.groupby('quarter').agg({
        'scheduled_quantity': ['sum', 'mean'],
        'loc_name': 'nunique',
        'pipeline_name': 'nunique',
        'connecting_entity': 'nunique'
    }).round(2)

    # Flatten column names
    quarterly_activity.columns = ['total_volume', 'avg_transaction_size', 'active_locations', 'active_pipelines', 'active_entities']

    # Calculate growth rates
    quarterly_activity['volume_growth'] = quarterly_activity['total_volume'].pct_change() * 100
    quarterly_activity['location_growth'] = quarterly_activity['active_locations'].pct_change() * 100

    # Identify trends
    recent_quarters = quarterly_activity.tail(8)  # Last 2 years
    volume_trend = "Increasing" if recent_quarters['volume_growth'].mean() > 0 else "Decreasing"
    activity_trend = "Expanding" if recent_quarters['location_growth'].mean() > 0 else "Contracting"

    return {
        "data": {
            "quarterly_activity": recent_quarters.to_dict('index'),
            "volume_trend": volume_trend,
            "activity_trend": activity_trend,
            "average_quarterly_volume_growth": float(recent_quarters['volume_growth'].mean()),
            "average_quarterly_location_growth": float(recent_quarters['location_growth'].mean())
        },
        "metadata": {
            "method_used": "Quarterly activity trend analysis",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity", "loc_name", "pipeline_name", "connecting_entity"],
            "total_rows_before_filter": len(df),
            "quarters_analyzed": len(quarterly_activity),
            "recent_quarters_shown": len(recent_quarters),
            "duplicates_removed": True
        }
    }

def calculate_correlation(tool_context: ToolContext, x_metric: str, y_metric: str,
                         time_granularity: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate correlation between two metrics over time (always deduplicated for accuracy).

    Args:
        x_metric: First metric ('volume', 'active_locations', 'transaction_count', 'unique_pipelines')
        y_metric: Second metric ('volume', 'active_locations', 'transaction_count', 'unique_pipelines')
        time_granularity: Time grouping ('daily', 'weekly', 'monthly', 'quarterly')
        filters: Optional filters to apply (year, state, category, transaction_type, etc.)
    """

    df = get_dataset(remove_duplicates=True)
    df_filtered = df.copy()
    filters_applied = []

    # Apply filters
    for column, condition in filters.items():
        if column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        elif column == 'category':
            df_filtered = df_filtered[df_filtered['category_short'] == condition]
            filters_applied.append(f"category_short == {condition}")
        elif column == 'transaction_type':
            if condition.lower() == 'receipts':
                df_filtered = df_filtered[df_filtered['rec_del_sign'] == 1]
                filters_applied.append("rec_del_sign == 1 (receipts)")
            elif condition.lower() == 'deliveries':
                df_filtered = df_filtered[df_filtered['rec_del_sign'] == -1]
                filters_applied.append("rec_del_sign == -1 (deliveries)")
        else:
            df_filtered = df_filtered[df_filtered[column] == condition]
            filters_applied.append(f"{column} == {condition}")

    # Create time grouping
    if time_granularity.lower() == 'daily':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.date
    elif time_granularity.lower() == 'weekly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.isocalendar().week.astype(str) + '-' + df_filtered['eff_gas_day'].dt.year.astype(str)
    elif time_granularity.lower() == 'monthly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.to_period('M')
    elif time_granularity.lower() == 'quarterly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.to_period('Q')
    else:
        return {"error": f"Unsupported time granularity: {time_granularity}"}

    # Calculate metrics by time period
    def calculate_metric_series(metric_name):
        if metric_name == 'volume':
            return df_filtered.groupby('time_key')['scheduled_quantity'].sum()
        elif metric_name == 'active_locations':
            return df_filtered.groupby('time_key')['loc_name'].nunique()
        elif metric_name == 'transaction_count':
            return df_filtered.groupby('time_key').size()
        elif metric_name == 'unique_pipelines':
            return df_filtered.groupby('time_key')['pipeline_name'].nunique()
        elif metric_name == 'unique_entities':
            return df_filtered.groupby('time_key')['connecting_entity'].nunique()
        elif metric_name == 'average_quantity':
            return df_filtered.groupby('time_key')['scheduled_quantity'].mean()
        else:
            return None

    # Get time series for both metrics
    x_series = calculate_metric_series(x_metric)
    y_series = calculate_metric_series(y_metric)

    if x_series is None or y_series is None:
        return {
            "error": f"Unsupported metrics. Available: volume, active_locations, transaction_count, unique_pipelines, unique_entities, average_quantity",
            "metadata": {"filters_applied": filters_applied}
        }

    # Align the two time series (keep only common dates)
    aligned_data = pd.DataFrame({'x': x_series, 'y': y_series}).dropna()

    if len(aligned_data) < 2:
        return {
            "error": "Insufficient data points for correlation analysis",
            "metadata": {"filters_applied": filters_applied, "data_points": len(aligned_data)}
        }

    # Calculate correlation
    correlation_coeff = float(aligned_data['x'].corr(aligned_data['y']))

    # Calculate additional statistics
    x_values = aligned_data['x'].values
    y_values = aligned_data['y'].values

    # Linear regression for trend line
    slope, intercept = np.polyfit(x_values, y_values, 1)
    r_squared = float(correlation_coeff ** 2)

    # Statistical significance (basic)
    n = len(aligned_data)
    if n > 2:
        t_statistic = correlation_coeff * np.sqrt((n - 2) / (1 - correlation_coeff ** 2))
        # Approximate p-value for |t| > 2 (95% confidence)
        p_value_approx = "< 0.05" if abs(t_statistic) > 2 else "> 0.05"
    else:
        t_statistic = 0
        p_value_approx = "N/A"

    # Interpretation
    if abs(correlation_coeff) > 0.8:
        strength = "Very Strong"
    elif abs(correlation_coeff) > 0.6:
        strength = "Strong"
    elif abs(correlation_coeff) > 0.4:
        strength = "Moderate"
    elif abs(correlation_coeff) > 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"

    direction = "Positive" if correlation_coeff > 0 else "Negative"

    # Sample of aligned data for verification
    sample_data = []
    for i, (time_key, row) in enumerate(aligned_data.head(10).iterrows()):
        sample_data.append({
            "time_period": str(time_key),
            "x_value": float(row['x']),
            "y_value": float(row['y'])
        })

    return {
        "data": {
            "correlation_coefficient": correlation_coeff,
            "r_squared": r_squared,
            "strength": strength,
            "direction": direction,
            "sample_data": sample_data
        },
        "metadata": {
            "method_used": f"Pearson correlation between {x_metric} and {y_metric} over {time_granularity} periods",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity", "loc_name", "pipeline_name", "connecting_entity"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": filters_applied,
            "time_granularity": time_granularity,
            "data_points": n,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "linear_regression": {
                "slope": float(slope),
                "intercept": float(intercept)
            },
            "statistical_test": {
                "t_statistic": float(t_statistic),
                "p_value_approx": p_value_approx,
                "significance": "Significant" if p_value_approx == "< 0.05" else "Not significant"
            },
            "duplicates_removed": True
        }
    }


def compare_before_after(tool_context: ToolContext, cutoff_date: str, metric: str,
                        time_granularity: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Compare metrics before and after a cutoff date (always deduplicated for accuracy).

    Args:
        cutoff_date: Date in YYYY-MM-DD format to split the analysis
        metric: Metric to compare ('volume', 'transaction_count', 'unique_locations', 'average_quantity')
        time_granularity: Time grouping ('daily', 'weekly', 'monthly')
        filters: Filters to apply (pipeline, state, category, transaction_type, etc.)
    """

    df = get_dataset(remove_duplicates=True)
    df_filtered = df.copy()
    filters_applied = []

    # Apply filters
    for column, condition in filters.items():
        if column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        elif column == 'category':
            df_filtered = df_filtered[df_filtered['category_short'] == condition]
            filters_applied.append(f"category_short == {condition}")
        elif column == 'transaction_type':
            if condition.lower() == 'receipts':
                df_filtered = df_filtered[df_filtered['rec_del_sign'] == 1]
                filters_applied.append("rec_del_sign == 1 (receipts)")
            elif condition.lower() == 'deliveries':
                df_filtered = df_filtered[df_filtered['rec_del_sign'] == -1]
                filters_applied.append("rec_del_sign == -1 (deliveries)")
        elif column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'month':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.month == condition]
            filters_applied.append(f"eff_gas_day month == {condition}")
        else:
            df_filtered = df_filtered[df_filtered[column] == condition]
            filters_applied.append(f"{column} == {condition}")

    # Convert cutoff_date to datetime
    try:
        cutoff_dt = pd.to_datetime(cutoff_date)
    except:
        return {"error": f"Invalid date format: {cutoff_date}. Use YYYY-MM-DD format."}

    # Split data into before and after
    before_data = df_filtered[df_filtered['eff_gas_day'] < cutoff_dt].copy()
    after_data = df_filtered[df_filtered['eff_gas_day'] >= cutoff_dt].copy()

    if len(before_data) == 0:
        return {"error": f"No data found before {cutoff_date}"}
    if len(after_data) == 0:
        return {"error": f"No data found after {cutoff_date}"}

    # Create time grouping for both periods
    def add_time_key(data):
        if time_granularity.lower() == 'daily':
            data['time_key'] = data['eff_gas_day'].dt.date
        elif time_granularity.lower() == 'weekly':
            data['time_key'] = data['eff_gas_day'].dt.isocalendar().week.astype(str) + '-' + data['eff_gas_day'].dt.year.astype(str)
        elif time_granularity.lower() == 'monthly':
            data['time_key'] = data['eff_gas_day'].dt.to_period('M')
        return data

    before_data = add_time_key(before_data)
    after_data = add_time_key(after_data)

    # Calculate metric for both periods
    def calculate_metric_by_period(data, metric_name):
        if metric_name == 'volume':
            return data.groupby('time_key')['scheduled_quantity'].sum()
        elif metric_name == 'transaction_count':
            return data.groupby('time_key').size()
        elif metric_name == 'unique_locations':
            return data.groupby('time_key')['loc_name'].nunique()
        elif metric_name == 'average_quantity':
            return data.groupby('time_key')['scheduled_quantity'].mean()
        else:
            return None

    before_series = calculate_metric_by_period(before_data, metric)
    after_series = calculate_metric_by_period(after_data, metric)

    if before_series is None or after_series is None:
        return {
            "error": f"Unsupported metric: {metric}. Available: volume, transaction_count, unique_locations, average_quantity"
        }

    # Calculate summary statistics
    before_mean = float(before_series.mean())
    after_mean = float(after_series.mean())

    # Calculate percentage change
    pct_change = ((after_mean - before_mean) / before_mean * 100) if before_mean != 0 else 0

    # Statistical comparison (basic t-test approximation)
    if len(before_series) > 1 and len(after_series) > 1:
        t_stat, p_value = stats.ttest_ind(before_series, after_series)
        significance = "Significant" if p_value < 0.05 else "Not significant"
    else:
        t_stat, p_value, significance = 0, 1, "Insufficient data"

    # Sample data from each period
    before_sample = []
    for i, (time_key, value) in enumerate(before_series.head(5).items()):
        before_sample.append({
            "time_period": str(time_key),
            "value": float(value)
        })

    after_sample = []
    for i, (time_key, value) in enumerate(after_series.head(5).items()):
        after_sample.append({
            "time_period": str(time_key),
            "value": float(value)
        })

    return {
        "data": {
            "before_period": {
                "mean": before_mean,
                "count": len(before_series),
                "total_records": len(before_data),
                "date_range": f"{before_data['eff_gas_day'].min().date()} to {before_data['eff_gas_day'].max().date()}",
                "sample_data": before_sample
            },
            "after_period": {
                "mean": after_mean,
                "count": len(after_series),
                "total_records": len(after_data),
                "date_range": f"{after_data['eff_gas_day'].min().date()} to {after_data['eff_gas_day'].max().date()}",
                "sample_data": after_sample
            },
            "comparison": {
                "absolute_change": after_mean - before_mean,
                "percentage_change": pct_change,
                "direction": "Increase" if pct_change > 0 else "Decrease" if pct_change < 0 else "No change"
            }
        },
        "metadata": {
            "method_used": f"Before/after comparison of {metric} using {time_granularity} aggregation",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity", "loc_name"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": filters_applied,
            "cutoff_date": cutoff_date,
            "metric": metric,
            "time_granularity": time_granularity,
            "statistical_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significance": significance
            },
            "duplicates_removed": True
        }
    }

def perform_clustering(tool_context: ToolContext, filters: Dict[str, Any],
                      group_by_column: str, time_aggregation: str,
                      n_clusters: int, normalize: bool) -> Dict[str, Any]:
    """
    Generic clustering tool that groups data by specified dimensions and applies K-Means clustering (always deduplicated for accuracy).

    Args:
        filters: Dictionary of filters to apply (year, pipeline, state, etc.)
        group_by_column: Column to group by ('loc_name', 'connecting_entity', 'pipeline_name', etc.)
        time_aggregation: How to aggregate time ('monthly', 'quarterly', 'weekly')
        n_clusters: Number of clusters for K-Means
        normalize: Whether to normalize the data before clustering

    Returns:
        Dict with cluster assignments, centroids, and analysis metrics
    """
    df = get_dataset(remove_duplicates=True)
    df_filtered = df.copy()
    filters_applied = []

    # Apply filters
    for column, condition in filters.items():
        if column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        elif column == 'transaction_type':
            sign_value = 1 if condition == 'receipts' else -1 if condition == 'deliveries' else condition
            df_filtered = df_filtered[df_filtered['rec_del_sign'] == sign_value]
            filters_applied.append(f"rec_del_sign == {sign_value}")
        else:
            df_filtered = df_filtered[df_filtered[column] == condition]
            filters_applied.append(f"{column} == {condition}")

    if len(df_filtered) == 0:
        return {"error": "No data matches the specified filters"}

    # Create time-based aggregation
    if time_aggregation.lower() == 'monthly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.month
        time_range = range(1, 13)
    elif time_aggregation.lower() == 'quarterly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.quarter
        time_range = range(1, 5)
    elif time_aggregation.lower() == 'weekly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.week
        time_range = range(1, 54)
    else:
        return {"error": f"Unsupported time_aggregation: {time_aggregation}"}

    # Aggregate by group and time
    profiles = df_filtered.groupby([group_by_column, 'time_key'])['scheduled_quantity'].sum().unstack(fill_value=0)

    # Ensure we have all time periods
    for period in time_range:
        if period not in profiles.columns:
            profiles[period] = 0

    # Sort columns to ensure consistent order
    profiles = profiles.reindex(sorted(profiles.columns), axis=1)

    # Remove entities with all zero values
    profiles = profiles[(profiles != 0).any(axis=1)]

    if len(profiles) < n_clusters:
        return {"error": f"Not enough entities ({len(profiles)}) for {n_clusters} clusters"}

    # Normalize profiles if requested
    if normalize:
        scaler = StandardScaler()
        clustering_data = scaler.fit_transform(profiles)
    else:
        clustering_data = profiles.values

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_data)

    # Calculate silhouette score for cluster quality assessment
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(clustering_data, cluster_labels)
    else:
        silhouette_avg = 0.0

    # Create cluster assignments
    cluster_assignments = pd.DataFrame({
        group_by_column: profiles.index,
        'cluster': cluster_labels
    })

    # Calculate cluster centroids and statistics
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_data = profiles.iloc[cluster_labels == cluster_id]

        if len(cluster_data) > 0:
            centroid = cluster_data.mean()
            total_volume = cluster_data.sum(axis=1)

            cluster_stats[f"cluster_{cluster_id}"] = {
                "entity_count": len(cluster_data),
                "sample_entities": cluster_data.index.tolist()[:5],  # Show only top 5 examples
                "avg_total_volume": float(total_volume.mean()),
                "peak_period": int(centroid.idxmax()),
                "peak_volume": float(centroid.max()),
                "min_period": int(centroid.idxmin()),
                "min_volume": float(centroid.min()),
                "coefficient_of_variation": float(centroid.std() / centroid.mean()) if centroid.mean() != 0 else 0,
                "usage_pattern": "High-volume winter peak" if centroid.idxmax() in [12, 1, 2] else
                                "High-volume summer peak" if centroid.idxmax() in [6, 7, 8] else
                                "Consistent year-round usage"
            }

    return {
        "data": {
            "cluster_statistics": cluster_stats,
            "clustering_quality": {
                "silhouette_score": float(silhouette_avg),
                "quality_interpretation": "Excellent" if silhouette_avg > 0.7 else
                                        "Good" if silhouette_avg > 0.5 else
                                        "Fair" if silhouette_avg > 0.25 else "Poor"
            }
        },
        "metadata": {
            "method_used": f"K-Means clustering with {n_clusters} clusters on {time_aggregation} profiles",
            "columns_accessed": [group_by_column, "scheduled_quantity", "eff_gas_day"],
            "total_entities_clustered": len(profiles),
            "time_aggregation": time_aggregation,
            "group_by_column": group_by_column,
            "normalization_applied": normalize,
            "clustering_algorithm": "K-Means with random_state=42",
            "filters_applied": filters_applied,
            "feature_dimensions": len(profiles.columns),
            "duplicates_removed": True
        }
    }
