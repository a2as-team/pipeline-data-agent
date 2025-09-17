"""
Basic Retrieval and Analysis tools for the Pipeline data analysis agent.
This module contains fundamental data retrieval and analysis functions that provide
basic information about the dataset and perform common analytical queries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
import warnings
warnings.filterwarnings('ignore')

from .utils.data_access import get_dataset

def get_dataset_info(tool_context: ToolContext) -> Dict[str, Any]:
    """Get comprehensive information about the dataset including schema, statistics, and data quality."""

    df = get_dataset(remove_duplicates=None)  # Keep duplicates for data quality assessment

    # Basic information
    info = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": df.columns.tolist(),
        "data_types": {col: str(df[col].dtype) for col in df.columns},
    }

    # Data quality assessment
    missing_data = {}
    duplicate_info = {"total_duplicates": int(df.duplicated().sum())}

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_data[col] = {
            "missing_count": int(missing_count),
            "missing_percentage": round(float(missing_pct), 2)
        }

    info["missing_data"] = missing_data
    info["duplicates"] = duplicate_info

    # Basic numeric column info
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    info["numeric_columns"] = numeric_cols
    info["numeric_column_count"] = len(numeric_cols)

    # Categorical column info
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    info["categorical_columns"] = categorical_cols
    info["categorical_column_count"] = len(categorical_cols)

    return info

def query_total_volume(tool_context: ToolContext, year: str) -> Dict[str, Any]:
    """Get total volume for a specific year (always deduplicated for accuracy).

    Args:
        year: Year to filter data (e.g., '2024')
    """
    df = get_dataset(remove_duplicates=True)

    # Filter by year
    df_filtered = df[df['eff_gas_day'].dt.year == int(year)]
    total_volume = float(df_filtered['scheduled_quantity'].sum())

    return {
        "data": total_volume,
        "metadata": {
            "method_used": f"SUM aggregation on scheduled_quantity for year {year}",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": [f"eff_gas_day year == {year}"],
            "duplicates_removed": True
        }
    }

def count_unique_pipelines(tool_context: ToolContext) -> Dict[str, Any]:
    """Count unique pipelines in the dataset."""

    df = get_dataset(remove_duplicates=None)  # Unique count not affected by duplicates
    unique_count = int(df['pipeline_name'].nunique())

    return {
        "data": unique_count,
        "metadata": {
            "method_used": "DISTINCT COUNT on pipeline_name column",
            "columns_accessed": ["pipeline_name"],
            "total_rows_after_filter": len(df),
            "total_rows_before_filter": len(df),
            "filters_applied": [],
            "duplicates_removed": False,
        }
    }

def find_top_state_by_activity(tool_context: ToolContext) -> Dict[str, Any]:
    """Find the state with the most pipeline activity."""

    df = get_dataset(remove_duplicates=None)  # Count all activity records
    state_activity = df['state_abb'].value_counts()
    top_state = state_activity.index[0]
    top_count = int(state_activity.iloc[0])

    return {
        "data": f"{top_state}: {top_count}",
        "metadata": {
            "method_used": "GROUP BY state_abb with COUNT aggregation",
            "columns_accessed": ["state_abb"],
            "total_rows_after_filter": len(df),
            "total_rows_before_filter": len(df),
            "filters_applied": [],
            "duplicates_removed": False,
        }
    }

def get_top_pipelines_by_volume(tool_context: ToolContext, limit: int) -> Dict[str, Any]:
    """Get top N pipelines by total volume (always deduplicated for accuracy).

    Args:
        limit: Number of top pipelines to return
    """
    df = get_dataset(remove_duplicates=True)
    pipeline_volumes = df.groupby('pipeline_name')['scheduled_quantity'].sum().sort_values(ascending=False).head(limit)

    return {
        "data": pipeline_volumes.to_dict(),
        "metadata": {
            "method_used": f"GROUP BY pipeline_name with SUM aggregation, ordered DESC, limited to {limit}",
            "columns_accessed": ["pipeline_name", "scheduled_quantity"],
            "total_rows_after_filter": len(df),
            "total_rows_before_filter": len(df),
            "filters_applied": [],
            "duplicates_removed": True
        }
    }

def calculate_average_quantity(tool_context: ToolContext) -> Dict[str, Any]:
    """Calculate average scheduled quantity per transaction (always deduplicated for accuracy)."""
    df = get_dataset(remove_duplicates=True)
    avg_quantity = float(df['scheduled_quantity'].mean())
    median_quantity = float(df['scheduled_quantity'].median())

    return {
        "data": avg_quantity,
        "metadata": {
            "method_used": "MEAN aggregation on scheduled_quantity",
            "columns_accessed": ["scheduled_quantity"],
            "total_rows_after_filter": len(df),
            "total_rows_before_filter": len(df),
            "filters_applied": [],
            "additional_stats": f"Median: {median_quantity:.2f}",
            "duplicates_removed": True,
        }
    }

def analyze_receipts_vs_deliveries(tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze receipts vs deliveries in the dataset."""

    df = get_dataset(remove_duplicates=None)  # Count all transactions
    rec_del_counts = df['rec_del_sign'].value_counts()

    return {
        "data": rec_del_counts.to_dict(),
        "metadata": {
            "method_used": "GROUP BY rec_del_sign with COUNT aggregation",
            "columns_accessed": ["rec_del_sign"],
            "total_rows_after_filter": len(df),
            "total_rows_before_filter": len(df),
            "filters_applied": [],
            "duplicates_removed": False,
        }
    }

def calculate_net_flows(tool_context: ToolContext, year: int, group_by_column: str) -> Dict[str, Any]:
    """Calculate net flows grouped by specified column for a given year (always deduplicated for accuracy).

    Args:
        year: Year to analyze
        group_by_column: Column to group by ('pipeline_name', 'state_abb', etc.)
    """
    df = get_dataset(remove_duplicates=True)

    # Filter by year
    df_filtered = df[df['eff_gas_day'].dt.year == year].copy()

    # Calculate net flows (scheduled_quantity * rec_del_sign)
    df_filtered['net_flow'] = df_filtered['scheduled_quantity'] * df_filtered['rec_del_sign']

    # Group by specified column and sum net flows
    net_flows = df_filtered.groupby(group_by_column)['net_flow'].sum().sort_values(ascending=False)

    # Convert to native Python types
    net_flows_dict = {str(key): float(value) for key, value in net_flows.head(20).items()}

    return {
        "data": net_flows_dict,
        "metadata": {
            "method_used": f"Net flow calculation: scheduled_quantity * rec_del_sign, grouped by {group_by_column}",
            "columns_accessed": [group_by_column, "scheduled_quantity", "rec_del_sign", "eff_gas_day"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": [f"eff_gas_day year == {year}"],
            "grouping_column": group_by_column,
            "year_analyzed": year,
            "calculation_note": "Positive values = net sources (more receipts in), Negative values = net sinks (more deliveries out)",
            "duplicates_removed": True
        }
    }

def get_top_locations_by_metric(tool_context: ToolContext, year: int, transaction_type: str,
                                location_column: str, metric: str, limit: int) -> Dict[str, Any]:
    """Get top locations by specified metric (sum, mean, count) for receipts or deliveries.

    Args:
        year: Year to analyze
        transaction_type: 'receipts' or 'deliveries'
        location_column: Column to group by ('state_abb', 'loc_name', etc.)
        metric: 'sum', 'mean', 'count', or 'average_daily'
        limit: Number of top locations to return
    """

    # Smart default: deduplicate for volume metrics, keep duplicates for counts
    remove_duplicates = metric.lower() in ['sum', 'mean', 'average_daily']

    df = get_dataset(remove_duplicates=remove_duplicates)

    # Column name mapping for common aliases
    column_mappings = {
        'counterparty': 'connecting_entity',
        'counterparties': 'connecting_entity',
        'entity': 'connecting_entity',
        'state': 'state_abb',
        'location': 'loc_name',
        'pipeline': 'pipeline_name'
    }

    # Map location column name
    if location_column in column_mappings:
        location_column = column_mappings[location_column]

    # Validate location column exists
    if location_column not in df.columns:
        available_location_columns = [col for col in df.columns if 'loc' in col.lower() or 'name' in col.lower() or 'connect' in col.lower()]
        return {
            "error": f"Column '{location_column}' not found. Available columns: {available_location_columns}",
            "suggestion": "Try using 'loc_name' for location analysis or 'connecting_entity' for counterparty analysis"
        }

    # Filter by year
    df_filtered = df[df['eff_gas_day'].dt.year == year].copy()

    # Filter by transaction type
    if transaction_type.lower() == 'receipts':
        df_filtered = df_filtered[df_filtered['rec_del_sign'] == 1]
    elif transaction_type.lower() == 'deliveries':
        df_filtered = df_filtered[df_filtered['rec_del_sign'] == -1]

    # Calculate metric by location
    if metric.lower() == 'average_daily':
        # Group by location and date first, then calculate average
        daily_volumes = df_filtered.groupby([location_column, df_filtered['eff_gas_day'].dt.date])['scheduled_quantity'].sum()
        result = daily_volumes.groupby(location_column).mean().sort_values(ascending=False).head(limit)
    elif metric.lower() == 'sum':
        result = df_filtered.groupby(location_column)['scheduled_quantity'].sum().sort_values(ascending=False).head(limit)
    elif metric.lower() == 'mean':
        result = df_filtered.groupby(location_column)['scheduled_quantity'].mean().sort_values(ascending=False).head(limit)
    elif metric.lower() == 'count':
        result = df_filtered.groupby(location_column).size().sort_values(ascending=False).head(limit)
    else:
        return {"error": f"Unsupported metric: {metric}. Use 'sum', 'mean', 'count', or 'average_daily'"}

    # Convert to native Python types
    result_dict = {str(location): float(value) for location, value in result.items()}

    return {
        "data": result_dict,
        "metadata": {
            "method_used": f"{metric} calculation by {location_column} for {transaction_type} in {year}",
            "columns_accessed": [location_column, "scheduled_quantity", "rec_del_sign", "eff_gas_day"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": [f"eff_gas_day year == {year}", f"rec_del_sign for {transaction_type}"],
            "transaction_type": transaction_type,
            "metric_calculated": metric,
            "location_column": location_column,
            "limit": limit,
            "duplicates_removed": remove_duplicates
        }
    }

def aggregate_by_multiple_dimensions(tool_context: ToolContext, year: int, group_columns: List[str],
                                   time_period: str, metric: str) -> Dict[str, Any]:
    """Aggregate data by multiple dimensions with time grouping.

    Args:
        year: Year to analyze
        group_columns: List of columns to group by
        time_period: 'monthly', 'quarterly', or 'weekly'
        metric: 'sum', 'mean', 'count'
    """

    # Smart default: deduplicate for volume metrics, keep duplicates for counts
    remove_duplicates = metric.lower() in ['sum', 'mean']

    df = get_dataset(remove_duplicates=remove_duplicates)

    # Column name mapping for common aliases
    column_mappings = {
        'counterparty': 'connecting_entity',
        'counterparties': 'connecting_entity',
        'entity': 'connecting_entity',
        'state': 'state_abb',
        'location': 'loc_name',
        'pipeline': 'pipeline_name'
    }

    # Map column names
    mapped_group_columns = []
    for col in group_columns:
        mapped_col = column_mappings.get(col, col)
        mapped_group_columns.append(mapped_col)

    # Filter by year
    df_filtered = df[df['eff_gas_day'].dt.year == year].copy()

    # Check if columns exist
    missing_cols = [col for col in mapped_group_columns if col not in df_filtered.columns]
    if missing_cols:
        return {
            "error": f"Columns not found: {missing_cols}. Available columns: {list(df_filtered.columns)}",
            "metadata": {}
        }

    # Add time period column
    if time_period.lower() == 'monthly':
        df_filtered['time_period'] = df_filtered['eff_gas_day'].dt.month
        time_col = 'time_period'
    elif time_period.lower() == 'daily':
        df_filtered['time_period'] = df_filtered['eff_gas_day'].dt.date
        time_col = 'time_period'
    elif time_period.lower() == 'quarterly':
        df_filtered['time_period'] = df_filtered['eff_gas_day'].dt.quarter
        time_col = 'time_period'
    elif time_period.lower() == 'seasonal':
        df_filtered['time_period'] = df_filtered['eff_gas_day'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        time_col = 'time_period'
    else:
        time_col = None

    # Build grouping columns
    all_group_columns = [time_col] + mapped_group_columns if time_col else mapped_group_columns

    # Calculate aggregation
    if metric.lower() == 'sum':
        aggregated = df_filtered.groupby(all_group_columns)['scheduled_quantity'].sum()
    elif metric.lower() == 'mean':
        aggregated = df_filtered.groupby(all_group_columns)['scheduled_quantity'].mean()
    elif metric.lower() == 'count':
        aggregated = df_filtered.groupby(all_group_columns).size()
    else:
        return {"error": f"Unsupported metric: {metric}"}

    # Get top results
    top_results = aggregated.sort_values(ascending=False).head(20)

    # Format results
    formatted_results = []
    for index, value in top_results.items():
        if isinstance(index, tuple):
            result_dict = dict(zip(all_group_columns, index))
        else:
            result_dict = {all_group_columns[0]: index}
        result_dict['value'] = float(value)
        formatted_results.append(result_dict)

    return {
        "data": formatted_results,
        "metadata": {
            "method_used": f"{metric} aggregation by {', '.join(all_group_columns)}",
            "columns_accessed": all_group_columns + ["scheduled_quantity"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": [f"eff_gas_day year == {year}"],
            "grouping_dimensions": all_group_columns,
            "time_period": time_period,
            "metric": metric,
            "top_results_count": len(formatted_results),
            "duplicates_removed": remove_duplicates
        }
    }

def filter_and_rank(tool_context: ToolContext, filters: Dict[str, Any],
                   group_by_column: str, metric: str, limit: int) -> Dict[str, Any]:
    """Apply filters and rank by specified metric.

    Enhanced with smart duplicate handling:

    Args:
        filters: Dictionary of filters to apply. Common filters:
            - 'year': int (e.g., 2024) - filters by eff_gas_day year
            - 'state': str (e.g., 'PA') - filters by state_abb
            - 'transaction_type': str ('receipts' or 'deliveries') - filters by rec_del_sign
            - 'pipeline': str - filters by pipeline_name
        group_by_column: Column to group by ('connecting_entity', 'pipeline_name', 'state_abb', etc.)
        metric: Aggregation metric ('sum', 'mean', 'count')
        limit: Number of top results to return

    CRITICAL: Always include 'year' filter when user mentions specific years!
    """

    # Smart default: deduplicate for volume metrics, keep duplicates for counts
    remove_duplicates = metric.lower() in ['sum', 'mean', 'average']

    df = get_dataset(remove_duplicates=remove_duplicates)
    df_filtered = df.copy()
    filters_applied = []

    # Column name mapping for common aliases
    column_mappings = {
        'counterparty': 'connecting_entity',
        'counterparties': 'connecting_entity',
        'entity': 'connecting_entity',
        'state': 'state_abb',
        'location': 'loc_name',
        'pipeline': 'pipeline_name'
    }

    # Map group_by_column if needed
    if group_by_column in column_mappings:
        group_by_column = column_mappings[group_by_column]

    # Check if group_by_column exists
    if group_by_column not in df_filtered.columns:
        return {
            "error": f"Column '{group_by_column}' not found in dataset. Available columns: {list(df_filtered.columns)}",
            "metadata": {"filters_applied": []}
        }

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
        elif column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        elif column == 'season':
            # Add seasonal filtering capability
            df_filtered = df_filtered.copy()
            df_filtered['month'] = df_filtered['eff_gas_day'].dt.month
            season_mapping = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11]
            }
            if condition in season_mapping:
                df_filtered = df_filtered[df_filtered['month'].isin(season_mapping[condition])]
                filters_applied.append(f"season == {condition} (months {season_mapping[condition]})")
            else:
                return {
                    "error": f"Invalid season '{condition}'. Valid seasons: {list(season_mapping.keys())}",
                    "metadata": {"filters_applied": filters_applied}
                }
        else:
            # Generic filter with error handling
            try:
                if column not in df_filtered.columns:
                    return {
                        "error": f"Column '{column}' not found in dataset. Available columns: {list(df_filtered.columns)}",
                        "metadata": {"filters_applied": filters_applied}
                    }

                # Handle different data types appropriately
                if df_filtered[column].dtype == 'datetime64[ns]' and isinstance(condition, str):
                    # Convert string to datetime for comparison
                    condition = pd.to_datetime(condition)

                df_filtered = df_filtered[df_filtered[column] == condition]
                filters_applied.append(f"{column} == {condition}")
            except Exception as e:
                return {
                    "error": f"Error filtering by {column} == {condition}: {str(e)}",
                    "metadata": {"filters_applied": filters_applied}
                }

    if len(df_filtered) == 0:
        return {
            "data": {"error": "No data matches the specified filters"},
            "metadata": {
                "filters_applied": filters_applied,
                "total_rows_after_filter": 0
            }
        }

    # Calculate metric
    if metric.lower() == 'sum':
        result = df_filtered.groupby(group_by_column)['scheduled_quantity'].sum()
    elif metric.lower() == 'mean':
        result = df_filtered.groupby(group_by_column)['scheduled_quantity'].mean()
    elif metric.lower() == 'count':
        result = df_filtered.groupby(group_by_column).size()
    else:
        return {"error": f"Unsupported metric: {metric}"}

    # Get top results
    top_results = result.sort_values(ascending=False).head(limit)
    result_dict = {str(key): float(value) for key, value in top_results.items()}

    return {
        "data": result_dict,
        "metadata": {
            "method_used": f"{metric} calculation grouped by {group_by_column} with custom filters",
            "columns_accessed": [group_by_column, "scheduled_quantity"] + list(filters.keys()),
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": filters_applied,
            "grouping_column": group_by_column,
            "metric": metric,
            "limit": limit,
            "duplicates_removed": remove_duplicates
        }
    }

def analyze_time_series(tool_context: ToolContext, filters: Dict[str, Any],
                       time_granularity: str, metric: str) -> Dict[str, Any]:
    """Analyze time series data with specified granularity (always deduplicated for accuracy).

    Args:
        filters: Dictionary of filters to apply
        time_granularity: Time grouping ('daily', 'weekly', 'monthly', etc.)
        metric: Metric to analyze ('sum', 'mean', 'count')
    """
    df = get_dataset(remove_duplicates=True)
    df_filtered = df.copy()
    filters_applied = []

    # Apply filters
    for column, condition in filters.items():
        if column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'month':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.month == condition]
            filters_applied.append(f"eff_gas_day month == {condition}")
        elif column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        else:
            df_filtered = df_filtered[df_filtered[column] == condition]
            filters_applied.append(f"{column} == {condition}")

    if len(df_filtered) == 0:
        return {"error": "No data matches the specified filters"}

    # Create time series based on granularity
    if time_granularity.lower() == 'daily':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.date
    elif time_granularity.lower() == 'monthly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.to_period('M')
    elif time_granularity.lower() == 'yearly':
        df_filtered['time_key'] = df_filtered['eff_gas_day'].dt.year
    else:
        return {"error": f"Unsupported time granularity: {time_granularity}"}

    # Calculate net flow if needed
    if metric.lower() == 'net_flow':
        df_filtered['metric_value'] = df_filtered['scheduled_quantity'] * df_filtered['rec_del_sign']
    else:
        df_filtered['metric_value'] = df_filtered['scheduled_quantity']

    # Aggregate by time
    if metric.lower() in ['sum', 'net_flow']:
        time_series = df_filtered.groupby('time_key')['metric_value'].sum()
    elif metric.lower() == 'mean':
        time_series = df_filtered.groupby('time_key')['metric_value'].mean()
    elif metric.lower() == 'count':
        time_series = df_filtered.groupby('time_key').size()
    else:
        return {"error": f"Unsupported metric: {metric}. Use 'sum', 'net_flow', 'mean', or 'count'."}

    # Get top/peak periods
    top_periods = time_series.sort_values(ascending=False).head(10)

    # Format results
    time_series_dict = {str(key): float(value) for key, value in time_series.items()}
    top_periods_dict = {str(key): float(value) for key, value in top_periods.items()}

    return {
        "data": {
            "time_series": time_series_dict,
            "top_periods": top_periods_dict,
            "summary_stats": {
                "total_periods": len(time_series),
                "average": float(time_series.mean()),
                "maximum": float(time_series.max()),
                "minimum": float(time_series.min())
            }
        },
        "metadata": {
            "method_used": f"{metric} time series analysis with {time_granularity} granularity",
            "columns_accessed": ["eff_gas_day", "scheduled_quantity", "rec_del_sign"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": filters_applied,
            "time_granularity": time_granularity,
            "metric": metric,
            "date_range": f"{df_filtered['eff_gas_day'].min().date()} to {df_filtered['eff_gas_day'].max().date()}",
            "duplicates_removed": True
        }
    }

def compare_across_years(tool_context: ToolContext, start_year: int, end_year: int,
                        group_by_column: str, metric: str) -> Dict[str, Any]:
    """Compare metrics across multiple years with year-over-year analysis (always deduplicated for accuracy).

    Args:
        start_year: Starting year for comparison
        end_year: Ending year for comparison
        group_by_column: Column to group by
        metric: Metric to compare
    """
    df = get_dataset(remove_duplicates=True)

    # Filter for year range
    years = list(range(start_year, end_year + 1))
    df_filtered = df[df['eff_gas_day'].dt.year.isin(years)].copy()
    df_filtered['year'] = df_filtered['eff_gas_day'].dt.year

    # Calculate metric by year and group
    if metric.lower() == 'sum':
        yearly_data = df_filtered.groupby(['year', group_by_column])['scheduled_quantity'].sum()
    elif metric.lower() == 'percentage_share':
        yearly_totals = df_filtered.groupby('year')['scheduled_quantity'].sum()
        yearly_group = df_filtered.groupby(['year', group_by_column])['scheduled_quantity'].sum()
        yearly_data = yearly_group.div(yearly_totals, level='year') * 100
    else:
        yearly_data = df_filtered.groupby(['year', group_by_column])['scheduled_quantity'].sum()

    # Restructure data
    yearly_comparison = yearly_data.unstack(fill_value=0)

    # Calculate year-over-year changes
    if metric.lower() == 'percentage_share':
        yoy_changes = yearly_comparison.diff()  # Percentage point changes
    else:
        yoy_changes = yearly_comparison.pct_change() * 100  # Percentage changes

    # Format results
    comparison_data = {}
    yoy_data = {}

    for year in yearly_comparison.index:
        comparison_data[int(year)] = {str(col): float(val) for col, val in yearly_comparison.loc[year].items()}

    for year in yoy_changes.index:
        if not pd.isna(yoy_changes.loc[year].sum()):
            yoy_data[int(year)] = {str(col): float(val) if not pd.isna(val) else 0.0 for col, val in yoy_changes.loc[year].items()}

    return {
        "data": {
            "yearly_comparison": comparison_data,
            "yoy_changes": yoy_data,
            "total_by_year": {int(year): float(total) for year, total in yearly_comparison.sum(axis=1).items()}
        },
        "metadata": {
            "method_used": f"Multi-year {metric} comparison by {group_by_column} with YoY change calculation",
            "columns_accessed": ["eff_gas_day", group_by_column, "scheduled_quantity"],
            "total_rows_after_filter": len(df_filtered),
            "total_rows_before_filter": len(df),
            "filters_applied": [f"eff_gas_day year in {years}"],
            "years_analyzed": years,
            "grouping_column": group_by_column,
            "metric": metric,
            "change_calculation": "Percentage point changes" if metric == 'percentage_share' else "Percentage changes",
            "duplicates_removed": True
        }
    }
