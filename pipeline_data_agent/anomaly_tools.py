"""
Anomaly Detection tools for the Pipeline data analysis agent.
Handles outlier detection, rule violations, and statistical anomalies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
import warnings
warnings.filterwarnings('ignore')

from .utils.data_access import get_dataset

def detect_business_rule_violations(tool_context: ToolContext, category: str,
                                   expected_direction: str, filters: Dict[str, Any],
                                   max_examples: int) -> Dict[str, Any]:
    """
    Detect business rule violations by checking if a category is acting contrary to expected direction.

    Args:
        category: Category to check (e.g., 'Production', 'LDC', 'Industrial', 'Storage')
        expected_direction: Expected transaction direction ('receipts' or 'deliveries')
        filters: Dictionary of additional filters to apply (year, state, etc.)
        max_examples: Maximum number of violation examples to return

    Examples:
        - category='Production', expected_direction='receipts' → finds Production acting as deliveries
        - category='LDC', expected_direction='deliveries' → finds LDC acting as receipts
        - category='Industrial', expected_direction='deliveries' → finds Industrial acting as receipts
    """
    # Handle parameter defaults
    if not filters:
        filters = {}
    if not max_examples:
        max_examples = 10
    if not category:
        return {"error": "Category parameter is required"}
    if not expected_direction:
        return {"error": "Expected direction parameter is required ('receipts' or 'deliveries')"}

    # Validate expected_direction
    if expected_direction.lower() not in ['receipts', 'deliveries']:
        return {"error": "Expected direction must be 'receipts' or 'deliveries'"}

    df = get_dataset(remove_duplicates=None)  # Keep all records for business rule checking
    df_filtered = df.copy()
    filters_applied = []

    # Apply additional filters first
    for column, condition in filters.items():
        if column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        elif column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        else:
            if column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column] == condition]
                filters_applied.append(f"{column} == {condition}")

    try:
        # Check if category exists
        category_data = df_filtered[df_filtered['category_short'] == category]
        if len(category_data) == 0:
            available_categories = df_filtered['category_short'].dropna().unique().tolist()
            return {"error": f"Category '{category}' not found. Available categories: {available_categories}"}

        # Generic violation detection based on expected direction
        if expected_direction.lower() == 'receipts':
            # Find records NOT acting as receipts (rec_del_sign != 1)
            violations = category_data[category_data['rec_del_sign'] != 1]
            actual_direction = "deliveries"
            expected_sign = 1
        else:  # expected_direction == 'deliveries'
            # Find records NOT acting as deliveries (rec_del_sign != -1)
            violations = category_data[category_data['rec_del_sign'] != -1]
            actual_direction = "receipts"
            expected_sign = -1

        violation_count = len(violations)
        total_category_records = len(category_data)

        rule_description = f"{category} facilities acting as {actual_direction} (expected: {expected_direction})"
        expected_behavior = f"{category} facilities should act as {expected_direction} (rec_del_sign = {expected_sign})"

        if violation_count > 0:
            # Sample violations for examples
            sample_violations = violations.head(max_examples)
            examples = []

            for _, row in sample_violations.iterrows():
                examples.append(f"{row['pipeline_name']} - {row['loc_name']}: {category} acting as {actual_direction}")

            violation_percentage = (violation_count / total_category_records) * 100

            return {
                "data": f"Found {violation_count:,} violations ({violation_percentage:.1f}% of {category} records). Rule: {rule_description}. Examples: {'; '.join(examples[:3])}",
                "metadata": {
                    "method_used": f"Generic business rule violation detection",
                    "columns_accessed": ["category_short", "rec_del_sign", "pipeline_name", "loc_name"],
                    "total_rows_after_filter": len(df_filtered),
                    "total_rows_before_filter": len(df),
                    "filters_applied": filters_applied,
                    "violation_count": violation_count,
                    "total_category_records": total_category_records,
                    "violation_percentage": round(violation_percentage, 1),
                    "category_checked": category,
                    "expected_direction": expected_direction,
                    "actual_direction": actual_direction,
                    "rule_description": rule_description,
                    "expected_behavior": expected_behavior,
                    "all_examples": examples,
                    "duplicates_removed": False,
                }
            }
        else:
            return {
                "data": f"No violations found. All {total_category_records:,} {category} records act as expected {expected_direction}",
                "metadata": {
                    "method_used": f"Generic business rule violation detection",
                    "rule_description": rule_description,
                    "filters_applied": filters_applied,
                    "violation_count": 0,
                    "total_category_records": total_category_records,
                    "category_checked": category,
                    "expected_direction": expected_direction,
                    "duplicates_removed": False,
                }
            }

    except Exception as e:
        return {"error": f"Business rule violation detection failed: {str(e)}"}

def detect_data_completeness_issues(tool_context: ToolContext, category: str,
                                  required_fields: List[str], completeness_rule: str,
                                  filters: Dict[str, Any], max_examples: int) -> Dict[str, Any]:
    """
    Detect data completeness issues by checking for missing required fields in records.

    Args:
        category: Category to check (e.g., 'Interconnect', 'Production', 'LDC')
        required_fields: List of fields that should be complete (e.g., ['connecting_pipeline', 'connecting_entity'])
        completeness_rule: How to check completeness:
            - 'all_required': ALL listed fields must be present/non-empty
            - 'any_required': At least ONE listed field must be present/non-empty
        filters: Dictionary of additional filters to apply (year, state, etc.)
        max_examples: Maximum number of incomplete record examples to return

    Examples:
        - category='Interconnect', required_fields=['connecting_pipeline', 'connecting_entity'], completeness_rule='all_required'
          → Finds Interconnect records missing BOTH partner fields
        - category='Production', required_fields=['scheduled_quantity'], completeness_rule='all_required'
          → Finds Production records missing quantity data
    """
    # Handle parameter defaults
    if not filters:
        filters = {}
    if not max_examples:
        max_examples = 10
    if not category:
        return {"error": "Category parameter is required"}
    if not required_fields:
        return {"error": "Required fields parameter is required (list of field names)"}
    if not completeness_rule:
        completeness_rule = 'all_required'

    # Validate completeness_rule
    if completeness_rule not in ['all_required', 'any_required']:
        return {"error": "Completeness rule must be 'all_required' or 'any_required'"}

    df = get_dataset(remove_duplicates=None)  # Keep all records for business rule checking
    df_filtered = df.copy()
    filters_applied = []

    # Apply additional filters first
    for column, condition in filters.items():
        if column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        elif column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        else:
            if column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column] == condition]
                filters_applied.append(f"{column} == {condition}")

    try:
        # Filter to specified category
        category_data = df_filtered[df_filtered['category_short'] == category]
        if len(category_data) == 0:
            available_categories = df_filtered['category_short'].dropna().unique().tolist()
            return {"error": f"Category '{category}' not found. Available categories: {available_categories}"}

        # Check if required fields exist in dataset
        missing_columns = [field for field in required_fields if field not in df_filtered.columns]
        if missing_columns:
            available_columns = df_filtered.columns.tolist()
            return {"error": f"Required fields {missing_columns} not found in dataset. Available columns: {available_columns}"}

        # Build completeness condition for each field (missing or empty)
        field_conditions = []
        for field in required_fields:
            field_missing = (category_data[field].isna()) | (category_data[field] == '')
            field_conditions.append(field_missing)

        # Apply completeness rule
        if completeness_rule == 'all_required':
            # Find records where ALL required fields are missing/empty
            incomplete_mask = field_conditions[0]
            for condition in field_conditions[1:]:
                incomplete_mask = incomplete_mask & condition
            rule_description = f"ALL fields {required_fields} are missing/empty"
        else:  # any_required
            # Find records where ALL required fields are missing/empty (none present)
            incomplete_mask = field_conditions[0]
            for condition in field_conditions[1:]:
                incomplete_mask = incomplete_mask & condition
            rule_description = f"At least one of {required_fields} should be present, but ALL are missing/empty"

        incomplete_records = category_data[incomplete_mask]
        incomplete_count = len(incomplete_records)
        total_category_records = len(category_data)

        if incomplete_count > 0:
            # Sample incomplete records for examples
            sample_incomplete = incomplete_records.head(max_examples)
            examples = []

            for _, row in sample_incomplete.iterrows():
                missing_fields = []
                for field in required_fields:
                    if pd.isna(row[field]) or row[field] == '':
                        missing_fields.append(field)
                examples.append(f"{row['pipeline_name']} - {row['loc_name']}: Missing {missing_fields}")

            incompleteness_percentage = (incomplete_count / total_category_records) * 100

            return {
                "data": f"Found {incomplete_count:,} incomplete {category} records ({incompleteness_percentage:.1f}% of {category} records). Rule: {rule_description}. Examples: {'; '.join(examples[:3])}",
                "metadata": {
                    "method_used": f"Data completeness validation",
                    "columns_accessed": ["category_short", "pipeline_name", "loc_name"] + required_fields,
                    "total_rows_after_filter": len(df_filtered),
                    "total_rows_before_filter": len(df),
                    "filters_applied": filters_applied,
                    "incomplete_count": incomplete_count,
                    "total_category_records": total_category_records,
                    "incompleteness_percentage": round(incompleteness_percentage, 1),
                    "category_checked": category,
                    "required_fields": required_fields,
                    "completeness_rule": completeness_rule,
                    "rule_description": rule_description,
                    "all_examples": examples,
                    "duplicates_removed": False,
                }
            }
        else:
            return {
                "data": f"All {total_category_records:,} {category} records have complete data for {required_fields}",
                "metadata": {
                    "method_used": f"Data completeness validation",
                    "rule_description": rule_description,
                    "filters_applied": filters_applied,
                    "incomplete_count": 0,
                    "total_category_records": total_category_records,
                    "category_checked": category,
                    "required_fields": required_fields,
                    "completeness_rule": completeness_rule,
                    "duplicates_removed": False,
                }
            }

    except Exception as e:
        return {"error": f"Data completeness validation failed: {str(e)}"}

def detect_entity_duplicates_across_dimensions(tool_context: ToolContext, entity_fields: List[str],
                                             dimension_fields: List[str], filters: Dict[str, Any],
                                             max_examples: int) -> Dict[str, Any]:
    """
    Detect entity duplicates where the same entity combination appears across multiple dimensions.
    - automatically runs separate analysis for each dimension field.

    Args:
        entity_fields: Fields that define the entity (e.g., ['pipeline_name', 'loc_name'])
        dimension_fields: Fields to check for duplicates across (['state_abb', 'county_name'])
        filters: Dictionary of additional filters to apply (year, state, etc.)
        max_examples: Maximum number of duplicate examples to return

    Examples:
        - entity_fields=['pipeline_name', 'loc_name'], dimension_fields=['state_abb', 'county_name']
          → Finds pipeline-location combinations appearing in multiple states AND counties (separate analysis)
        - entity_fields=['connecting_entity'], dimension_fields=['category_short']
          → Finds entities appearing in multiple categories
        - entity_fields=['employee_id'], dimension_fields=['department', 'location']
          → Finds employees in multiple departments AND locations
    """
    # Handle parameter defaults
    if not filters:
        filters = {}
    if not max_examples:
        max_examples = 10
    if not entity_fields:
        return {"error": "Entity fields parameter is required (list of field names)"}
    if not dimension_fields:
        return {"error": "Dimension fields parameter is required (list of field names)"}

    df = get_dataset(remove_duplicates=None)  # Keep all records for business rule checking
    df_filtered = df.copy()
    filters_applied = []

    # Apply additional filters first
    for column, condition in filters.items():
        if column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        elif column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        else:
            if column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column] == condition]
                filters_applied.append(f"{column} == {condition}")

    try:
        # Check if all fields exist in dataset
        all_fields = entity_fields + dimension_fields
        missing_columns = [field for field in all_fields if field not in df_filtered.columns]
        if missing_columns:
            available_columns = df_filtered.columns.tolist()
            return {"error": f"Fields {missing_columns} not found in dataset. Available columns: {available_columns}"}

        # Run analysis for each dimension field separately
        results_by_dimension = {}
        total_duplicate_count = 0
        all_examples = []

        for dimension_field in dimension_fields:
            # Get unique entity-dimension combinations for this dimension
            unique_combinations = df_filtered[entity_fields + [dimension_field]].drop_duplicates()

            # Group by entity fields and count unique dimension values
            entity_dim_counts = unique_combinations.groupby(entity_fields)[dimension_field].agg(['nunique', list]).reset_index()
            entity_dim_counts.columns = entity_fields + ['unique_values', 'value_list']

            # Find entities that appear in multiple dimension values
            duplicates = entity_dim_counts[entity_dim_counts['unique_values'] > 1]

            duplicate_count = len(duplicates)
            total_entities = len(entity_dim_counts)
            duplicate_percentage = (duplicate_count / total_entities) * 100 if total_entities > 0 else 0

            # Collect examples for this dimension
            dimension_examples = []
            if duplicate_count > 0:
                sample_duplicates = duplicates.head(max_examples)
                for _, row in sample_duplicates.iterrows():
                    entity_desc = ' - '.join([str(row[field]) for field in entity_fields])
                    values = row['value_list']
                    dimension_examples.append(f"{entity_desc}: appears in {dimension_field} {values}")

            results_by_dimension[dimension_field] = {
                "duplicate_count": duplicate_count,
                "total_entities": total_entities,
                "duplicate_percentage": round(duplicate_percentage, 2),
                "examples": dimension_examples
            }

            total_duplicate_count += duplicate_count
            all_examples.extend(dimension_examples[:2])  # Take 2 examples per dimension

        # Format comprehensive results
        summary_parts = []
        for dimension_field, results in results_by_dimension.items():
            if results["duplicate_count"] > 0:
                summary_parts.append(f"{results['duplicate_count']:,} entities appear in multiple {dimension_field} ({results['duplicate_percentage']:.2f}%)")
            else:
                summary_parts.append(f"No duplicates found across {dimension_field}")

        summary = "; ".join(summary_parts)

        return {
            "data": f"Multi-dimensional duplicate analysis: {summary}. Examples: {'; '.join(all_examples[:3])}",
            "metadata": {
                "method_used": f"Multi-dimensional duplicate detection across {len(dimension_fields)} dimensions",
                "columns_accessed": entity_fields + dimension_fields,
                "total_rows_after_filter": len(df_filtered),
                "total_rows_before_filter": len(df),
                "filters_applied": filters_applied,
                "entity_fields": entity_fields,
                "dimension_fields": dimension_fields,
                "results_by_dimension": results_by_dimension,
                "total_duplicate_count": total_duplicate_count,
                "all_examples": all_examples,
                "duplicates_removed": False,
            }
        }

    except Exception as e:
        return {"error": f"Multi-dimensional duplicate detection failed: {str(e)}"}

def detect_anomalies_simple(tool_context: ToolContext, method: str, filters: Dict[str, Any],
                           group_by_column: str, rolling_days: int, std_threshold: float,
                           max_examples: int) -> Dict[str, Any]:
    """
    Detect anomalies using various methods with flexible LLM-orchestrated parameters.

    Args:
        method: 'zscore', 'rolling', 'change', 'percentile'
        filters: Dictionary of filters to apply (year, state, etc.)
        group_by_column: Column to group by ('loc_name', 'connecting_entity', 'pipeline_name', 'total')
        rolling_days: Number of days for window (e.g., 90, 20, 30)
        std_threshold: Threshold for detection (std devs, percentile, change %)
        max_examples: Maximum number of examples to return
    """
    # Handle parameter defaults
    if not filters:
        filters = {}
    if not method:
        method = 'rolling'
    if not group_by_column:
        group_by_column = 'loc_name'
    if not rolling_days:
        rolling_days = 90
    if not std_threshold:
        std_threshold = 3.0
    if not max_examples:
        max_examples = 10

    # Smart default: deduplicate for volume-based anomalies, keep duplicates for count-based
    remove_duplicates = method.lower() in ['zscore', 'rolling', 'percentile']  # Volume-based methods

    df = get_dataset(remove_duplicates=remove_duplicates)
    df_filtered = df.copy()
    filters_applied = []

    # Track deduplication in metadata
    if remove_duplicates:
        filters_applied.append("exact_duplicates_removed")

    # Apply filters
    for column, condition in filters.items():
        if column == 'year':
            df_filtered = df_filtered[df_filtered['eff_gas_day'].dt.year == condition]
            filters_applied.append(f"eff_gas_day year == {condition}")
        elif column == 'state':
            df_filtered = df_filtered[df_filtered['state_abb'] == condition]
            filters_applied.append(f"state_abb == {condition}")
        elif column == 'pipeline':
            df_filtered = df_filtered[df_filtered['pipeline_name'] == condition]
            filters_applied.append(f"pipeline_name == {condition}")
        else:
            if column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column] == condition]
                filters_applied.append(f"{column} == {condition}")

    if len(df_filtered) == 0:
        return {"error": "No data matches the specified filters"}

    try:
        if method == "zscore":
            # Simple z-score method
            clean_data = df_filtered['scheduled_quantity'].dropna()
            z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
            anomalies = df_filtered.loc[clean_data[z_scores > std_threshold].index]

            return {
                "data": f"Found {len(anomalies)} anomalies using Z-score method (threshold > {std_threshold})",
                "metadata": {
                    "method_used": f"Z-score anomaly detection with threshold > {std_threshold}",
                    "columns_accessed": ["scheduled_quantity"],
                    "total_rows_after_filter": len(clean_data),
                    "total_rows_before_filter": len(df),
                    "filters_applied": filters_applied,
                    "anomaly_count": len(anomalies),
                    "duplicates_removed": remove_duplicates
                }
            }

        elif method == "rolling":
            # Rolling window anomaly detection (Z-score based)
            return _rolling_anomaly_detection(df_filtered, group_by_column, rolling_days, std_threshold, max_examples, filters_applied, "z-score", remove_duplicates)

        elif method == "change":
            # Significant change detection from recent period
            return _change_anomaly_detection(df_filtered, group_by_column, rolling_days, std_threshold, max_examples, filters_applied, remove_duplicates)

        elif method == "percentile":
            # Percentile-based anomaly detection
            return _percentile_anomaly_detection(df_filtered, group_by_column, rolling_days, std_threshold, max_examples, filters_applied, remove_duplicates)
        else:
            return {"error": f"Method {method} not supported. Use 'zscore', 'rolling', 'change', or 'percentile'."}

    except Exception as e:
        return {"error": f"Anomaly detection failed: {str(e)}"}

def _rolling_anomaly_detection(df_filtered, group_by_column, rolling_days, std_threshold, max_examples, filters_applied, method_type, remove_duplicates):
    """Rolling window anomaly detection helper."""
    if group_by_column != 'total' and group_by_column not in df_filtered.columns:
        return {"error": f"Column {group_by_column} not found in dataset"}

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_filtered['eff_gas_day']):
        df_filtered['eff_gas_day'] = pd.to_datetime(df_filtered['eff_gas_day'])

    # Create daily aggregates
    df_temp = df_filtered.copy()
    df_temp["date"] = df_temp["eff_gas_day"].dt.date

    if group_by_column == 'total':
        # Total system-wide daily volumes
        daily = df_temp.groupby("date")["scheduled_quantity"].sum().reset_index()
        daily['entity'] = 'Total System'
        entity_col = 'entity'
    else:
        daily = df_temp.groupby([group_by_column, "date"])["scheduled_quantity"].sum().reset_index()
        entity_col = group_by_column

    if len(daily) == 0:
        return {"data": "No data available for rolling anomaly analysis"}

    # Calculate rolling statistics
    min_periods = max(5, rolling_days // 4)
    if group_by_column == 'total':
        daily["roll_mean"] = daily["scheduled_quantity"].rolling(rolling_days, min_periods=min_periods).mean()
        daily["roll_std"] = daily["scheduled_quantity"].rolling(rolling_days, min_periods=min_periods).std()
    else:
        daily["roll_mean"] = daily.groupby(entity_col)["scheduled_quantity"].transform(
            lambda s: s.rolling(rolling_days, min_periods=min_periods).mean()
        )
        daily["roll_std"] = daily.groupby(entity_col)["scheduled_quantity"].transform(
            lambda s: s.rolling(rolling_days, min_periods=min_periods).std()
        )

    daily["zscore"] = (daily["scheduled_quantity"] - daily["roll_mean"]) / daily["roll_std"]

    # Find anomalies
    anomalies = daily[
        (daily["zscore"] > std_threshold) & daily["roll_std"].notna()
    ].sort_values("zscore", ascending=False).head(max_examples)

    return _format_anomaly_results(anomalies, entity_col, rolling_days, std_threshold, filters_applied, f"{rolling_days}-day rolling window z-score", method_type, remove_duplicates)

def _change_anomaly_detection(df_filtered, group_by_column, rolling_days, change_threshold, max_examples, filters_applied, remove_duplicates):
    """Significant change detection from recent period."""
    if group_by_column != 'total' and group_by_column not in df_filtered.columns:
        return {"error": f"Column {group_by_column} not found in dataset"}

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_filtered['eff_gas_day']):
        df_filtered['eff_gas_day'] = pd.to_datetime(df_filtered['eff_gas_day'])

    # Create daily aggregates
    df_temp = df_filtered.copy()
    df_temp["date"] = df_temp["eff_gas_day"].dt.date

    if group_by_column == 'total':
        daily = df_temp.groupby("date")["scheduled_quantity"].sum().reset_index()
        daily['entity'] = 'Total System'
        entity_col = 'entity'
    else:
        daily = df_temp.groupby([group_by_column, "date"])["scheduled_quantity"].sum().reset_index()
        entity_col = group_by_column

    if len(daily) == 0:
        return {"data": "No data available for change analysis"}

    # Calculate percentage change from recent period average
    min_periods = max(5, rolling_days // 4)
    if group_by_column == 'total':
        daily["recent_avg"] = daily["scheduled_quantity"].rolling(rolling_days, min_periods=min_periods).mean()
    else:
        daily["recent_avg"] = daily.groupby(entity_col)["scheduled_quantity"].transform(
            lambda s: s.rolling(rolling_days, min_periods=min_periods).mean()
        )

    daily["pct_change"] = ((daily["scheduled_quantity"] - daily["recent_avg"]) / daily["recent_avg"]) * 100

    # Find significant changes (using threshold as percentage)
    anomalies = daily[
        (abs(daily["pct_change"]) > change_threshold) & daily["recent_avg"].notna()
    ].sort_values("pct_change", key=abs, ascending=False).head(max_examples)

    return _format_change_results(anomalies, entity_col, rolling_days, change_threshold, filters_applied, remove_duplicates)

def _percentile_anomaly_detection(df_filtered, group_by_column, rolling_days, percentile_threshold, max_examples, filters_applied, remove_duplicates):
    """Percentile-based anomaly detection."""
    if group_by_column != 'total' and group_by_column not in df_filtered.columns:
        return {"error": f"Column {group_by_column} not found in dataset"}

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_filtered['eff_gas_day']):
        df_filtered['eff_gas_day'] = pd.to_datetime(df_filtered['eff_gas_day'])

    # Create daily aggregates
    df_temp = df_filtered.copy()
    df_temp["date"] = df_temp["eff_gas_day"].dt.date

    if group_by_column == 'total':
        daily = df_temp.groupby("date")["scheduled_quantity"].sum().reset_index()
        daily['entity'] = 'Total System'
        entity_col = 'entity'
    else:
        daily = df_temp.groupby([group_by_column, "date"])["scheduled_quantity"].sum().reset_index()
        entity_col = group_by_column

    if len(daily) == 0:
        return {"data": "No data available for percentile analysis"}

    # Calculate rolling percentiles
    min_periods = max(5, rolling_days // 4)
    if group_by_column == 'total':
        daily["roll_percentile"] = daily["scheduled_quantity"].rolling(rolling_days, min_periods=min_periods).quantile(percentile_threshold/100)
    else:
        daily["roll_percentile"] = daily.groupby(entity_col)["scheduled_quantity"].transform(
            lambda s: s.rolling(rolling_days, min_periods=min_periods).quantile(percentile_threshold/100)
        )

    # Find values above percentile threshold
    anomalies = daily[
        (daily["scheduled_quantity"] > daily["roll_percentile"]) & daily["roll_percentile"].notna()
    ].sort_values("scheduled_quantity", ascending=False).head(max_examples)

    return _format_anomaly_results(anomalies, entity_col, rolling_days, percentile_threshold, filters_applied, f"{rolling_days}-day rolling {percentile_threshold}th percentile", "percentile", remove_duplicates)

def _format_anomaly_results(anomalies, entity_col, rolling_days, threshold, filters_applied, method_desc, method_type, remove_duplicates):
    """Format anomaly results consistently."""
    if len(anomalies) > 0:
        examples = []
        for _, row in anomalies.iterrows():
            if method_type == "z-score":
                examples.append(f"{row[entity_col]} on {row['date']}: {row['scheduled_quantity']:,.0f} ({row['zscore']:.1f}σ above)")
            else:
                examples.append(f"{row[entity_col]} on {row['date']}: {row['scheduled_quantity']:,.0f} (>{threshold}th percentile)")

        entity_type = entity_col.replace('_', ' ') if entity_col != 'entity' else 'system'

        return {
            "data": f"Found {len(anomalies)} anomalous days across {anomalies[entity_col].nunique()} {entity_type}s. Examples: " + "; ".join(examples[:3]),
            "metadata": {
                "method_used": method_desc,
                "columns_accessed": [entity_col, "eff_gas_day", "scheduled_quantity"],
                "entities_processed": anomalies[entity_col].nunique(),
                "threshold": f"{threshold} {'standard deviations' if method_type == 'z-score' else 'percentile'}",
                "filters_applied": filters_applied,
                "anomaly_count": len(anomalies),
                "affected_entities": anomalies[entity_col].nunique(),
                "rolling_window_days": rolling_days,
                "all_examples": examples,
                "duplicates_removed": remove_duplicates
            }
        }
    else:
        entity_type = entity_col.replace('_', ' ') if entity_col != 'entity' else 'system'
        return {
            "data": f"No {entity_type} anomalies detected using {method_desc}",
            "metadata": {
                "method_used": method_desc,
                "threshold": f"{threshold} {'standard deviations' if method_type == 'z-score' else 'percentile'}",
                "filters_applied": filters_applied,
                "rolling_window_days": rolling_days,
                "duplicates_removed": remove_duplicates
            }
        }

def _format_change_results(anomalies, entity_col, rolling_days, change_threshold, filters_applied, remove_duplicates):
    """Format change detection results."""
    if len(anomalies) > 0:
        examples = []
        for _, row in anomalies.iterrows():
            change_type = "increase" if row['pct_change'] > 0 else "decrease"
            examples.append(f"{row[entity_col]} on {row['date']}: {row['scheduled_quantity']:,.0f} ({abs(row['pct_change']):.1f}% {change_type})")

        entity_type = entity_col.replace('_', ' ') if entity_col != 'entity' else 'system'

        return {
            "data": f"Found {len(anomalies)} days with significant changes across {anomalies[entity_col].nunique()} {entity_type}s. Examples: " + "; ".join(examples[:3]),
            "metadata": {
                "method_used": f"{rolling_days}-day change detection",
                "columns_accessed": [entity_col, "eff_gas_day", "scheduled_quantity"],
                "entities_processed": anomalies[entity_col].nunique(),
                "threshold": f"{change_threshold}% change",
                "filters_applied": filters_applied,
                "anomaly_count": len(anomalies),
                "affected_entities": anomalies[entity_col].nunique(),
                "rolling_window_days": rolling_days,
                "all_examples": examples,
                "duplicates_removed": remove_duplicates
            }
        }
    else:
        entity_type = entity_col.replace('_', ' ') if entity_col != 'entity' else 'system'
        return {
            "data": f"No significant {change_threshold}% changes detected in {entity_type} volumes",
            "metadata": {
                "method_used": f"{rolling_days}-day change detection",
                "threshold": f"{change_threshold}% change",
                "filters_applied": filters_applied,
                "rolling_window_days": rolling_days,
                "duplicates_removed": remove_duplicates
            }
        }
