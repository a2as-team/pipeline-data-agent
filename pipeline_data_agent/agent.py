"""
Main agent definition for Pipeline data analysis agent.
This file defines the root agent that orchestrates data analysis tasks.
"""

import os
import pandas as pd
from datetime import datetime
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm

from .retrieval_tools import (
    get_dataset_info,
    query_total_volume,
    count_unique_pipelines,
    find_top_state_by_activity,
    get_top_pipelines_by_volume,
    calculate_average_quantity,
    analyze_receipts_vs_deliveries,
    calculate_net_flows,
    get_top_locations_by_metric,
    aggregate_by_multiple_dimensions,
    filter_and_rank,
    analyze_time_series,
    compare_across_years
)
from .pattern_tools import (
    analyze_seasonal_patterns,
    analyze_monthly_trends,
    find_consistent_pipelines,
    analyze_receipts_deliveries_patterns,
    analyze_activity_over_time,
    calculate_correlation,
    compare_before_after,
    perform_clustering
)
from .anomaly_tools import (
    detect_anomalies_simple,
    detect_business_rule_violations,
    detect_data_completeness_issues,
    detect_entity_duplicates_across_dimensions
)
from .prompts import AGENT_INSTRUCTION
from .utils.data_access import set_dataset
from .utils.data_downloader import ensure_dataset_available

date_today = datetime.now().strftime("%Y-%m-%d")

# Validate and load dataset at module initialization (startup time)
print("🔍 Pipeline Agent - Initializing dataset...")
try:
    data_path = ensure_dataset_available()
    dataset = pd.read_parquet(data_path)
    # Convert date columns
    if 'eff_gas_day' in dataset.columns:
        dataset['eff_gas_day'] = pd.to_datetime(dataset['eff_gas_day'])

    # Set dataset globally for tools to access
    set_dataset(dataset)
    print(f"✅ Dataset loaded: {len(dataset):,} rows, {len(dataset.columns)} columns")

    # Store dataset metadata for callbacks
    _dataset_metadata = {
        "dataset_loaded": True,
        "dataset_path": data_path,
        "dataset_rows": len(dataset),
        "dataset_columns": len(dataset.columns),
        "today": date_today
    }

except Exception as e:
    print(f"❌ Failed to initialize dataset: {e}")
    print("🛑 Agent cannot start without dataset. Please follow the manual download instructions in README.md")
    raise SystemExit(1)

def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Setup function called before agent execution."""
    context = callback_context

    # Dataset is already loaded at module init, just set the metadata
    context.state.update(_dataset_metadata)
    print(f"🔄 Agent ready - Dataset: {_dataset_metadata['dataset_rows']:,} rows")

# Configure LLM model via LiteLLM (supports both Anthropic and OpenAI)
# Default to Claude, but allow OpenAI via environment variable
default_model = "anthropic/claude-3-5-haiku-20241022"
model_name = os.getenv("LLM_MODEL", default_model)

# Model validation
# tested for claude-3-5-haiku
# other models should work but needs comprehensive testing
supported_models = {
    "anthropic/claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    # "anthropic/claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    # "gpt-4o": "GPT-4 Optimized",
    # "gpt-4o-mini": "GPT-4 Optimized Mini",
    # "gpt-3.5-turbo": "GPT-3.5 Turbo"
}

if model_name not in supported_models:
    print(f"Warning: Model '{model_name}' not in tested models. Supported: {list(supported_models.keys())}")

llm_model = LiteLlm(model=model_name)

# Main agent definition
root_agent = Agent(
    model=llm_model,
    name="pipeline_data_agent",
    tools=[
        get_dataset_info,
        query_total_volume,
        count_unique_pipelines,
        find_top_state_by_activity,
        get_top_pipelines_by_volume,
        calculate_average_quantity,
        analyze_receipts_vs_deliveries,
        detect_anomalies_simple,
        detect_business_rule_violations,
        detect_data_completeness_issues,
        detect_entity_duplicates_across_dimensions,
        analyze_seasonal_patterns,
        analyze_monthly_trends,
        find_consistent_pipelines,
        analyze_receipts_deliveries_patterns,
        analyze_activity_over_time,
        calculate_net_flows,
        get_top_locations_by_metric,
        aggregate_by_multiple_dimensions,
        filter_and_rank,
        analyze_time_series,
        compare_across_years,
        calculate_correlation,
        compare_before_after,
        perform_clustering
    ],
    instruction=AGENT_INSTRUCTION.format(today=date_today),
    global_instruction=f"You are a pipeline data analysis expert. Today's date: {date_today}",
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.1)
)
