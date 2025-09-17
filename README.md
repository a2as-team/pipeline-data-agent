# Data Agent

A sophisticated data analysis agent built with Google ADK and Claude for comprehensive pipeline data analysis. The agent handles queries from simple data retrieval to advanced pattern recognition and anomaly detection using modular tools.

## Features

- **Comprehensive Analysis**: Analysis tools covering retrieval, patterns, and anomaly detection
- **Auto-Download**: Automatically downloads pipeline dataset from Google Drive if not present
- **LLM Orchestration**: Uses Claude 3.5 Haiku to intelligently combine tools for complex queries
- **Large Dataset Support**: Optimized for 23M+ record datasets with performance safeguards

## Installation

### Prerequisites
- Python 3.10+
- uv (Python package manager)
- Anthropic API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pipeline-data-agent
   ```

2. Install UV (if not already installed)
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or via pip
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Activate Env**
   ```bash
   source .venv/bin/activate
   ```

5. **Set your API key**
   ```bash
   # For Claude (default)
   export ANTHROPIC_API_KEY=your-anthropic-api-key-here
   ```

6. **Run the agent**
   ```bash
   uv run adk run pipeline_data_agent
   ```

## Dataset Setup

The agent requires a large pipeline dataset (23.8M+ records). Due to Google Drive restrictions on large file downloads, **manual download is recommended**.

### Manual Dataset Download (Recommended)

**Step-by-Step Instructions:**

1. **Visit the Google Drive link:**
   ```
   https://drive.google.com/file/d/109vhmnSLN3oofjFdyb58l_rUZRa0d6C8/view
   ```

2. **Download the file:**
   - Click the "Download" button in Google Drive
   - The file will download as `pipeline_data.parquet` (159MB)

3. **Place the file in the correct location:**
   ```bash
   # Create data directory if it doesn't exist
   mkdir -p data

   # Move downloaded file to correct location
   mv ~/Downloads/pipeline_data.parquet data/pipeline_data.parquet
   ```

4. **Verify the file is in place:**
   ```bash
   ls -la data/pipeline_data.parquet
   # Should show: -rw-r--r-- ... 159013112 ... pipeline_data.parquet
   ```

### Alternative: Automatic Download
The agent will attempt automatic download on first run, but this often fails for large files. If you see download errors, use the manual method above.

## Dataset Information

- **Size**: 159MB, 23.8M+ records
- **Schema**: 13 columns including pipeline_name, scheduled_quantity, eff_gas_day, rec_del_sign, category_short, state_abb
- **Categories**: 'LDC' (utilities), 'Industrial', 'Production', 'Storage', etc.
- **Transaction Types**: rec_del_sign (1 = receipts/inflow, -1 = deliveries/outflow)
- **Time Range**: Multi-year pipeline transaction data

## Example Queries & Outputs

### 1. Basic Dataset Information
**🔍 Query:** "What's the basic information about this dataset?"

**🤖 Agent Output:**
```
Dataset Overview:
- **Total Rows:** 23,854,855
- **Total Columns:** 13

**Columns and Their Types:**
1. `pipeline_name` (object): Pipeline identification
2. `loc_name` (object): Location name
3. `connecting_pipeline` (object): Connected pipeline
4. `connecting_entity` (object): Business entity
5. `rec_del_sign` (int64): Transaction direction (1 = receipts, -1 = deliveries)
6. `category_short` (object): Short category classification
7. `country_name` (object): Country
8. `state_abb` (object): State abbreviation
9. `county_name` (object): County name
10. `latitude` (object): Geographical latitude
11. `longitude` (object): Geographical longitude
12. `eff_gas_day` (datetime): Effective gas day
13. `scheduled_quantity` (float64): Gas volume scheduled

**Missing Data Highlights:**
- `connecting_pipeline`: 78.23% missing (18,662,066 rows)
- `connecting_entity`: 12.47% missing (2,975,770 rows)
- `latitude` and `longitude`: 100% missing (23,854,855 rows)
- `county_name`: 10.38% missing (2,477,210 rows)
- Most other columns have minimal missing data

**Duplicate Records:**
- Total duplicates: 404,743 records

**Key Insights:**
1. Large dataset with over 23 million transactions
2. Comprehensive pipeline and location information
3. Significant missing data in connecting pipeline and geolocation fields
4. Tracks gas flow direction and scheduled quantities

**Data Quality Considerations:**
- High percentage of missing connecting pipeline information
- No geographical coordinates available
- Relatively clean for core transaction fields (rec_del_sign, eff_gas_day, scheduled_quantity)
```

### 2. Retrieval
**🔍 Query:** "What are the net flows by pipeline in 2024, ranked from largest source to largest sink?"

**🤖 Agent Output:**
```
Net Flows by Pipeline in 2024 (Ranked from Largest Source to Largest Sink)

**Top Sources (Positive Net Flows):**
1. Pacific Gas and Electric Company: +1,373,606,500
2. Dominion Cove Point LNG, LP: +292,307,514
3. Gulf South Pipeline Company, LP: +188,600,569
4. Enable Gas Transmission: +163,316,566
5. ANR Pipeline Company: +159,975,165

**Top Sinks (Negative Net Flows):**
1. Transcontinental Gas Pipe Line Company, LLC: Largest sink
2. Northwest Pipeline GP: Major sink
3. Northern Natural Gas Company: Significant sink
4. Florida Gas Transmission Company, LLC: Large sink
5. Texas Eastern Transmission, LP: Substantial sink

**Supporting Evidence:**
- Method: Net flow calculation (receipts - deliveries)
- Columns Used: `pipeline_name`, `scheduled_quantity`, `rec_del_sign`
- Sample Size: 6,433,211 rows
- Year: 2024
- Filters: Transactions in 2024

**Key Insights:**
1. Source-Sink Dynamics:
   - Pacific Gas and Electric leads as the largest net source
   - Transcontinental Gas Pipe Line is the largest net sink
   - Significant variability in pipeline flow patterns

2. Regional Observations:
   - West Coast (Pacific Gas and Electric): Major source
   - East Coast (Transcontinental, Florida Gas): Major sinks
   - Gulf Coast (Gulf South): Moderate source

**Visualization of Net Flows:**
```
Sources (+)                   Sinks (-)
Pacific Gas & Electric    →   Transcontinental Gas Pipe Line
Dominion Cove Point LNG  →   Northwest Pipeline GP
Gulf South Pipeline      →   Northern Natural Gas
Enable Gas Transmission  →   Florida Gas Transmission
ANR Pipeline Company     →   Texas Eastern Transmission
```

**Potential Explanations:**
1. Sources may represent:
   - Production regions
   - LNG export points
   - Storage injection points

2. Sinks may represent:
   - Major consumption markets
   - Distribution to local utilities
   - Storage withdrawal points

**Limitations:**
- Snapshot of 2024 data
- Aggregated annual view
- Does not capture intra-year variations
- Influenced by seasonal demand, infrastructure, and market conditions

**Recommendations:**
1. Investigate drivers of significant source/sink imbalances
2. Analyze seasonal variations in pipeline flows
3. Assess infrastructure constraints
4. Examine market dynamics influencing net flow patterns
```

### 3. Pattern Recognition
**🔍 Query:** "Compare winter vs summer mean deliveries for utility category by state"

**🤖 Agent Output:**
```
To interpret this data, I'll map the quarters to seasons:
- Q1 (Winter): December-February
- Q2 (Spring): March-May
- Q3 (Summer): June-August
- Q4 (Fall): September-November

Let me break down the winter (Q1) and summer (Q3) mean deliveries for utility categories by state:

Winter (Q1) Mean Deliveries:
1. California (CA): 180,585.72
2. DC: 155,868.28
3. North Carolina (NC): 100,106.39
4. Maryland (MD): 99,084.70
5. New Jersey (NJ): 97,092.00

Summer (Q3) Mean Deliveries:
1. California (CA): 166,466.97
2. North Carolina (NC): 96,784.13
3. Maryland (MD): 66,332.03
4. Washington (WA): 68,299.27

Key Observations:
1. California shows the highest mean deliveries in both winter and summer.
2. Most states show a decrease in mean deliveries from winter to summer.
3. The reduction varies by state:
   - California: ~7.8% decrease
   - North Carolina: ~3.3% decrease
   - Maryland: ~33% decrease

To validate these findings and get more statistical rigor, I'll use the `calculate_correlation` function to see how these metrics relate:
**Answer:** Winter vs Summer Mean Deliveries for Utility Category by State

**Supporting Evidence:**
- Method: Quarterly aggregation by state
- Columns Used: `state_abb`, `scheduled_quantity`
- Sample Size: 6,433,211 rows
- Filters: Utility category, year 2024

**Key Insights:**
1. California consistently leads in utility delivery volumes
2. Most states show a decline in mean deliveries from winter to summer
3. Delivery reductions range from 3.3% (NC) to 33% (MD)

**Possible Explanations:**
- Winter heating demands increase utility gas deliveries
- Summer reduced heating needs lead to lower gas consumption
- Regional variations in energy mix and climate impact delivery patterns

**Limitations:**
- Data limited to 2024
- Only a few states represented in the dataset
- Quarterly aggregation might mask monthly variations

**Recommendations:**
- Conduct more granular monthly analysis
- Investigate state-specific factors driving delivery variations
- Consider expanding data collection to more states
```

### 4. Anomaly Detection
**🔍 Query:** "Check for direction–category mismatches for production facilities acting as deliveries (should be receipts)"

**🤖 Agent Output:**
```
Significant Direction-Category Mismatches for Production Facilities

**Supporting Evidence:**
- Total Violations: 840,410 records (14.1% of Production records)
- Violation Type: Production facilities showing as deliveries (rec_del_sign = -1)
- Data Source: Comprehensive pipeline transaction dataset

**Key Findings:**

1. **Violation Characteristics:**
   - 14.1% of Production records incorrectly marked as deliveries
   - Expected: Receipts (injecting gas into pipeline)
   - Actual: Deliveries (removing gas from pipeline)

2. **Top Pipelines with Violations:**
   - El Paso Natural Gas Company: 137,383 violation instances
   - Williston Basin Interstate Pipeline Company: 62,146 instances
   - Transcontinental Gas Pipe Line Company, LLC: 61,640 instances

3. **Data Completeness Issues:**
   - 110,668 incomplete Production records (13.2% of violating records)
   - Missing critical fields:
     * connecting_pipeline
     * connecting_entity

4. **Notable Example Locations:**
   - ANR Pipeline Company locations:
     * ALLEN PLANT
     * BEULAH INTERCONNECT
     * DUNN INTERCONNECT
     * HANOVER INT (DEL)
     * LOWRY PLANT INLET

**Business Insights:**
1. **Potential Root Causes:**
   - Data entry errors
   - Misclassification of transaction directions
   - Complex interconnect or exchange agreements
   - Systemic issues in transaction recording

2. **Implications:**
   - Compromises data integrity
   - Potential financial reconciliation challenges
   - Misrepresentation of production flow dynamics

**Recommendations:**
1. **Data Validation:**
   - Implement strict validation rules for category-direction mapping
   - Create automated checks to flag inconsistent transactions
   - Develop training programs for data entry personnel

2. **Technical Remediation:**
   - Audit and correct existing misclassified records
   - Enhance data collection processes
   - Implement real-time validation mechanisms

3. **Operational Review:**
   - Investigate specific pipelines with high violation rates
   - Review interconnect and exchange agreements
   - Conduct detailed review of ANR Pipeline Company locations

**Limitations:**
- Analysis based on available dataset
- May not capture all nuanced transaction scenarios
- Requires manual verification of specific cases

**Visualization Suggestion:**
- Pie chart showing:
  * Correct Production Receipts
  * Incorrect Production Deliveries
  * Incomplete Production Records
```

### 5. Causal-style hypotheses (with caveats)
**🔍 Query:** "For the pipeline Kinder Morgan Louisiana Pipeline LLC, do deliveries concentrate in a small set of counties? Compute and interpret a concentration index."

**🤖 Agent Output:**
```
To compute a concentration index, I'll use the Herfindahl-Hirschman Index (HHI) approach. The HHI is calculated by squaring the market share of each county and then summing these squared market shares.

Calculation:
1. Total deliveries: 1,875,140,162
2. Market shares:
   - Cameron Parish: 1,813,302,799 / 1,875,140,162 = 0.9664 (96.64%)
   - Calcasieu Parish: 50,371,267 / 1,875,140,162 = 0.0269 (2.69%)
   - Acadia Parish: 10,585,690 / 1,875,140,162 = 0.0056 (0.56%)
   - Evangeline Parish: 880,406 / 1,875,140,162 = 0.0005 (0.05%)
   - Jefferson Davis Parish: 0 / 1,875,140,162 = 0 (0%)

3. HHI Calculation:
   (0.9664)² + (0.0269)² + (0.0056)² + (0.0005)² + (0)² = 0.9342 + 0.0007 + 0.00003 + 0.000000 + 0 = 0.9350

Let's validate this with a clustering analysis to see if we can identify geographic concentration:
**Answer:** Delivery Concentration Analysis for Kinder Morgan Louisiana Pipeline LLC

**Concentration Metrics:**
1. Herfindahl-Hirschman Index (HHI): 0.9350 (Extremely High Concentration)
   - HHI Scale:
     * 0-0.1: Low Concentration
     * 0.1-0.18: Moderate Concentration
     * >0.18: High Concentration

**Delivery Distribution:**
1. Cameron Parish: 96.64% of total deliveries
2. Calcasieu Parish: 2.69% of total deliveries
3. Acadia Parish: 0.56% of total deliveries
4. Evangeline Parish: 0.05% of total deliveries
5. Jefferson Davis Parish: 0% of deliveries

**Clustering Insights:**
- 3 Distinct Clusters Identified:
  1. Cameron Parish (Cluster 1): Dominant cluster
     - Highest volume (1,813,302,799)
     - Winter peak delivery pattern
  2. Calcasieu Parish (Cluster 0): Secondary cluster
     - Moderate volume (50,371,267)
     - Summer peak delivery pattern
  3. Acadia & Evangeline Parishes (Cluster 2): Minor cluster
     - Low volume (5,733,048)
     - High variability in delivery patterns

**Supporting Evidence:**
- Method: HHI Concentration Index
- Columns Used: `county_name`, `scheduled_quantity`
- Sample Size: 19,943 rows
- Filters: Kinder Morgan Louisiana Pipeline LLC, Deliveries

**Business Insights:**
1. Extreme Geographic Concentration: 96.64% of deliveries in Cameron Parish
2. Potential Reasons:
   - Industrial infrastructure
   - Specific energy demand
   - Geographical constraints
   - Pipeline network configuration

**Limitations:**
- Data from single year
- Limited county-level granularity
- Potential seasonal variations not fully captured

**Recommendations:**
1. Investigate why Cameron Parish dominates deliveries
2. Assess infrastructure and demand in other parishes
3. Evaluate potential for network diversification
4. Conduct deeper analysis of delivery drivers

**Interpretation of HHI (0.9350):**
- Indicates an extremely concentrated delivery network
- Suggests high dependency on a single county (Cameron Parish)
- Potential business risk due to lack of geographic diversification
```

## Other queries to try
### Retrieval Queries
- **🔍 Query: What are the net flows by pipeline in 2024, ranked from largest source to largest sink?**
- **🔍 Query: Which  10 locations had the highest average daily volume in 2023 (separately for receipts and deliveries)?**
- **🔍 Query: Show monthly total volumes in 2024 split by category and state.**
- **🔍 Query: For Pennsylvania in 2024, which five counterparties received the most deliveries?**
- **🔍 Query: List the top five peak days for daily net flow for Kinder Morgan Louisiana Pipeline LLC in 2024**
- **🔍 Query: Which counties in New York saw the largest deliveries in 2024**

### Pattern & Trends Queries
- **🔍 Query: Compare winter vs summer mean deliveries for utility category by state in 2023**
- **🔍 Query: Are days with more active locations associated with higher total volumes? Quantify the relationship**
- **🔍 Query: Compare mean daily volume before/after July 1, 2023 for ANR Pipeline Company for receipt**
- **🔍 Query: Bucket states into high/medium/low delivery consumers using quantiles on total deliveries**
- **🔍 Query: Clusters over normalized 12-month profiles per location for 2023**

### Anomalies & Rule Check Queries
- **🔍 Query: Flag days where a location’s volume is more than three standard deviations above its 90-day rolling average. Provide 10 examples**
- **🔍 Query: Check for direction–category mismatches for production acting like a delivery. List number of violations**
- **🔍 Query: For interconnect rows, find cases missing a partner name or partner pipeline. Quantify the gap.**
- **🔍 Query: Report the share of rows missing geographic coordinates by state.**
- **🔍 Query: Detect potential duplicates where the same pipeline–location label appears in multiple states or counties.**

### Casual-style hypotheses (with caveats)
- **🔍 Query: Test whether utility deliveries rise in winter vs summer by state and report significance.**
- **🔍 Query: After a new interconnect to a named partner began operating in a state, did net receipts for that state increase?**
- **🔍 Query: During large spikes at an interconnect, do nearby pipelines show simultaneous drops? Provide three case studies**
- **🔍 Query: For the pipeline Kinder Morgan Louisiana Pipeline LLC, do deliveries concentrate in a small set of counties? Compute and interpret a concentration index**


## Architecture

### Design Principles
- **Google ADK Patterns**: High-level, flexible tools orchestrated by LLM
- **LLM Orchestration**: Via Claude for complex analysis
- **Modular Design**: Composable and reusable tools across query types


## Assumptions & Limitations

### Data Assumptions
- **Dataset Format**: Expects pipeline data in Parquet format with specific schema
- **Date Columns**: `eff_gas_day` column contains valid datetime data
- **Numeric Columns**: `scheduled_quantity` contains valid numeric measurements
- **Transaction Signs**: `rec_del_sign` uses 1 for receipts/inflow, -1 for deliveries/outflow
- **Categories**: Uses standard pipeline categories ('LDC', 'Industrial', 'Production', 'Storage')
- **Missing Data**: Properly handles NaN/None values in optional columns

### Analysis Limitations
- **Statistical Validity**: Results depend on sufficient sample sizes for statistical tests
- **Data Quality**: Analysis accuracy depends on input data quality and completeness
- **Domain Context**: Agent provides technical analysis but lacks pipeline industry expertise
- **Causal Claims**: Tools identify correlations and patterns, not definitive causal relationships
- **Time Series**: Assumes regular time intervals for temporal analysis functions

### Technical Limitations
- **Model Support**: Supports Anthropic Claude models via LiteLLM (OpenAI model should work but not tested extensively)
- **Dataset Size**: Performance optimized for 10M+ records but may vary with complex queries
- **Memory Usage**: Large datasets require sufficient system memory for processing
- **Internet Required**: Initial dataset download requires internet connection
- **API Dependencies**: Requires valid API keys for chosen LLM provider
- **ToDos**: Future work would involve breaking this into multi-agent with Coordinator/Dispatcher Pattern, artifacts, data sharing between tools, eval etc.

### Performance Considerations
- **Large Datasets**: Processing time increases with dataset size and query complexity
- **Error Handling**: Graceful degradation for malformed queries or data issues

### Model-Specific Notes
- **Claude Models**: Tested extensively, optimized response patterns
- **OpenAI Models**: Functional but may have different response styles
- **Temperature Setting**: Uses low temperature (0.1) for analytical consistency

## License

Copyright (c) 2025 Biplov Bhandari. All rights reserved.

This software is provided for the purpose agreed upon with the author only. No license is granted for any other use, reproduction, distribution, or modification without express written permission from the copyright holder.

For licensing inquiries, please contact the author.
