import streamlit as st
import requests
import pandas as pd
import json
import uuid
from io import StringIO
from requests.auth import HTTPBasicAuth
import traceback
import plotly.express as px

# --- Constants for Charting Logic ---
# These constants define thresholds used in the charting logic to make decisions
# about chart types or to issue warnings (e.g., for performance with large datasets).
MAX_CATEGORIES_FOR_BAR = 20
MAX_GROUPS_FOR_STACKED_OR_PAIRED_BAR = 10
MAX_LINES_FOR_MULTI_LINE = 15
MAX_POINTS_FOR_SCATTER_WARN = 5000

# --- Utility Functions ---
def parse_n8n_data_to_dataframe(data_rows_str: str, data_schema: dict) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Parses data rows (as a JSON string) and a data schema (dictionary)
    into a Pandas DataFrame. This is used for processing data
    received from an N8N webhook.

    It attempts to infer and convert column types based on the provided schema,
    paying special attention to datetime columns for time series analysis.

    Args:
        data_rows_str: A string containing JSON-formatted data rows.
        data_schema: A dictionary describing the schema of the data, including field names and types.

    Returns:
        A tuple containing:
            - A Pandas DataFrame if parsing is successful, otherwise None.
            - A list of column names identified as datetime columns.
    """
    datetime_column_names = []
    try:
        # Validate the basic structure of the data schema.
        if not isinstance(data_schema, dict) or 'fields' not in data_schema or not isinstance(data_schema['fields'], list):
            st.error("Invalid 'dataSchema' format.")
            return None, datetime_column_names
        try:
            # Extract column names and types from the schema.
            columns = [field['name'] for field in data_schema['fields']]
            types = {field['name']: field.get('type', 'STRING') for field in data_schema['fields']}
        except (KeyError, TypeError) as e:
            st.error(f"Error in 'dataSchema' fields structure: {e}")
            return None, datetime_column_names

        # Handle empty data rows string.
        if not isinstance(data_rows_str, str) or not data_rows_str.strip():
            return pd.DataFrame(columns=columns), datetime_column_names
        try:
            # Decode the JSON string of data rows.
            data_list = json.loads(data_rows_str)
            if not isinstance(data_list, list):
                st.error("Error: 'dataRows' does not contain a valid JSON list.")
                return None, datetime_column_names
        except json.JSONDecodeError as e:
            st.error(f"Error decoding 'dataRows' JSON: {e}")
            return None, datetime_column_names

        parsed_rows = []
        expected_cols = len(columns)
        # Process each row from the decoded JSON.
        for i, row_dict in enumerate(data_list):
            if isinstance(row_dict, dict) and 'f' in row_dict and isinstance(row_dict['f'], list):
                row_values = [item.get('v') if isinstance(item, dict) else None for item in row_dict['f']]
                if len(row_values) == expected_cols: parsed_rows.append(row_values)
                else: st.warning(f"Warning: Row {i+1}: Value/Col mismatch ({len(row_values)} values for {expected_cols} cols). Skipped.")
            else: st.warning(f"Warning: Row {i+1}: Bad format. Skipped."); continue

        df = pd.DataFrame(parsed_rows, columns=columns if columns else None)

        # Attempt to convert DataFrame columns to types specified in the schema.
        for col_name in df.columns:
            col = str(col_name) # Ensure column name is a string for dictionary lookups.
            # If schema is completely missing columns list but types exist (unlikely with validation), or col not in types from schema
            if col not in types and not columns:
                 pass # No type information available, leave as is.
            elif col not in types and columns:
                 types[col] = 'STRING' # Default to STRING if column present but type missing in schema.

            target_type = types.get(col, "STRING").upper() # Default to STRING if type not found.
            current_col_data = df[col_name]

            if current_col_data.isnull().all(): continue # Skip conversion for fully null columns.
            try:
                # Type conversion logic based on schema type.
                if target_type in ("DATE", "TIMESTAMP", "DATETIME"):
                    df[col_name] = pd.to_datetime(current_col_data, errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(df[col_name]) and not df[col_name].isnull().all():
                        datetime_column_names.append(col_name)
                elif target_type in ("INTEGER", "INT64", "INT"): df[col_name] = pd.to_numeric(current_col_data, errors='coerce').astype(pd.Int64Dtype())
                elif target_type in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"): df[col_name] = pd.to_numeric(current_col_data, errors='coerce').astype(pd.Float64Dtype())
                elif target_type == "BOOLEAN":
                    map_bool = {'true': True, 'false': False, '1': True, '0': False, True: True, False: False, '': None, None: None}
                    if pd.api.types.is_string_dtype(current_col_data) or current_col_data.dtype == 'object':
                         df[col_name] = current_col_data.astype(str).str.lower().map(map_bool).astype(pd.BooleanDtype())
                    else:
                         df[col_name] = current_col_data.map(map_bool).astype(pd.BooleanDtype())
                elif target_type in ("STRING", "VARCHAR"):
                     if not pd.api.types.is_string_dtype(df[col_name]): df[col_name] = df[col_name].astype(pd.StringDtype())
            except Exception as e: st.warning(f"Warning: Conversion Col '{col}' to type {target_type} failed (err: {e}). Kept as is.")
        return df, datetime_column_names
    except Exception as e:
        # Catch-all for major errors during parsing.
        st.error(f"Major error during N8N data parsing: {e}");
        st.error(traceback.format_exc());
        return None, []


# --- Plotly Charting Logic ---

def get_column_types_for_plotting(df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Classifies DataFrame columns into date, numeric, and categorical types
    to assist in chart generation and UI control population.

    Args:
        df: The Pandas DataFrame to analyze.

    Returns:
        A tuple containing three lists:
            - date_cols: Names of columns identified as datetime.
            - numeric_cols: Names of columns identified as numeric (excluding datetimes).
            - categorical_cols: Names of columns identified as categorical (strings, objects, booleans, or low-cardinality numerics).
    """
    if df is None or df.empty:
        return [], [], []

    all_cols = df.columns.tolist()
    # Identify datetime columns that are not entirely null.
    date_cols = [col for col in all_cols if pd.api.types.is_datetime64_any_dtype(df[col]) and not df[col].isnull().all()]
    # Identify numeric columns, excluding those already classified as date columns.
    numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df[col]) and col not in date_cols]

    # Identify categorical columns (strings, objects, booleans).
    categorical_cols = [
        col for col in all_cols
        if col not in date_cols and col not in numeric_cols and (
            pd.api.types.is_string_dtype(df[col]) or \
            pd.api.types.is_object_dtype(df[col]) or \
            pd.api.types.is_boolean_dtype(df[col])
        )
    ]
    # Additionally, treat low-cardinality numeric columns as potentially categorical.
    # This allows using such numeric columns for grouping or coloring in charts.
    for col in numeric_cols:
        if df[col].nunique() < 15 and col not in date_cols and col not in categorical_cols :
            if col not in categorical_cols: # Ensure not to add duplicates
                 categorical_cols.append(col)
            # Note: These columns remain in numeric_cols as well, as they can serve both purposes.
    return date_cols, numeric_cols, categorical_cols


def _get_plotly_chart_config_from_session(msg_key_prefix: str, default_config: dict) -> dict:
    """
    Retrieves chart configuration for a specific message from Streamlit's session state.
    If no configuration exists, it initializes with a copy of the default_config.
    Ensures that y_axis for line/bar charts is a list for backward compatibility.

    Args:
        msg_key_prefix: A unique prefix string for the session state key, typically message-specific.
        default_config: A dictionary containing default chart configuration values.

    Returns:
        A dictionary with the chart configuration.
    """
    config_key = msg_key_prefix + "_chart_config"
    config = st.session_state.get(config_key, default_config.copy())
    # Ensure all keys from default_config are present.
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    # Data migration: ensure y_axis is a list for line/bar charts if it was stored as a single string.
    if config.get("chart_type") in ["line", "bar"] and config.get("y_axis") and not isinstance(config["y_axis"], list):
        config["y_axis"] = [config["y_axis"]]
    return config

def _store_plotly_chart_config_in_session(msg_key_prefix: str, config: dict):
    """
    Stores the provided chart configuration into Streamlit's session state
    using a message-specific key.

    Args:
        msg_key_prefix: A unique prefix string for the session state key.
        config: The chart configuration dictionary to store.
    """
    st.session_state[msg_key_prefix + "_chart_config"] = config


def render_interactive_plotly_chart(df: pd.DataFrame, chart_type_hint: str, msg_key_prefix: str):
    """
    Renders an interactive Plotly chart based on the input DataFrame and a chart type hint.
    Provides UI controls (select boxes) for users to customize the chart (e.g., axes, color).
    The state of these controls is persisted in the session for each message.

    Args:
        df: The Pandas DataFrame containing data to plot.
        chart_type_hint: A string suggesting an initial chart type (e.g., "line", "bar", "scatter").
        msg_key_prefix: A unique prefix for session state keys to manage chart config per message.
    """
    if df is None or df.empty:
        st.info("ℹ️ No data to visualize.")
        # Display an empty DataFrame if not a table visualization to show raw data section.
        if chart_type_hint != "table":
            st.write("--- Raw Data (empty) ---")
            st.dataframe(pd.DataFrame(), use_container_width=True)
        return

    # Determine column types for populating UI selectors.
    date_cols, numeric_cols, categorical_cols = get_column_types_for_plotting(df)
    all_plottable_cols = sorted(list(set(date_cols + numeric_cols + categorical_cols)))

    # Handle KPI display logic.
    if chart_type_hint == "kpi":
        # Specific conditions for a KPI: single numeric column, single row, no other column types.
        if len(numeric_cols) == 1 and df.shape[0] == 1 and not categorical_cols and not date_cols:
            kpi_col = numeric_cols[0]
            st.write(f"🔢 **KPI Metric**")
            value = df[kpi_col].iloc[0]
            try: # Format numeric value nicely.
                formatted_value = f"{float(value):,.2f}" if pd.notna(value) else "N/A"
            except (ValueError, TypeError): # Fallback for non-numeric or unformattable values.
                formatted_value = str(value) if pd.notna(value) else "N/A"
            st.metric(label=f"Indicator: {kpi_col}", value=formatted_value)
        else:
            # If KPI conditions not met, fall back to displaying a table.
            st.warning("KPI conditions not met. Displaying table.")
            st.dataframe(df, use_container_width=True)
        return

    # If hinted as table or no plottable columns identified, display as a table.
    if chart_type_hint == "table" or not all_plottable_cols:
        st.info("Displaying data as a table.")
        st.dataframe(df, use_container_width=True)
        return

    st.write("### Interactive Visualization")

    # Default configuration for charts.
    default_config = {
        "chart_type": chart_type_hint,
        "x_axis": None, "y_axis": None, "color_col": None, # y_axis can be string or list
        "size_col": None, "facet_row": None, "facet_col": None,
        "barmode": "group", "log_x": False, "log_y": False,
        "agg_func": "sum" # Default aggregation for bar charts.
    }
    # Load or initialize chart configuration from session state.
    chart_config = _get_plotly_chart_config_from_session(msg_key_prefix, default_config)

    # --- UI Controls for Chart Customization ---
    cols_ui = st.columns([1, 1, 1, 1]) # Layout for selectors.

    # Initialize x_axis intelligently if not already set.
    if chart_config["x_axis"] is None:
        if chart_type_hint == "line" and date_cols: chart_config["x_axis"] = date_cols[0]
        elif chart_type_hint == "bar" and categorical_cols: chart_config["x_axis"] = categorical_cols[0]
        elif chart_type_hint == "bar" and date_cols: chart_config["x_axis"] = date_cols[0] # Bar charts can also use date axes.
        elif chart_type_hint == "scatter" and numeric_cols: chart_config["x_axis"] = numeric_cols[0]
        elif all_plottable_cols: chart_config["x_axis"] = all_plottable_cols[0] # Fallback to first plottable.

    # Initialize y_axis intelligently if not already set.
    if chart_config["y_axis"] is None:
        if numeric_cols:
            if chart_config["chart_type"] in ["line", "bar"]:
                chart_config["y_axis"] = [numeric_cols[0]] # Default to list for multi-select y-axis types.
            else:
                chart_config["y_axis"] = numeric_cols[0]   # Default to single string for others.
        # Fallback if no numeric cols, try using other plottable columns.
        elif len(all_plottable_cols) > 1 and chart_config["x_axis"] != all_plottable_cols[1]:
             if chart_config["chart_type"] in ["line", "bar"]: chart_config["y_axis"] = [all_plottable_cols[1]]
             else: chart_config["y_axis"] = all_plottable_cols[1]
        elif all_plottable_cols and chart_config["x_axis"] != all_plottable_cols[0]:
             if chart_config["chart_type"] in ["line", "bar"]: chart_config["y_axis"] = [all_plottable_cols[0]]
             else: chart_config["y_axis"] = all_plottable_cols[0]


    # Chart type selector.
    selected_chart_type = cols_ui[0].selectbox(f"**Use the selectors below to modify the chart.**",
                                             options=["line", "bar", "scatter", "histogram", "box"],
                                             index=["line", "bar", "scatter", "histogram", "box"].index(chart_config["chart_type"]),
                                             key=f"{msg_key_prefix}_type")
    # Adjust y_axis format (list vs. string) if chart type changes between multi-Y and single-Y types.
    if selected_chart_type != chart_config["chart_type"]:
        if selected_chart_type in ["line", "bar"]: # These support multiple Y-axes.
            if chart_config["y_axis"] and not isinstance(chart_config["y_axis"], list):
                chart_config["y_axis"] = [chart_config["y_axis"]] # Convert to list.
            elif not chart_config["y_axis"] and numeric_cols: # If Y was None, init with first numeric.
                chart_config["y_axis"] = [numeric_cols[0]]
        else: # Scatter, box, histogram usually take a single Y (or None for histogram count).
            if chart_config["y_axis"] and isinstance(chart_config["y_axis"], list):
                chart_config["y_axis"] = chart_config["y_axis"][0] if chart_config["y_axis"] else None # Take first if list, or None if empty.
            elif not chart_config["y_axis"] and numeric_cols: # If Y was None, init with first numeric.
                 chart_config["y_axis"] = numeric_cols[0]
    chart_config["chart_type"] = selected_chart_type


    # X-axis selector.
    chart_config["x_axis"] = cols_ui[1].selectbox("X-axis", options=[None] + all_plottable_cols,
                                         index=([None] + all_plottable_cols).index(chart_config["x_axis"]) if chart_config["x_axis"] in ([None] + all_plottable_cols) else 0,
                                         key=f"{msg_key_prefix}_x")

    # Y-axis selector: multiselect for line/bar, single selectbox for others.
    if chart_config["chart_type"] in ["line", "bar"]:
        current_y_values = chart_config["y_axis"]
        # Ensure current_y_values is a list for multiselect default.
        if not isinstance(current_y_values, list):
            current_y_values = [current_y_values] if current_y_values is not None else []
        
        # Filter current_y_values to only include valid numeric cols for selection (Y-axes are usually numeric).
        default_y_selection = [val for val in current_y_values if val in numeric_cols]

        chart_config["y_axis"] = cols_ui[2].multiselect("Y-axis (multiple allowed)",
                                               options=numeric_cols,
                                               default=default_y_selection,
                                               key=f"{msg_key_prefix}_y_multiselect")
        # For bar charts, Y-axis can be empty (implies count). For line charts, it's often required.
        if not chart_config["y_axis"] and chart_config["chart_type"] == "bar":
            pass # Y can be empty/None for bar (implies count of X categories).
        elif not chart_config["y_axis"] and chart_config["chart_type"] == "line" and numeric_cols:
             # If line chart and Y becomes empty, user needs to select one.
             pass

    else: # Single Y-axis for scatter, box, histogram (if y is used for values).
        y_options = [None] + all_plottable_cols # For histogram, Y can be None (count).
        current_y_value = chart_config["y_axis"]
        # If y_axis was a list (e.g., from switching from line/bar), take the first element.
        if isinstance(current_y_value, list):
            current_y_value = current_y_value[0] if current_y_value else None
        
        y_idx = 0 # Default to first option (None).
        if current_y_value in y_options:
            y_idx = y_options.index(current_y_value)

        chart_config["y_axis"] = cols_ui[2].selectbox("Y-axis", options=y_options,
                                             index=y_idx,
                                             key=f"{msg_key_prefix}_y_select")

    # Color selector: allows coloring by categorical or numeric columns.
    chart_config["color_col"] = cols_ui[3].selectbox("Color by", options=[None] + categorical_cols + numeric_cols,
                                           index=([None] + categorical_cols + numeric_cols).index(chart_config["color_col"]) if chart_config["color_col"] in ([None] + categorical_cols + numeric_cols) else 0,
                                           key=f"{msg_key_prefix}_color")

    # Store the (potentially updated) chart configuration back into session state.
    _store_plotly_chart_config_in_session(msg_key_prefix, chart_config)

    # --- Data Preparation for Plotly (e.g., aggregation for bar charts) ---
    plot_df = df.copy() # Work on a copy to avoid modifying the original DataFrame.

    # Aggregate data for bar charts if X is categorical/date and Y is numeric.
    if chart_config["chart_type"] == "bar" and chart_config["x_axis"] and chart_config["y_axis"] and \
       (chart_config["x_axis"] in categorical_cols or chart_config["x_axis"] in date_cols):

        y_axes_for_bar = chart_config["y_axis"]
        # Ensure y_axes_for_bar is a list for consistent processing.
        if not isinstance(y_axes_for_bar, list):
            y_axes_for_bar = [y_axes_for_bar]

        # Filter to only include numeric y-axes for aggregation.
        numeric_y_axes_to_agg = [col for col in y_axes_for_bar if col in numeric_cols]

        if numeric_y_axes_to_agg: # Proceed with aggregation only if there are valid numeric y-axes.
            try:
                grouping_cols = [chart_config["x_axis"]]
                # Add color column to grouping if it's specified, different from x_axis, and exists.
                if chart_config["color_col"] and \
                   chart_config["color_col"] != chart_config["x_axis"] and \
                   chart_config["color_col"] in plot_df.columns:
                    grouping_cols.append(chart_config["color_col"])

                # Ensure all grouping columns are valid and exist in plot_df.
                valid_grouping_cols = [col for col in grouping_cols if col in plot_df.columns]
                if not valid_grouping_cols:
                    st.warning("Invalid grouping columns for bar chart aggregation. Skipping aggregation.")
                else:
                    # Define aggregation specification (e.g., sum, mean).
                    agg_spec = {y_col: chart_config["agg_func"] for y_col in numeric_y_axes_to_agg}
                    plot_df = plot_df.groupby(valid_grouping_cols, as_index=False, observed=True).agg(agg_spec)
                    # Plotly's px.bar can handle multiple y columns if plot_df is now aggregated and "wide" enough.
            except Exception as e:
                st.error(f"Error during bar chart data aggregation: {e}")
                st.error(traceback.format_exc(limit=2)) # Provide limited traceback for debugging.
                return
        # If no numeric_y_axes_to_agg, plot_df remains unaggregated.
        # px.bar might still work if y_axis is None (for count).

    # --- Generate Plotly Figure ---
    fig = None
    y_for_plot = chart_config["y_axis"] # This can be a string, a list of strings, or None.

    try:
        # X-axis is generally required for most charts (except perhaps some histograms of a single variable).
        if chart_config["x_axis"] is None and chart_config["chart_type"] != "histogram":
             st.info("Please select an X-axis.")
             return

        # For line and bar charts, if y_for_plot is an empty list, adjust its meaning.
        if chart_config["chart_type"] in ["line", "bar"] and isinstance(y_for_plot, list) and not y_for_plot:
            if chart_config["chart_type"] == "line":
                st.info("Please select at least one Y-axis for the line chart.")
                return
            y_for_plot = None # For bar chart, an empty list of Ys means count on X.

        # Generate chart based on selected type.
        if chart_config["chart_type"] == "line":
            if not y_for_plot: # Line chart requires a Y-axis.
                st.info("Please select Y-axis for the line chart.")
                return
            fig = px.line(plot_df, x=chart_config["x_axis"], y=y_for_plot, color=chart_config["color_col"],
                          log_x=chart_config["log_x"], log_y=chart_config["log_y"],
                          facet_row=chart_config["facet_row"], facet_col=chart_config["facet_col"],
                          title=f"Line chart of {y_for_plot if y_for_plot else ''} vs {chart_config['x_axis'] or ''}")
        elif chart_config["chart_type"] == "bar":
            # y_for_plot can be a list for multi-bar, string for single, or None for count.
            fig = px.bar(plot_df, x=chart_config["x_axis"], y=y_for_plot, color=chart_config["color_col"],
                         barmode=chart_config["barmode"], log_x=chart_config["log_x"], log_y=chart_config["log_y"],
                         facet_row=chart_config["facet_row"], facet_col=chart_config["facet_col"],
                         title=f"Bar chart of {y_for_plot or 'Count'} by {chart_config['x_axis'] or ''}")
        elif chart_config["chart_type"] == "scatter":
            if y_for_plot is None or (isinstance(y_for_plot, list) and not y_for_plot): # Y is required for scatter.
                st.info("Please select a Y-axis for the scatter plot.")
                return
            # If y_for_plot became a list (e.g. from line/bar type switch), take first element for scatter.
            y_scatter = y_for_plot[0] if isinstance(y_for_plot, list) and y_for_plot else y_for_plot

            fig = px.scatter(plot_df, x=chart_config["x_axis"], y=y_scatter, color=chart_config["color_col"],
                             size=chart_config["size_col"], log_x=chart_config["log_x"], log_y=chart_config["log_y"],
                             facet_row=chart_config["facet_row"], facet_col=chart_config["facet_col"],
                             title=f"Scatter plot of {y_scatter} vs {chart_config['x_axis']}")
        elif chart_config["chart_type"] == "histogram":
            # For histogram, y_for_plot specifies values for aggregation; if None, it's a count.
            # Assuming histogram y is single or None for simplicity.
            y_hist = y_for_plot[0] if isinstance(y_for_plot, list) and y_for_plot else y_for_plot
            fig = px.histogram(plot_df, x=chart_config["x_axis"], y=y_hist, color=chart_config["color_col"],
                               log_y=chart_config["log_y"],
                               facet_row=chart_config["facet_row"], facet_col=chart_config["facet_col"],
                               title=f"Histogram of {chart_config['x_axis']}{f' by {y_hist}' if y_hist else ''}")
        elif chart_config["chart_type"] == "box":
            if not y_for_plot: # Box plot requires a Y-axis.
                st.info("Please select Y-axis for the box plot.")
                return
            # If y_for_plot became a list, take first element for box plot.
            y_box = y_for_plot[0] if isinstance(y_for_plot, list) and y_for_plot else y_for_plot
            fig = px.box(plot_df, x=chart_config["x_axis"], y=y_box, color=chart_config["color_col"],
                         log_y=chart_config["log_y"],
                         facet_row=chart_config["facet_row"], facet_col=chart_config["facet_col"],
                         title=f"Box plot of {y_box or ''} by {chart_config['x_axis'] or ''}")

        if fig:
            fig.update_layout(title_x=0.5) # Center the chart title.
            # Use a unique key for the Plotly chart to ensure proper state management and updates in Streamlit.
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_fig_{msg_key_prefix}")
        elif chart_config["x_axis"]: # If no figure but X-axis selected, prompt for more info.
            st.info(f"Select Y-axis or check configuration for {chart_config['chart_type']} chart.")

    except Exception as e:
        st.error(f"😳 Failed to generate Plotly chart: {e}")
        st.error(traceback.format_exc(limit=5)) # Increased limit for more traceback details.


def select_and_plot_chart(df: pd.DataFrame, chart_override: str | None = None, msg_key_prefix: str = "default_msg"):
    """
    Orchestrates chart rendering. It auto-detects a suitable initial chart type ("hint")
    based on data properties. However, if a chart configuration is already stored in the
    session state for the given message (msg_key_prefix), that configuration's chart type
    will be used, allowing user customizations to persist.
    The `chart_override` parameter is largely a legacy/unused concept now that interactive
    Plotly controls and session state manage chart type.

    Args:
        df: The Pandas DataFrame to plot.
        chart_override: (Largely deprecated) An optional string to force a chart type.
        msg_key_prefix: A unique prefix for session state keys, associating charts with messages.
    """
    container = st.container() # Use a container for layout.
    with container:
        date_cols, numeric_cols, categorical_cols = get_column_types_for_plotting(df)

        # Auto-detect a sensible default chart type hint based on data characteristics.
        auto_detected_hint = "table" # Default to table if no other type fits.
        if df is None or df.empty: auto_detected_hint = "table"
        elif len(numeric_cols) == 1 and df.shape[0] == 1 and not categorical_cols and not date_cols: auto_detected_hint = "kpi"
        elif date_cols and numeric_cols: auto_detected_hint = "line" # Good for time series.
        elif categorical_cols and numeric_cols: auto_detected_hint = "bar" # Good for categorical comparisons.
        elif len(numeric_cols) >= 2: auto_detected_hint = "scatter" # Good for relationships between two numerics.
        elif numeric_cols or categorical_cols : auto_detected_hint = "histogram" # Good for distributions.

        final_chart_hint = auto_detected_hint
        # Check if a user-selected chart type exists in session state for this message.
        current_config_for_message = st.session_state.get(msg_key_prefix + "_chart_config", None)
        if current_config_for_message and current_config_for_message.get("chart_type"):
            final_chart_hint = current_config_for_message["chart_type"] # User's choice overrides auto-detection.
        
        # Render the interactive chart using the determined hint.
        render_interactive_plotly_chart(df, final_chart_hint, msg_key_prefix)

        # Display the raw data below the chart.
        st.write("### Raw Data")
        try:
            # Adjust height for small DataFrames.
            height_val = 150 if df is not None and df.shape[0] > 3 else None
            st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True, height=height_val)
        except Exception as e:
            st.error(f"Error displaying DataFrame: {e}")
            st.text("Raw text fallback:")
            st.text(df.to_string() if df is not None else "")


# --- Streamlit UI / Webhook call / Message handling ---
def call_n8n_webhook(prompt: str) -> dict | list | None:
    """
    Calls a configured N8N webhook with the user's prompt and current session ID.
    Handles authentication and error responses from the webhook.

    Args:
        prompt: The user's input string (e.g., a question for the AI).

    Returns:
        The JSON response from the N8N webhook as a dictionary or list,
        or None if an error occurs.
    """
    try:
        # Load N8N configuration from Streamlit secrets.
        cfg = st.secrets.get("n8n",{})
        url=cfg.get("webhook_url")
        usr=cfg.get("username")
        pwd=cfg.get("password")
        key=cfg.get("input_key","userPrompt") # The key in JSON payload for the prompt.
    except FileNotFoundError:
        st.error("`secrets.toml` not found. N8N webhook cannot be called.")
        return None
    except Exception as e: # Catch other potential errors reading secrets.
        st.error(f"Error reading `secrets.toml`: {e}")
        return None

    if not url:
        st.error("N8N webhook URL not configured in `secrets.toml`.")
        return None
    sid = st.session_state.get("sessionId")
    if not sid:
        st.error("Session ID missing. Cannot call N8N webhook.")
        return None

    # Prepare authentication if username and password are provided.
    auth = HTTPBasicAuth(usr, pwd) if usr and pwd else None
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data_payload = {key: prompt, "sessionId": sid}

    try:
        # Make the POST request to the N8N webhook.
        response = requests.post(url, headers=headers, data=json.dumps(data_payload), auth=auth, timeout=180) # 3-minute timeout.
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx).
        try:
            return response.json() # Parse JSON response.
        except json.JSONDecodeError:
            st.error(f"Webhook response was not valid JSON. Status: {response.status_code}")
            st.code(response.text[:1000], language='text') # Show beginning of text response.
            return None
    except requests.exceptions.Timeout:
        st.error("⏳ N8N webhook request timed out.")
        return None
    except requests.exceptions.HTTPError as e:
         st.error(f"❌ HTTP Error from N8N: {e.response.status_code} {e.response.reason}")
         if e.response is not None:
             st.code(e.response.text[:1000] if e.response.text else "(No response body)", language='text')
         return None
    except requests.exceptions.ConnectionError as e:
        st.error(f"🔌 N8N Connection Error: {e}")
        return None
    except requests.exceptions.RequestException as e: # Catch other requests-related errors.
        st.error(f"🌐 N8N Request Error: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors.
        st.error(f"😳 Unexpected error calling N8N: {e}")
        st.error(traceback.format_exc())
        return None


# --- Main Streamlit Application Setup ---
st.set_page_config(page_title="AI Chart Assistant", page_icon="📊", layout="wide")

# --- Sidebar for Session Information and Controls ---
with st.sidebar:
    st.header("ℹ️ Session Info")
    # Initialize sessionId if not already in session state.
    if "sessionId" not in st.session_state:
        st.session_state.sessionId = str(uuid.uuid4())
    st.code(f"Session ID: {st.session_state.sessionId}", language=None)

    # Initialize messages list for chat history if not present.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.write(f"Messages in history: {len(st.session_state.messages)}")

    # Button to clear chat history and reset the session.
    if st.button("Clear Chat History & Reset Session"):
        st.session_state.messages = []
        # Clear any stored chart configurations from session state.
        for key in list(st.session_state.keys()): # Iterate over a copy of keys for safe deletion.
            if key.endswith("_chart_config"):
                del st.session_state[key]
        st.session_state.sessionId = str(uuid.uuid4()) # Generate a new session ID.
        st.session_state.new_prompt_to_process = None # Clear any pending prompt.
        st.rerun() # Rerun the app to reflect changes.

# --- Main Page Title and Introduction ---
st.title("📊 AI Data Visualization Assistant")
st.caption("Powered by Streamlit, n8n & Plotly Express. Ask questions, then customize your charts!")

# Initialize state for managing new prompts if it doesn't exist.
if "new_prompt_to_process" not in st.session_state:
    st.session_state.new_prompt_to_process = None

# --- Display Chat History ---
for msg_idx, message_from_state in enumerate(st.session_state.messages):
    with st.chat_message(message_from_state["role"]): # Display user/assistant role.
        st.markdown(message_from_state["content"]) # Display text content.

        # If SQL query is present in the message, display it in an expander.
        if message_from_state.get("sqlQuery"):
            with st.expander("View SQL Query", expanded=False):
                st.code(message_from_state["sqlQuery"], language='sql')

        # Unique key for rendering elements associated with this message.
        msg_render_key = message_from_state.get("message_id", f"msgidx_{msg_idx}")

        # If DataFrame JSON is present, parse and display it (potentially as a chart).
        if "dataframe_json" in message_from_state and isinstance(message_from_state.get("dataframe_json"), str):
            df_display = None
            try:
                df_display_json_str = message_from_state["dataframe_json"]
                # Handle cases where dataframe_json might be an empty string.
                if not df_display_json_str or df_display_json_str.isspace():
                     # Avoid redundant "empty data" messages if there's SQL and no text content.
                     if not (message_from_state.get("sqlQuery") and not message_from_state.get("content")):
                        if not message_from_state.get("content"): st.caption(f"Note (msg {msg_render_key}): Data for this message is empty.")
                     continue # Skip plotting for empty data string.

                # Deserialize DataFrame from JSON string.
                # Assumes 'iso' date format was used for `to_json` for proper datetime parsing.
                df_display = pd.read_json(StringIO(df_display_json_str), orient='split')
                
                # Notify if query returned no data.
                if df_display is not None and df_display.empty and message_from_state.get("sqlQuery"):
                     st.caption(f"ℹ️ (msg {msg_render_key}) The SQL query returned no data to display.")

                if df_display is not None:
                    # Call the main charting function. chart_override is less relevant here
                    # as session state for msg_render_key will preserve user choices.
                    select_and_plot_chart(df_display, chart_override=None, msg_key_prefix=msg_render_key)

            except json.JSONDecodeError as e_json_hist:
                st.error(f"Error decoding DataFrame JSON for historical message {msg_render_key}: {e_json_hist}")
            except Exception as e_display_hist: # Catch-all for other display errors.
                st.error(f"Error displaying chart for historical message {msg_render_key}: {e_display_hist}\n{traceback.format_exc(limit=2)}")
                if df_display is not None: # Fallback to raw DataFrame display on error.
                    st.write(f"--- Raw Data (Fallback for msg {msg_render_key}) ---")
                    st.dataframe(df_display, height=150, use_container_width=True)

        # Display error details if present in the message.
        elif "error_details" in message_from_state and message_from_state["error_details"] is not None:
            st.error(message_from_state["error_details"])
        # If only SQL query is present without data or explicit content, show a note.
        elif message_from_state.get("sqlQuery") and not message_from_state.get("content") and not message_from_state.get("dataframe_json"):
            st.caption("SQL query provided. If it's a SELECT statement, it might not have been executed or returned no data.")


# --- Handle New User Input ---
if prompt_input := st.chat_input("Ask your question here...", key="chat_input_main"):
    msg_id = str(uuid.uuid4()) # Generate a unique ID for the user message.
    # Add user's message to chat history.
    st.session_state.messages.append({"role": "user", "content": prompt_input, "message_id": msg_id})
    # Set flag to process this new prompt.
    st.session_state.new_prompt_to_process = prompt_input
    st.rerun() # Rerun to display user message immediately and trigger processing block.

# --- Process New Prompt (if any) ---
if st.session_state.new_prompt_to_process:
    prompt_to_run = st.session_state.new_prompt_to_process
    st.session_state.new_prompt_to_process = None # Clear the flag.

    with st.spinner("🧠 Querying AI Assistant... Please wait..."):
        n8n_resp = call_n8n_webhook(prompt_to_run) # Call the backend N8N webhook.

    assistant_message_id = str(uuid.uuid4()) # Unique ID for the assistant's response.
    # Initialize structure for assistant's message data.
    assist_msg_data = {
        "role": "assistant", "content": "", "message_id": assistant_message_id,
        "dataframe_json": None, "datetime_col_names": [], # To store parsed data and identified datetime columns.
        "error_details": None, "sqlQuery": None
    }

    if n8n_resp is not None:
        # Determine if response is a list (common from N8N split-in-batches) or a single dict.
        response_item = None
        if isinstance(n8n_resp, list):
            if len(n8n_resp) > 0:
                response_item = n8n_resp[0] # Process the first item if it's a list.
            else:
                assist_msg_data["error_details"] = "N8N webhook returned an empty list."
        elif isinstance(n8n_resp, dict):
            response_item = n8n_resp # Process the dict directly.
        else:
            assist_msg_data["error_details"] = f"Unexpected response type from webhook. Expected list or dict, got {type(n8n_resp)}. Response: {str(n8n_resp)[:200]}"

        if response_item and isinstance(response_item, dict):
            assist_msg_data["sqlQuery"] = response_item.get("sqlQuery")
            ai_answer = response_item.get("aiAgentAnswer") # Textual answer from AI.
            
            # Check if data is expected to be displayed.
            if response_item.get("dataToDisplay") == "true" or response_item.get("dataToDisplay") is True :
                assist_msg_data["content"] = ai_answer if ai_answer else "_Displaying data..._" # Default message if only data.
                rows_str, schema = response_item.get("dataRows"), response_item.get("dataSchema")

                if isinstance(rows_str, str) and isinstance(schema, dict):
                    if not rows_str.strip(): # Handle empty data rows string.
                        # Store empty DataFrame as JSON.
                        assist_msg_data["dataframe_json"] = pd.DataFrame().to_json(orient='split', date_format='iso')
                        # Update content if it was the default "Displaying data...".
                        if not assist_msg_data["content"] or assist_msg_data["content"] == "_Displaying data..._":
                             assist_msg_data["content"] = ai_answer if ai_answer else "The query returned no data."
                    else:
                        # Parse the received data rows and schema.
                        df_new, dt_cols_new = parse_n8n_data_to_dataframe(rows_str, schema)
                        if df_new is not None:
                            assist_msg_data["datetime_col_names"] = dt_cols_new # Store identified datetime columns (currently not directly used later, but good for future).
                            try:
                                # Serialize DataFrame to JSON for storing in session state.
                                # `date_format='iso'` helps preserve datetime precision.
                                # `default_handler=str` for any non-serializable types.
                                assist_msg_data["dataframe_json"] = df_new.to_json(orient='split', date_format='iso', default_handler=str)
                            except Exception as e_json:
                                assist_msg_data["error_details"] = f"Error saving DataFrame to JSON: {e_json}"
                        else: # Parsing failed.
                            current_error = assist_msg_data.get("error_details") or ""
                            assist_msg_data["error_details"] = (current_error + " Failed to parse received data for new chart.").strip()
                else:
                    # Data was expected but 'dataRows' or 'dataSchema' were missing/invalid.
                    assist_msg_data["error_details"] = "Warning: Data ('dataRows' or 'dataSchema') for display is missing or has an invalid type."
            
            # Fallback to other potential output fields from N8N.
            elif "output" in response_item: # E.g., a general text output from an LLM node.
                assist_msg_data["content"] = response_item["output"]
            elif ai_answer: # If only AI answer is provided.
                assist_msg_data["content"] = ai_answer
            elif assist_msg_data["sqlQuery"] and not assist_msg_data["content"]:
                # If only SQL query is returned, provide a placeholder message.
                assist_msg_data["content"] = "_SQL query provided. Execute to see data or refine the query._"
            # If none of the expected fields are present.
            elif not assist_msg_data["content"] and not assist_msg_data["sqlQuery"]:
                existing_error = assist_msg_data.get("error_details") or ""
                error_msg = (f"Unrecognized JSON response format or missing main content fields. "
                                f"N8N item (first 500 chars): {json.dumps(response_item, default=str)[:500]}")
                assist_msg_data["error_details"] = (existing_error + " " + error_msg).strip()
        
        elif not assist_msg_data.get("error_details"): # If response_item was None/empty but no specific error yet.
            assist_msg_data["error_details"] = "Received no processable item from N8N webhook."
    else: # n8n_resp is None (webhook call failed).
         assist_msg_data["content"] = assist_msg_data.get("content","") # Keep any content if set by prior logic.
         if not assist_msg_data["error_details"]: # If no specific error message from call_n8n_webhook.
            assist_msg_data["error_details"] = "Error connecting to N8N or N8N returned no response. Check service status."

    # Add assistant's message to chat history if it has any content, data, error, or SQL.
    if assist_msg_data["content"] or assist_msg_data.get("dataframe_json") is not None or assist_msg_data.get("error_details") or assist_msg_data.get("sqlQuery"):
        st.session_state.messages.append(assist_msg_data)
    else:
        # Fallback for truly empty or unprocessable responses.
        st.session_state.messages.append({
            "role": "assistant", "content": "_I received an empty or unprocessable response._",
            "message_id": assistant_message_id, "error_details": "Empty or unprocessable response from backend."
        })
    st.rerun() # Rerun to display the assistant's response.