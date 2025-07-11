from corebehrt.functional.visualize.timeline import visualize_meds_timeline


import pandas as pd

# Assuming 'simulated_df' is the output from your CausalSimulator
# and 'config' is your simulation configuration object.
simulated_df = pd.read_parquet(
    r"C:\Users\fjn197\PhD\projects\PHAIR\pipelines\PHAIR_EHR\example_data\synthea_meds_causal\tuning\0.parquet"
)
# 1. Define the events you want to visualize and their appearance
events_to_show = {
    "EXPOSURE": {"color": "#1f77b4", "symbol": "circle"},  # Muted Blue
    "OUTCOME": {"color": "#d62728", "symbol": "diamond"},  # Red
    "OUTCOME_2": {"color": "#ff7f0e", "symbol": "x"},  # Orange
    "OUTCOME_3": {"color": "#2ca02c", "symbol": "cross"},  # Green
    "D/25675004": {"color": "#1f77b4", "symbol": "circle-open"},  # confounders
    "D/431855005": {"color": "#1f77b4", "symbol": "circle-open"},  # confounders
    "D/80583007": {"color": "#1f77b4", "symbol": "circle-open"},  # confounders
    "D/105531004": {"color": "#1f77b4", "symbol": "circle-open"},  # confounders
    "D/65363002": {"color": "#1f77b4", "symbol": "circle-open"},  # confounders
    "D/125605004": {"color": "violet", "symbol": "circle-open"},  # prognostic outcome 1
    "D/384709000": {"color": "violet", "symbol": "circle-open"},  # prognostic outcome 1
    "D/157141000119108": {
        "color": "orange",
        "symbol": "circle-open",
    },  # prognostic outcome 2
    "D/1121000119107": {
        "color": "orange",
        "symbol": "circle-open",
    },  # prognostic outcome 2
    "D/65363002": {"color": "green", "symbol": "circle-open"},  # prognostic outcome 3
    "D/384709000": {"color": "green", "symbol": "circle-open"},  # prognostic outcome 3
}

# 2. Call the visualization function
# To plot 10 random subjects
import plotly.io

plotly.io.renderers.default = "svg"
visualize_meds_timeline(
    meds_df=simulated_df,
    events_of_interest=events_to_show,
    n_random_subjects=5,
    title="Timelines for Simulated Exposure and Outcomes",
    save_path="outputs/figs/meds_timeline.svg",
)

# To plot specific subjects
# visualize_meds_timeline(
#     meds_df=simulated_df,
#     events_of_interest=events_to_show,
#     subject_ids=[101, 205, 312] # Example IDs
# )
