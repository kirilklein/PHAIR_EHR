import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple


def visualize_meds_timeline(
    meds_df: pd.DataFrame,
    events_of_interest: dict,
    subject_ids: list = None,
    n_random_subjects: int = 5,
    title: str = "Patient Event Timelines",
    follow_up_times: dict = None,
    index_dates: dict = None,
    index_date_marker: str = "I",
    save_path: str = None,
):
    """
    Creates an interactive timeline visualization for event data.

    Args:
        meds_df: DataFrame with ['subject_id', 'time', 'code'].
        events_of_interest: Dictionary mapping event codes to visual properties.
        subject_ids: A list of specific subject_ids to plot.
        n_random_subjects: Number of random subjects to plot if subject_ids is None.
        title: The title of the plot.
        follow_up_times: Optional dict mapping subject_id to (start, end) tuples for brackets.
        index_dates: Optional dict mapping subject_id to a single index time.
                     Example: {101: '2022-05-01', 205: '2021-12-15'}
        index_date_marker: The symbol to use for all index dates. Defaults to '▼'.
    """
    # 1. Prepare data for plotting
    df_plot, id_to_y = _prepare_plot_data(
        meds_df, events_of_interest, subject_ids, n_random_subjects
    )
    if df_plot is None:
        return

    # 2. Create the base scatter plot
    fig = _create_base_figure(df_plot, events_of_interest, title)

    # 3. Add all annotations, shapes, and markers
    _add_annotations_and_shapes(
        fig, meds_df, id_to_y, index_dates, follow_up_times, index_date_marker
    )

    # 4. Finalize layout
    fig.update_layout(
        plot_bgcolor="white",
        yaxis_title=None,
        xaxis_title="Timeline",
        legend_title="Event Type",
        yaxis=dict(
            showticklabels=False, showline=False, showgrid=False, zeroline=False
        ),
        xaxis=dict(showline=True, linecolor="black", showgrid=False),
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))

    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def _prepare_plot_data(
    meds_df: pd.DataFrame,
    events_of_interest: dict,
    subject_ids: Optional[List[Any]],
    n_random_subjects: int,
) -> Tuple[Optional[pd.DataFrame], Dict[Any, int]]:
    """Filters and prepares the DataFrame for plotting."""
    df = meds_df.copy()
    df["time"] = pd.to_datetime(df["time"])

    codes_to_plot = list(events_of_interest.keys())
    df = df[df["code"].isin(codes_to_plot)]

    if not subject_ids:
        all_ids = df["subject_id"].unique()
        if len(all_ids) < n_random_subjects:
            n_random_subjects = len(all_ids)
        subject_ids = sorted(
            pd.Series(all_ids).sample(n_random_subjects, random_state=42).tolist()
        )

    df_plot = df[df["subject_id"].isin(subject_ids)].sort_values("time")
    if df_plot.empty:
        print("No data available for the selected subjects and event codes.")
        return None, {}

    id_to_y = {subject_id: i for i, subject_id in enumerate(subject_ids)}
    df_plot["y_pos"] = df_plot["subject_id"].map(id_to_y)

    return df_plot, id_to_y


def _create_base_figure(
    df_plot: pd.DataFrame, events_of_interest: dict, title: str
) -> go.Figure:
    """Creates the initial scatter plot figure."""
    fig = px.scatter(
        df_plot,
        x="time",
        y="y_pos",
        color="code",
        symbol="code",
        hover_name="code",
        hover_data={"time": "|%Y-%m-%d", "subject_id": True, "y_pos": False},
        title=title,
        color_discrete_map={k: v["color"] for k, v in events_of_interest.items()},
        symbol_map={
            k: v.get("symbol", "circle") for k, v in events_of_interest.items()
        },
    )
    return fig


def _add_annotations_and_shapes(
    fig: go.Figure,
    df: pd.DataFrame,
    id_to_y: Dict[Any, int],
    index_dates: Optional[Dict],
    follow_up_times: Optional[Dict],
    index_date_marker: str,
):
    """Adds lines, labels, and optional markers to the figure."""
    for subject_id, y_pos in id_to_y.items():
        subject_data = df[df["subject_id"] == subject_id]
        if subject_data.empty:
            continue

        min_time, max_time = subject_data["time"].min(), subject_data["time"].max()

        fig.add_shape(
            type="line",
            x0=min_time,
            y0=y_pos,
            x1=max_time,
            y1=y_pos,
            line=dict(color="lightgrey", width=1),
            layer="below",
        )

        fig.add_annotation(
            x=max_time,
            y=y_pos,
            text=f"<b>{subject_id}</b>",
            showarrow=False,
            xanchor="left",
            xshift=10,
            font=dict(size=10, color="black"),
        )

        # Add optional index date markers using the simplified format
        if index_dates and subject_id in index_dates:
            index_time = index_dates[subject_id]
            fig.add_annotation(
                x=pd.to_datetime(index_time),
                y=y_pos,
                text=index_date_marker,
                showarrow=False,
                font=dict(color="red", size=15),
            )

        if follow_up_times and subject_id in follow_up_times:
            start_fu, end_fu = follow_up_times[subject_id]
            fig.add_annotation(
                x=pd.to_datetime(start_fu),
                y=y_pos,
                text="[",
                showarrow=False,
                xanchor="right",
                font=dict(color="black", size=15),
            )
            fig.add_annotation(
                x=pd.to_datetime(end_fu),
                y=y_pos,
                text="]",
                showarrow=False,
                xanchor="left",
                font=dict(color="black", size=15),
            )
