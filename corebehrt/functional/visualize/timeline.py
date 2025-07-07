import pandas as pd
import plotly.express as px


def visualize_meds_timeline(
    meds_df: pd.DataFrame,
    events_of_interest: dict,
    subject_ids: list = None,
    n_random_subjects: int = 5,
    title: str = "Patient Event Timelines",
):
    """
    Creates an interactive timeline visualization with evenly spaced timelines
    and subject ID labels on the right.
    """
    # 1. Prepare the data
    df = meds_df.copy()
    df["time"] = pd.to_datetime(df["time"])

    codes_to_plot = list(events_of_interest.keys())
    df = df[df["code"].isin(codes_to_plot)]

    # Select subjects
    if subject_ids:
        plot_ids = subject_ids
    else:
        all_ids = df["subject_id"].unique()
        if len(all_ids) < n_random_subjects:
            n_random_subjects = len(all_ids)
        plot_ids = sorted(
            pd.Series(all_ids).sample(n_random_subjects, random_state=42).tolist()
        )

    df_plot = df[df["subject_id"].isin(plot_ids)].sort_values("time")

    if df_plot.empty:
        print("No data available for the selected subjects and event codes.")
        return

    # --- NEW: Map subject_id to an evenly spaced integer y-position ---
    id_to_y = {subject_id: i for i, subject_id in enumerate(plot_ids)}
    df_plot["y_pos"] = df_plot["subject_id"].map(id_to_y)

    # 2. Create the plot using the new y-position
    fig = px.scatter(
        df_plot,
        x="time",
        y="y_pos",  # Use the integer position for the y-axis
        color="code",
        symbol="code",
        hover_name="code",
        # Include subject_id in hover data for clarity
        hover_data={"time": "|%Y-%m-%d", "subject_id": True, "y_pos": False},
        title=title,
        color_discrete_map={k: v["color"] for k, v in events_of_interest.items()},
        symbol_map={
            k: v.get("symbol", "circle") for k, v in events_of_interest.items()
        },
    )

    # 3. Enhance the layout
    fig.update_layout(
        plot_bgcolor="white",
        yaxis_title=None,
        xaxis_title="Timeline",
        legend_title="Event Type",
        yaxis=dict(
            showticklabels=False,  # Hide the y-axis ticks (0, 1, 2...)
            showline=False,
            showgrid=False,
            zeroline=False,
        ),
        xaxis=dict(
            showline=True,
            linecolor="black",
            showgrid=False,
        ),
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))

    # 4. Add horizontal lines and subject ID annotations
    for subject_id, y_pos in id_to_y.items():
        subject_data = meds_df[meds_df["subject_id"] == subject_id]
        if not subject_data.empty:
            min_time = subject_data["time"].min()
            max_time = subject_data["time"].max()

            # Add a light grey line for the timeline
            fig.add_shape(
                type="line",
                x0=min_time,
                y0=y_pos,
                x1=max_time,
                y1=y_pos,
                line=dict(color="lightgrey", width=1),
                layer="below",
            )

            # Add the subject ID as a text annotation on the right
            fig.add_annotation(
                x=max_time,
                y=y_pos,
                text=f"<b>{subject_id}</b>",
                showarrow=False,
                xanchor="left",
                xshift=10,  # Shift text 10 pixels to the right
                font=dict(size=10, color="black"),
            )

    fig.show()
