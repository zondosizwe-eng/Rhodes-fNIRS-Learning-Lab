
import streamlit as st

st.set_page_config(
    page_title="Rhodes fNIRS Learning Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Rhodes fNIRS Learning Lab")

st.write("Welcome to our interactive fNIRS teaching application.")

st.subheader("Introduction")

st.write(
    "This application introduces students to functional "
    "near-infrared spectroscopy (fNIRS) analysis using "
    "MNE-Python and MNE-NIRS."
)

st.success("The application is ready for learning!")

# Optional teaching mode: use GLM results exported from a completed live run.
from pathlib import Path
import pandas as pd
import plotly.express as px

PRECOMPUTED_DIR = Path(__file__).resolve().parent / "precomputed_glm"
st.divider()
st.subheader("GLM teaching mode")
glm_mode = st.radio(
    "Choose how to explore the general linear model",
    ["Quick demonstration (precomputed results)", "Full analysis (run the GLM)"],
    index=0,
    horizontal=True,
    key="glm_teaching_mode",
)
precomputed_files = {
    "GLM coefficients": PRECOMPUTED_DIR / "glm_results.csv",
    "Left-hand tapping contrast": PRECOMPUTED_DIR / "left_contrast.csv",
    "Right-hand tapping contrast": PRECOMPUTED_DIR / "right_contrast.csv",
}
if glm_mode.startswith("Quick"):
    if all(path.is_file() for path in precomputed_files.values()):
        try:
            quick_glm = pd.read_csv(precomputed_files["GLM coefficients"])
            quick_left = pd.read_csv(precomputed_files["Left-hand tapping contrast"])
            quick_right = pd.read_csv(precomputed_files["Right-hand tapping contrast"])
            required = {"Condition", "Chroma", "ch_name", "theta"}
            if not required.issubset(quick_glm.columns):
                raise ValueError("The saved GLM results are missing required columns.")
            st.success("Precomputed GLM results loaded — no fitting required.")
            st.caption("Results from a previously completed run of the same Participant-1 pipeline.")
            quick_hbo = quick_glm.loc[
                (quick_glm["Chroma"] == "hbo")
                & quick_glm["Condition"].isin(["Tapping/Left", "Tapping/Right"])
            ].copy()
            if not quick_hbo.empty:
                fig_quick = px.bar(
                    quick_hbo, x="ch_name", y="theta", color="Condition",
                    barmode="group", title="Precomputed HbO GLM: Left vs Right Tapping",
                    labels={"ch_name": "fNIRS channel", "theta": "HbO coefficient (mol/L)"},
                )
                st.plotly_chart(fig_quick, width="stretch")
            selected_quick = st.selectbox(
                "Explore a precomputed contrast", ["Tapping/Left", "Tapping/Right"],
                key="precomputed_contrast_choice",
            )
            selected_df = quick_left if selected_quick == "Tapping/Left" else quick_right
            st.dataframe(selected_df, hide_index=True)
            st.caption("These are channel-level GLM estimates; they do not by themselves establish "
                       "statistically significant cortical activation.")
        except Exception as exc:
            st.error(f"Could not read the saved GLM results: {exc}")
    else:
        st.info("Precomputed results have not been added to this repository yet. "
                "Select Full analysis, run the GLM once, then download and upload "
                "the three CSV files to the precomputed_glm folder on GitHub.")
else:
    st.info("Full analysis is enabled below. The GLM can take several minutes.")


# Lesson 2: Load the example fNIRS dataset

import mne

st.divider()
st.subheader("Load example fNIRS dataset")

st.write(
    "Click the button below to load Participant 1 "
    "from the MNE finger-tapping experiment."
)


# Remember the dataset between interactions
if "raw" not in st.session_state:
    st.session_state.raw = None

if st.button("Load finger-tapping data"):
    with st.spinner("Loading fNIRS recordings..."):
        data_path = mne.datasets.fnirs_motor.data_path()
        st.session_state.raw = mne.io.read_raw_nirx(
            data_path / "Participant-1",
            preload=True
        )

# Display the viewer whenever data are available
if st.session_state.raw is not None:
    raw = st.session_state.raw

    st.success("Finger-tapping dataset loaded successfully!")

    st.write("Number of channels:", len(raw.ch_names))
    st.write("Sampling frequency:", raw.info["sfreq"], "Hz")
    st.write("Recording duration:", round(raw.times[-1], 1), "seconds")

    # Interactive raw-signal viewer
    import plotly.express as px
    import pandas as pd

    st.subheader("Raw fNIRS Signal Viewer")

    channel = st.selectbox(
        "Select an fNIRS channel",
        raw.ch_names
    )

    start_time = st.slider(
        "Start time (seconds)",
        min_value=0,
        max_value=int(raw.times[-1]) - 30,
        value=0,
        step=10
    )

    # Extract 30 seconds from the selected channel
    data, times = raw.copy().pick([channel]).get_data(
        start=int(start_time * raw.info["sfreq"]),
        stop=int((start_time + 30) * raw.info["sfreq"]),
        return_times=True
    )

    signal_df = pd.DataFrame({
        "Time (seconds)": times,
        "Light intensity": data[0]
    })

    fig = px.line(
        signal_df,
        x="Time (seconds)",
        y="Light intensity",
        title=f"Raw signal: {channel}"
    )

    st.plotly_chart(fig, width="stretch")

    # Lesson 4: Optical density and signal quality

    st.divider()
    st.subheader("Signal Quality Assessment")

    if st.button("Calculate Scalp Coupling Index"):
        with st.spinner("Calculating signal quality..."):
            raw_od = mne.preprocessing.nirs.optical_density(raw)
            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

        threshold = 0.5
        poor_channels = sum(sci < threshold)

        st.success("Signal quality assessment completed!")

        st.write("Total wavelength channels:", len(sci))
        st.write("SCI threshold:", threshold)
        st.write("Channels below threshold:", int(poor_channels))

        # Display SCI values for individual channels
        sci_df = pd.DataFrame({
            "Channel": raw_od.ch_names,
            "SCI": sci
        })

        fig_sci = px.bar(
            sci_df,
            x="Channel",
            y="SCI",
            title="Scalp Coupling Index by Channel"
        )

        fig_sci.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text="SCI threshold"
        )

        st.plotly_chart(fig_sci, width="stretch")


# Lesson 5: Optical density and motion correction

st.divider()
st.subheader("Motion Artefact Correction (TDDR)")

st.write(
    "The first step is to convert the recorded light intensity "
    "into optical density before correcting motion artefacts."
)


if "raw" in st.session_state and st.session_state["raw"] is not None and st.button("Convert to Optical Density"):
    with st.spinner("Converting to optical density..."):
        st.session_state["raw_od"] = (
            mne.preprocessing.nirs.optical_density(st.session_state["raw"])
        )

if "raw_od" in st.session_state:
    st.success("Optical density conversion completed!")
    st.write(
        "Number of optical density channels:",
        len(st.session_state["raw_od"].ch_names)
    )

# Apply TDDR motion correction

if "raw_od" in st.session_state:

    if st.button("Apply TDDR Motion Correction"):
        with st.spinner("Correcting motion artefacts..."):
            st.session_state["raw_od_corrected"] = (
                mne.preprocessing.nirs.temporal_derivative_distribution_repair(
                    st.session_state["raw_od"]
                )
            )

    if "raw_od_corrected" in st.session_state:
        st.success("TDDR motion correction completed!")
        st.write(
            "Number of corrected channels:",
            len(st.session_state["raw_od_corrected"].ch_names)
        )

# Lesson 5: Compare signals before and after TDDR

if "raw_od_corrected" in st.session_state:

    st.subheader("Before and After TDDR Comparison")    


    raw_od = st.session_state["raw_od"]    
    corrected = st.session_state["raw_od_corrected"]
    tddr_start = st.slider(
        "TDDR comparison start time (seconds)",
        min_value=0,
        max_value=int(raw_od.times[-1]) - 30,
        value=0,
        step=10,
        key="tddr_time_slider"
    )
    

    comparison_channel = st.selectbox(
        "Select a channel for comparison",
        raw_od.ch_names,
        key="tddr_comparison_channel"
    )

    channel_index = raw_od.ch_names.index(comparison_channel)

   # Display the selected 30-second time window
    times = raw_od.times

    mask = (times >= tddr_start) & (times < tddr_start + 30)

    before = raw_od.get_data()[channel_index, mask]
    after = corrected.get_data()[channel_index, mask]

    comparison_df = pd.DataFrame({
        "Time (seconds)": times[mask],
        "Before TDDR": before,
        "After TDDR": after
    })

    fig_comparison = px.line(
        comparison_df,
        x="Time (seconds)",
        y=["Before TDDR", "After TDDR"],
        title=f"TDDR Comparison: {comparison_channel}"
    )

    fig_comparison.update_layout(
        yaxis_title="Optical density",
        legend_title="Signal"
    )

    st.plotly_chart(fig_comparison, width="stretch")

# Stage 6: Haemoglobin conversion

st.divider()
st.subheader("Haemoglobin Conversion (HbO and HbR)")

st.write(
    "Convert motion-corrected optical density into "
    "oxygenated and deoxygenated haemoglobin concentration changes."
)

if "raw_od_corrected" in st.session_state:

    if st.button("Convert to HbO and HbR"):

        with st.spinner("Converting haemoglobin signals..."):

            st.session_state["raw_haemo"] = (
                mne.preprocessing.nirs.beer_lambert_law(
                    st.session_state["raw_od_corrected"],
                    ppf=6.0
                )
            )

    if "raw_haemo" in st.session_state:

        st.success("Haemoglobin conversion completed!")

        raw_haemo = st.session_state["raw_haemo"]

        st.write(
            "Number of haemoglobin channels:",
            len(raw_haemo.ch_names)
        )

        st.write(
            "HbO channels:",
            len(mne.pick_types(raw_haemo.info, fnirs="hbo"))
        )

        st.write(
            "HbR channels:",
            len(mne.pick_types(raw_haemo.info, fnirs="hbr"))
        )

# Stage 7: Interactive haemoglobin signal viewer

if "raw_haemo" in st.session_state:

    st.divider()
    st.subheader("HbO and HbR Signal Viewer")

    raw_haemo = st.session_state["raw_haemo"]

    # Select a source-detector pair
    hbo_channels = [
        ch for ch in raw_haemo.ch_names
        if ch.endswith("hbo")
    ]

    selected_hbo = st.selectbox(
        "Select a channel",
        hbo_channels
    )

    selected_hbr = selected_hbo.replace("hbo", "hbr")

    # Select the starting time
    start_time = st.slider(
        "Start time (seconds)",
        min_value=0,
        max_value=max(0, int(raw_haemo.times[-1]) - 30),
        value=0,
        step=10,
        key="hbo_time_slider"
    )

    # Extract a 30-second window
    times = raw_haemo.times

    mask = (times >= start_time) & (times <= start_time + 30)

    hbo_index = raw_haemo.ch_names.index(selected_hbo)
    hbr_index = raw_haemo.ch_names.index(selected_hbr)

    hbo = raw_haemo.get_data()[hbo_index, mask] * 1e6
    hbr = raw_haemo.get_data()[hbr_index, mask] * 1e6

    # Prepare the data for plotting
    signal_df = pd.DataFrame({
        "Time (seconds)": times[mask],
        "HbO": hbo,
        "HbR": hbr
    })

    # Create an interactive plot
    fig_hb = px.line(
        signal_df,
        x="Time (seconds)",
        y=["HbO", "HbR"],
        title=f"Haemoglobin signals: {selected_hbo}"
    )

    fig_hb.update_layout(
        yaxis_title="Concentration change (µM)",
        legend_title="Haemoglobin"
    )

    st.plotly_chart(fig_hb, width="stretch")

# Stage 8: Experimental events and task markers

if "raw_haemo" in st.session_state:

    st.divider()
    st.subheader("Experimental Events and Task Markers")

    st.write(
        "Explore the experimental conditions recorded "
        "during the finger-tapping task."
    )

    raw_haemo = st.session_state["raw_haemo"]

    # Read the event annotations
    annotations = raw_haemo.annotations

    event_df = pd.DataFrame({
        "Time (seconds)": annotations.onset,
        "Duration (seconds)": annotations.duration,
        "Event": annotations.description
    })

    # Rename task markers for students
    event_names = {
        "1.0": "Control",
        "2.0": "Tapping/Left",
        "3.0": "Tapping/Right"
    }

    event_df["Event"] = event_df["Event"].replace(event_names)

    # Keep the three experimental conditions
    task_events = event_df[
        event_df["Event"].isin([
            "Control",
            "Tapping/Left",
            "Tapping/Right"
        ])
    ].copy()

    # Count events in each condition
    event_counts = (
        task_events["Event"]
        .value_counts()
        .reindex([
            "Control",
            "Tapping/Left",
            "Tapping/Right"
        ], fill_value=0)
        .reset_index()
    )

    event_counts.columns = ["Condition", "Number of events"]

    st.write("Number of events per condition")

    st.dataframe(event_counts, hide_index=True)

    # Display the first ten event markers
    st.write("First 10 experimental events")

    st.dataframe(
        task_events.head(10),
        hide_index=True
    )

# Stage 8.2: Interactive event timeline

if "raw_haemo" in st.session_state:

    st.subheader("Experimental Event Timeline")

    raw_haemo = st.session_state["raw_haemo"]

    # Read the recorded event markers
    event_df = pd.DataFrame({
        "Time (seconds)": raw_haemo.annotations.onset,
        "Event": raw_haemo.annotations.description
    })

    # Give the task markers meaningful names
    event_df["Event"] = event_df["Event"].replace({
        "1.0": "Control",
        "2.0": "Tapping/Left",
        "3.0": "Tapping/Right"
    })

    # Select only the three task conditions
    event_df = event_df[
        event_df["Event"].isin([
            "Control",
            "Tapping/Left",
            "Tapping/Right"
        ])
    ]

    # Plot each event at its recorded onset
    fig_events = px.scatter(
        event_df,
        x="Time (seconds)",
        y="Event",
        color="Event",
        title="Finger-Tapping Experimental Events"
    )

    fig_events.update_traces(marker_size=9)

    fig_events.update_layout(
        xaxis_title="Recording time (seconds)",
        yaxis_title="Experimental condition"
    )

    st.plotly_chart(fig_events, width="stretch")

# Stage 9.1: Create event-related epochs

if "raw_haemo" in st.session_state:

    st.divider()
    st.subheader("Event-Related Epochs")

    st.write(
        "Extract haemoglobin signals around each "
        "experimental event."
    )

    if st.button("Create Epochs"):

        raw_haemo = st.session_state["raw_haemo"].copy()

        # Rename the event markers
        raw_haemo.annotations.rename({
            "1.0": "Control",
            "2.0": "Tapping/Left",
            "3.0": "Tapping/Right"
        })

        # Identify the three experimental conditions
        event_id = {
            "Control": 1,
            "Tapping/Left": 2,
            "Tapping/Right": 3
        }

        events, event_id = mne.events_from_annotations(
            raw_haemo,
            event_id=event_id
        )

        # Extract epochs around each event
        epochs = mne.Epochs(
            raw_haemo,
            events,
            event_id=event_id,
            tmin=-5,
            tmax=20,
            baseline=(-5, 0),
            preload=True
        )

        # Save the epochs for later analysis
        st.session_state["epochs"] = epochs

        st.success("Epoch creation completed!")

        st.write(
            "Total epochs:",
            len(epochs)
        )

        for condition in event_id:
            st.write(
                condition,
                len(epochs[condition])
            )

# Stage 9.2: Average haemodynamic responses

if "epochs" in st.session_state:

    st.divider()
    st.subheader("Average Haemodynamic Responses")

    st.write(
        "Compare average HbO and HbR responses "
        "across experimental conditions."
    )

    epochs = st.session_state["epochs"]

    condition = st.selectbox(
        "Select experimental condition",
        ["Control", "Tapping/Left", "Tapping/Right"],
        key="average_condition"
    )

    # Calculate the average response
    evoked = epochs[condition].average()

    # Select source-detector pairs
    hbo_channels = [
        ch for ch in evoked.ch_names
        if ch.endswith("hbo")
    ]

    selected_hbo = st.selectbox(
        "Select channel",
        hbo_channels,
        key="average_channel"
    )

    # Find the corresponding HbR channel
    selected_hbr = selected_hbo.replace("hbo", "hbr")

    hbo_index = evoked.ch_names.index(selected_hbo)
    hbr_index = evoked.ch_names.index(selected_hbr)

    # Convert haemoglobin concentrations to micromolar
    hbo = evoked.data[hbo_index] * 1e6
    hbr = evoked.data[hbr_index] * 1e6

    average_df = pd.DataFrame({
        "Time (seconds)": evoked.times,
        "HbO": hbo,
        "HbR": hbr
    })

    fig_average = px.line(
        average_df,
        x="Time (seconds)",
        y=["HbO", "HbR"],
        title=f"Average Response: {condition}"
    )

    fig_average.add_vline(
        x=0,
        line_dash="dash"
    )

    fig_average.update_layout(
        yaxis_title="Concentration change (µM)",
        legend_title="Haemoglobin"
    )

    st.plotly_chart(fig_average, width="stretch")

# Stage 9.3: Compare left- and right-hand tapping

if "epochs" in st.session_state:

    st.divider()
    st.subheader("Left vs Right Finger-Tapping Comparison")

    st.write(
        "Compare average HbO responses during "
        "left- and right-hand finger tapping."
    )

    epochs = st.session_state["epochs"]

    # Select an HbO channel
    hbo_channels = [
        ch for ch in epochs.ch_names
        if ch.endswith("hbo")
    ]

    selected_channel = st.selectbox(
        "Select HbO channel for comparison",
        hbo_channels,
        key="left_right_channel"
    )

    # Average the epochs for each condition
    left_evoked = epochs["Tapping/Left"].average()
    right_evoked = epochs["Tapping/Right"].average()

    # Find the selected channel
    channel_index = left_evoked.ch_names.index(selected_channel)

    # Extract HbO responses and convert to micromolar
    left_hbo = left_evoked.data[channel_index] * 1e6
    right_hbo = right_evoked.data[channel_index] * 1e6

    # Prepare data for plotting
    comparison_df = pd.DataFrame({
        "Time (seconds)": left_evoked.times,
        "Tapping/Left": left_hbo,
        "Tapping/Right": right_hbo
    })

    # Create comparison graph
    fig_lr = px.line(
        comparison_df,
        x="Time (seconds)",
        y=["Tapping/Left", "Tapping/Right"],
        title=f"Left vs Right Tapping: {selected_channel}"
    )

    # Mark task onset
    fig_lr.add_vline(
        x=0,
        line_dash="dash"
    )

    fig_lr.update_layout(
        yaxis_title="HbO concentration change (µM)",
        legend_title="Condition"
    )

    st.plotly_chart(fig_lr, width="stretch")

# Stage 10.1: GLM Design Matrix

from mne_nirs.experimental_design import make_first_level_design_matrix

if "raw_haemo" in st.session_state:

    st.divider()
    st.header("General Linear Model (GLM)")

    st.subheader("GLM Design Matrix")

    st.write(
        "The design matrix represents the experimental "
        "conditions and their predicted haemodynamic responses."
    )

    raw_haemo = st.session_state["raw_haemo"]
    # Use meaningful experimental condition names
    raw_haemo = raw_haemo.copy()

    raw_haemo.annotations.rename({
    "1.0": "Control",
    "2.0": "Tapping/Left",
    "3.0": "Tapping/Right"
})

    # Retain only the three experimental conditions
    keep = [
    description in ["Control", "Tapping/Left", "Tapping/Right"]
    for description in raw_haemo.annotations.description
]

    raw_haemo.set_annotations(raw_haemo.annotations[keep])


    # Create the GLM design matrix
    design_matrix = make_first_level_design_matrix(
        raw_haemo,
        drift_model="cosine",
        high_pass=0.01,
        hrf_model="spm"
    )

    # Save for the next stage
    st.session_state["design_matrix"] = design_matrix
    st.session_state["glm_raw"] = raw_haemo

    st.write("Design matrix created successfully!")

    # Display the first rows
    st.dataframe(design_matrix.head())

    # Visualise the design matrix
    fig_design = px.imshow(
        design_matrix.T,
        aspect="auto",
        labels={
            "x": "Time samples",
            "y": "Model regressors",
            "color": "Value"
        },
        title="Finger-Tapping GLM Design Matrix"
    )

    st.plotly_chart(fig_design, width="stretch")

# Stage 10.2: Fit the General Linear Model

from mne_nirs.statistics import run_glm

if (
    "raw_haemo" in st.session_state
    and "design_matrix" in st.session_state
):

    st.subheader("Fit the GLM")

    st.write(
        "Estimate haemodynamic responses for the "
        "experimental conditions using the GLM."
    )

    if st.button("Run GLM Analysis"):

        with st.spinner("Fitting the GLM..."):

            glm_est = run_glm(
                st.session_state["glm_raw"],
                st.session_state["design_matrix"]
            )

            st.session_state["glm_est"] = glm_est

        st.success("GLM analysis completed!")

    if "glm_est" in st.session_state:

        st.write("GLM results are available for the next stage.")
        st.caption("To enable the quick demonstration for everyone, export these real "
                   "results and upload them to GitHub under precomputed_glm/.")
        st.download_button(
            "Download glm_results.csv",
            st.session_state["glm_est"].to_dataframe().to_csv(index=False).encode("utf-8"),
            file_name="glm_results.csv", mime="text/csv",
            key="export_glm_results",
        )

# Stage 10.3: View GLM Results

if "glm_est" in st.session_state:

    st.divider()
    st.subheader("GLM Results")

    st.write(
        "Explore the estimated haemodynamic responses "
        "for each experimental condition and channel."
    )

    # Retrieve the fitted GLM
    glm_est = st.session_state["glm_est"]

    # Convert GLM results into a table
    glm_results = glm_est.to_dataframe()

    # Keep only the three experimental conditions
    conditions = [
        "Control",
        "Tapping/Left",
        "Tapping/Right"
    ]

    filtered_results = glm_results[
        glm_results["Condition"].isin(conditions)
    ]

    # Display the experimental results
    st.write("GLM Results: Experimental Conditions")
    st.dataframe(filtered_results)

    # Display available model regressors
    st.write("Experimental conditions:")

    st.write([
        "Control",
        "Tapping/Left",
        "Tapping/Right"
    ])


# Stage 10.4: Interactive GLM Comparison

if "glm_est" in st.session_state:

    st.divider()
    st.subheader("Left vs Right Finger-Tapping GLM Comparison")

    st.write(
        "Compare GLM-estimated HbO responses "
        "during left- and right-hand finger tapping."
    )

    # Retrieve the GLM results
    glm_est = st.session_state["glm_est"]
    glm_df = glm_est.to_dataframe()

    # Select HbO channels and tapping conditions
    comparison_df = glm_df[
        (glm_df["Chroma"] == "hbo")
        & (glm_df["Condition"].isin(
            ["Tapping/Left", "Tapping/Right"]
        ))
    ].copy()

    # Create an interactive bar chart
    fig_glm = px.bar(
        comparison_df,
        x="ch_name",
        y="theta",
        color="Condition",
        barmode="group",
        title="GLM HbO Estimates: Left vs Right Tapping",
        labels={
            "ch_name": "fNIRS channel",
            "theta": "HbO GLM coefficient (mol/L)",
            "Condition": "Experimental condition"
        }
    )

    st.plotly_chart(fig_glm, width="stretch")

# Stage 11: 3D Brain Visualisation

if "glm_est" in st.session_state:

    st.divider()
    st.header("3D Brain Visualisation")

    st.write(
        "Explore haemodynamic responses during "
        "left- and right-hand finger tapping "
        "on a standard brain template."
    )

    st.success("GLM results are available for brain visualisation.")

# Stage 11.3: Prepare GLM results for brain mapping

if "glm_est" in st.session_state:

    glm_est = st.session_state["glm_est"]
    design_matrix = st.session_state["design_matrix"]

    st.subheader("Prepare Brain Mapping Data")

    st.write(
        "The GLM results will be used to calculate "
        "left- and right-hand finger-tapping contrasts."
    )

    st.write("Available GLM regressors:")
    st.write(design_matrix.columns.tolist())

    # Create contrast vectors for left and right tapping
    left_vector = [
        1 if col == "Tapping/Left" else 0
        for col in design_matrix.columns
    ]

    right_vector = [
        1 if col == "Tapping/Right" else 0
        for col in design_matrix.columns
    ]

    # Calculate GLM contrasts
    left_contrast = glm_est.compute_contrast(
        left_vector, contrast_type="t"
    )

    right_contrast = glm_est.compute_contrast(
        right_vector, contrast_type="t"
    )

    # Convert contrasts into tables
    left_contrast_df = left_contrast.to_dataframe()
    right_contrast_df = right_contrast.to_dataframe()

    # Store results for brain visualisation
    st.session_state["left_contrast_df"] = left_contrast_df
    st.session_state["right_contrast_df"] = right_contrast_df

    st.success("Left and right tapping contrasts prepared!")
    st.download_button(
        "Download left_contrast.csv",
        left_contrast_df.to_csv(index=False).encode("utf-8"),
        file_name="left_contrast.csv", mime="text/csv", key="export_left_contrast",
    )
    st.download_button(
        "Download right_contrast.csv",
        right_contrast_df.to_csv(index=False).encode("utf-8"),
        file_name="right_contrast.csv", mime="text/csv", key="export_right_contrast",
    )


# Stage 11.5: Interactive HbO brain maps
# These are descriptive, template-based GLM effects, not statistical activation.

import numpy as np
from pathlib import Path
import nibabel as nib
import plotly.graph_objects as go
from mne.io import read_fiducials
from mne.coreg import fit_matched_points
from mne.transforms import Transform, apply_trans


@st.cache_data(show_spinner=False)
def load_surface(surface_file, decimate_step=5):
    """Load a template hemisphere, reducing mesh size for browser performance."""
    vertices, faces = nib.freesurfer.read_geometry(str(surface_file))
    # Subsample triangles rather than vertices; all referenced vertices remain valid.
    return vertices, faces[::decimate_step]


def get_initial_head_mri_transform(info, fiducial_file):
    """Fiducial-only rigid alignment; approximate, not validated coregistration."""
    fid_mri, _ = read_fiducials(str(fiducial_file))
    fid_head = [d for d in (info.get("dig") or [])
                if d["kind"] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL]
    ids = [mne.io.constants.FIFF.FIFFV_POINT_LPA,
           mne.io.constants.FIFF.FIFFV_POINT_NASION,
           mne.io.constants.FIFF.FIFFV_POINT_RPA]
    src = np.array([next(d["r"] for d in fid_head if d["ident"] == ident)
                    for ident in ids])
    dst = np.array([next(d["r"] for d in fid_mri if d["ident"] == ident)
                    for ident in ids])
    matrix = fit_matched_points(src_pts=src, tgt_pts=dst, out="trans")
    trans = Transform("head", "mri", matrix)
    err = np.linalg.norm(apply_trans(trans, src) - dst, axis=1) * 1000
    return trans, err


def make_hbo_map(info, contrast_df, subjects_dir, trans, hemi="lh", spread_mm=15):
    """Interpolate HbO channel effects around transformed channel midpoints."""
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist

    surf_file = subjects_dir / "fsaverage" / "surf" / f"{hemi}.pial"
    vertices, faces = load_surface(surf_file)
    picks = mne.pick_types(info, fnirs="hbo")
    channel_names = [info["ch_names"][i] for i in picks]
    positions = np.array([info["chs"][i]["loc"][:3] for i in picks])
    # Map contrast values by channel name, not by arbitrary dataframe order.
    hbo = contrast_df[contrast_df["Chroma"] == "hbo"].set_index("ch_name")
    missing = [name for name in channel_names if name not in hbo.index]
    if missing:
        raise ValueError(f"Missing HbO GLM estimates for channels: {missing[:4]}")
    effects = np.array([float(hbo.loc[name, "effect"]) for name in channel_names]) * 1e6
    positions_mm = apply_trans(trans, positions) * 1000
    if not np.isfinite(positions_mm).all() or not np.isfinite(effects).all():
        raise ValueError("Non-finite channel positions or GLM estimates")

    # Project positions onto nearest surface vertex. This is only an approximation.
    tree = cKDTree(vertices)
    _, nearest = tree.query(positions_mm)
    projected = vertices[nearest]
    # Limit Gaussian weighting to channels within ~3 standard deviations.
    distances = cdist(vertices, projected)
    weights = np.exp(-0.5 * (distances / spread_mm) ** 2)
    weights[distances > 3 * spread_mm] = 0
    denom = weights.sum(axis=1)
    values = np.divide(weights @ effects, denom, out=np.zeros(len(vertices)), where=denom > 1e-5)
    coverage = denom > 0.03
    return vertices, faces, values, coverage, projected, effects, channel_names


st.divider()
st.header("Interactive 3D HbO Brain Maps")
st.caption("Finger-tapping GLM effect estimates (µM), projected onto the fsaverage template. "
           "Exploratory visualisation only: not a statistical significance or individual anatomy map.")

subjects_dir = Path.home() / "mne_data" / "MNE-fsaverage-data"
fiducial_file = subjects_dir / "fsaverage" / "bem" / "fsaverage-fiducials.fif"
ready = all(k in st.session_state for k in
            ("raw_haemo", "left_contrast_df", "right_contrast_df"))

if not ready:
    st.info("To display brain maps: load data → convert to optical density → apply TDDR → "
            "convert to HbO/HbR → run the GLM. The maps will then appear here.")
elif not fiducial_file.exists():
    st.error(f"fsaverage fiducials were not found: {fiducial_file}")
else:
    st.success("HbO data and left/right GLM contrasts are ready.")
    condition = st.radio("Finger-tapping condition", ["Tapping/Left", "Tapping/Right"],
                         horizontal=True, key="brain_map_condition")
    hemisphere = st.radio("Hemisphere", ["Left", "Right"], horizontal=True,
                          key="brain_map_hemisphere")
    spread_mm = st.slider("Illustrative spatial smoothing (mm)", 8, 25, 15, 1)
    contrast = st.session_state["left_contrast_df" if condition == "Tapping/Left"
                                else "right_contrast_df"]
    info = st.session_state["raw_haemo"].info
    hemi = "lh" if hemisphere == "Left" else "rh"
    try:
        trans, errors = get_initial_head_mri_transform(info, fiducial_file)
        st.caption("Fiducial-only alignment; anatomical coverage has not been independently validated. "
                   "A low fiducial fit error does not establish accurate optode localisation.")
        vertices, faces, values, coverage, projected, effects, names = make_hbo_map(
            info, contrast, subjects_dir, trans, hemi=hemi, spread_mm=spread_mm)
        # Neutral grey outside estimated channel coverage, coloured within.
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="lightgray", opacity=0.25, name="Template cortex", hoverinfo="skip"))
        # Face intensity is derived from vertex intensities; masked outside coverage.
        face_values = values[faces].mean(axis=1)
        face_covered = coverage[faces].any(axis=1)
        coloured_faces = faces[face_covered]
        coloured_intensity = face_values[face_covered]
        if len(coloured_faces):
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=coloured_faces[:, 0], j=coloured_faces[:, 1], k=coloured_faces[:, 2],
                intensity=values, intensitymode="vertex",
                colorscale="RdBu_r", cmin=-1.3, cmax=1.3,
                colorbar=dict(title="HbO (µM)"), opacity=1,
                name="HbO GLM effect", hoverinfo="skip",
                lighting=dict(ambient=0.55, diffuse=0.7)))
        # Show projected channel midpoints to communicate sparse spatial sampling.
        side = projected[:, 0] < 0 if hemi == "lh" else projected[:, 0] >= 0
        fig.add_trace(go.Scatter3d(
            x=projected[side, 0], y=projected[side, 1], z=projected[side, 2],
            mode="markers", marker=dict(size=3, color="black"),
            text=[f"{names[i]}: {effects[i]:.3f} µM" for i in np.where(side)[0]],
            name="Projected channels", hovertemplate="%{text}<extra></extra>"))
        fig.update_layout(
            title=f"{condition} — {hemisphere} hemisphere (descriptive HbO)",
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                       aspectmode="data"),
            margin=dict(l=0, r=0, t=55, b=0), height=630)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rotate: drag | Zoom: scroll | Pan: right-drag. "
                   "Grey = outside approximate channel coverage. "
                   "Colour scale fixed at −1.3 to +1.3 µM for condition comparisons.")
        with st.expander("View channel-level HbO estimates"):
            st.dataframe(pd.DataFrame({"Channel": names, "GLM HbO effect (µM)": effects}),
                         hide_index=True)
            st.write("Initial fiducial fitting errors (mm):", np.round(errors, 3).tolist())
    except Exception as exc:
        st.error(f"Could not create the 3D map: {exc}")
        st.info("The earlier analysis stages remain available. Please share this error if it appears.")
