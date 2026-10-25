# import streamlit as st
# from src.predict import predict_machine

# st.set_page_config(
#     page_title="Predictive Maintenance",
#     page_icon="⚙️"
# )

# st.title("⚙️ Predictive Maintenance System")

# st.write(
#     "Predict machine failure probability and remaining useful life."
# )

# air = st.number_input(
#     "Air Temperature (K)",
#     value=300.0
# )

# process = st.number_input(
#     "Process Temperature (K)",
#     value=310.0
# )

# rpm = st.number_input(
#     "Rotational Speed (rpm)",
#     value=1500
# )

# torque = st.number_input(
#     "Torque (Nm)",
#     value=40.0
# )

# wear = st.number_input(
#     "Tool Wear (min)",
#     value=50
# )

# if st.button("Predict"):

#     prob, rul = predict_machine(
#         air,
#         process,
#         rpm,
#         torque,
#         wear
#     )

#     st.subheader("Results")

#     st.metric(
#         "Failure Probability",
#         f"{prob*100:.2f}%"
#     )

#     st.metric(
#         "Remaining Useful Life",
#         f"{rul:.2f} min"
#     )

#     if prob > 0.7:
#         st.error("⚠️ High Failure Risk")
#     elif prob > 0.3:
#         st.warning("⚠️ Medium Risk")
#     else:
#         st.success("✅ Machine Healthy")
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Predictive Maintenance system   ",
    page_icon="⚙️",
    layout="wide"
)

# ==========================
# Custom CSS
# ==========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1,h2,h3 {
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================

failure_model = joblib.load("models/failure_model.pkl")
rul_model = joblib.load("models/rul_model.pkl")

# ==========================
# Title
# ==========================

st.markdown("""
<h1 style='text-align:center; color:#00D4FF;'>
⚙️ Predictive Maintenance Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================
# Sidebar
# ==========================

st.sidebar.header("Machine Inputs")

air_temp = st.sidebar.slider(
    "Air Temperature (K)",
    280,
    330,
    300
)

process_temp = st.sidebar.slider(
    "Process Temperature (K)",
    290,
    350,
    310
)

rpm = st.sidebar.slider(
    "Rotational Speed (rpm)",
    1000,
    3000,
    1500
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    0,
    100,
    40
)

tool_wear = st.sidebar.slider(
    "Tool Wear (min)",
    0,
    300,
    50
)

# ==========================
# Prediction Button
# ==========================

if st.sidebar.button("Predict"):

    sample = np.array([[
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]])

    failure_prob = failure_model.predict_proba(sample)[0][1]

    rul = rul_model.predict(sample)[0]

    health_score = max(
        0,
        round((1 - failure_prob) * 100, 2)
    )

    # ==========================
    # Metrics
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Machine Health Score",
            f"{health_score}%"
        )

    with col2:
        st.metric(
            "Failure Probability",
            f"{failure_prob*100:.2f}%"
        )

    with col3:
        st.metric(
            "Remaining Useful Life",
            f"{rul:.2f} min"
        )

    st.markdown("---")

    # ==========================
    # Gauge Chart
    # ==========================

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=failure_prob * 100,
            title={"text": "Failure Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ]
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ==========================
    # Feature Importance
    # ==========================

    features = [
        "Air Temp",
        "Process Temp",
        "RPM",
        "Torque",
        "Tool Wear"
    ]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": failure_model.feature_importances_
    })

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================
    # Recommendation
    # ==========================

    st.subheader("Maintenance Recommendation")

    if failure_prob > 0.7:

        st.error(
            "🚨 HIGH RISK: Immediate inspection required."
        )

    elif failure_prob > 0.3:

        st.warning(
            "⚠️ MEDIUM RISK: Schedule maintenance soon."
        )

    else:

        st.success(
            "✅ LOW RISK: Machine operating normally."
        )

    # ==========================
    # Input Summary
    # ==========================

    st.subheader("Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Air Temperature",
            "Process Temperature",
            "RPM",
            "Torque",
            "Tool Wear"
        ],
        "Value": [
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True
    )

else:

    st.info(
        "Enter machine parameters from the sidebar and click Predict."
    )