import streamlit as st
import pandas as pd
import joblib
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="InfraVision AI",
    page_icon="🚧",
    layout="wide"
)


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


# Title
st.title("🚧 InfraVision AI")
st.subheader("AI-Powered Road Risk & Maintenance Priority System")

st.write(
    "Upload one or multiple road images. "
    "InfraVision AI will predict the road risk and rank roads "
    "by maintenance priority."
)

st.divider()


# Upload road images
uploaded_files = st.file_uploader(
    "📤 Upload Road Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# Road information
st.subheader("🛣️ Road Information")

col1, col2 = st.columns(2)

with col1:
    road_age = st.number_input(
        "Road Age (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

    traffic = st.selectbox(
        "Traffic Level",
        ["Low", "Medium", "High"]
    )

with col2:
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0,
        max_value=2000,
        value=500
    )

    previous_repairs = st.number_input(
        "Previous Repairs",
        min_value=0,
        max_value=20,
        value=1
    )


damage = st.selectbox(
    "Visible Damage Severity",
    ["Minor", "Moderate", "High", "Severe"]
)


# Convert values to numbers
traffic_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

damage_map = {
    "Minor": 1,
    "Moderate": 2,
    "High": 3,
    "Severe": 4
}


# Analyze button
if st.button("🚀 Analyze Infrastructure"):

    if not uploaded_files:
        st.warning("Please upload at least one road image.")

    else:
        results = []

        for uploaded_file in uploaded_files:

            # Create ML input
            input_data = pd.DataFrame({
                "road_age": [road_age],
                "traffic_level": [traffic_map[traffic]],
                "rainfall": [rainfall],
                "previous_repairs": [previous_repairs],
                "damage_severity": [damage_map[damage]]
            })

            # Predict risk
            risk_score = float(model.predict(input_data)[0])

            # Keep score between 0 and 100
            risk_score = max(0, min(100, risk_score))

            # Priority and recommendation
            if risk_score >= 80:
                priority = "Critical"
                recommendation = "Immediate repair recommended."

            elif risk_score >= 60:
                priority = "High"
                recommendation = "Schedule repair soon."

            elif risk_score >= 40:
                priority = "Medium"
                recommendation = "Monitor and plan maintenance."

            else:
                priority = "Low"
                recommendation = "Regular monitoring recommended."

            # Store result
            results.append({
                "Image": uploaded_file.name,
                "Risk Score": round(risk_score, 2),
                "Priority": priority,
                "Recommendation": recommendation
            })


        # Create results table
        results_df = pd.DataFrame(results)

        # Sort by highest risk
        results_df = results_df.sort_values(
            by="Risk Score",
            ascending=False
        )

        # Start ranking from 1
        results_df.index = range(1, len(results_df) + 1)

        st.success("Analysis completed successfully!")


        # Dashboard
        st.subheader("📊 Infrastructure Analysis Dashboard")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Roads Analyzed",
            len(results_df)
        )

        metric2.metric(
            "Average Risk",
            f"{results_df['Risk Score'].mean():.1f}%"
        )

        metric3.metric(
            "Highest Risk",
            f"{results_df['Risk Score'].max():.1f}%"
        )


        # Results table
        st.subheader("🏆 Maintenance Priority Ranking")

        st.dataframe(
            results_df,
            use_container_width=True
        )


        # Display uploaded images
        st.subheader("📷 Uploaded Road Images")

        image_columns = st.columns(3)

        for index, uploaded_file in enumerate(uploaded_files):

            image = Image.open(uploaded_file)

            with image_columns[index % 3]:
                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )