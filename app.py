import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from PIL import Image


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------
st.set_page_config(
    page_title="InfraVision AI",
    page_icon="🚧",
    layout="wide"
)


# --------------------------------
# LOAD ML MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


# --------------------------------
# IMAGE DAMAGE ANALYSIS
# --------------------------------
def analyze_road_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Resize image for faster processing
    gray = cv2.resize(gray, (400, 300))

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Calculate edge density
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size

    edge_density = (edge_pixels / total_pixels) * 100

    # Estimate visual damage
    if edge_density < 3:
        damage_level = "Minor"
        damage_value = 1
    elif edge_density < 6:
        damage_level = "Moderate"
        damage_value = 2
    elif edge_density < 10:
        damage_level = "High"
        damage_value = 3
    else:
        damage_level = "Severe"
        damage_value = 4

    return image, edge_density, damage_level, damage_value


# --------------------------------
# RISK EXPLANATION
# --------------------------------
def get_risk_reasons(
    damage_level,
    traffic,
    road_age,
    rainfall,
    previous_repairs
):

    reasons = []

    # Visual damage
    if damage_level == "Severe":
        reasons.append("🔴 Severe visual road damage detected")
    elif damage_level == "High":
        reasons.append("🟠 High level of visual road damage detected")
    elif damage_level == "Moderate":
        reasons.append("🟡 Moderate visual road damage detected")
    else:
        reasons.append("🟢 Minor visual road damage detected")

    # Traffic
    if traffic == "High":
        reasons.append("🚗 High traffic increases road wear and deterioration")
    elif traffic == "Medium":
        reasons.append("🚙 Medium traffic contributes to road deterioration")

    # Road age
    if road_age >= 15:
        reasons.append("🛣️ Road is old and may require structural maintenance")
    elif road_age >= 8:
        reasons.append("🛣️ Road age indicates increasing maintenance requirements")

    # Rainfall
    if rainfall >= 1200:
        reasons.append("🌧️ High rainfall can accelerate road deterioration")
    elif rainfall >= 700:
        reasons.append("🌦️ Rainfall may contribute to road surface damage")

    # Previous repairs
    if previous_repairs >= 5:
        reasons.append("🔧 Multiple previous repairs indicate recurring road issues")
    elif previous_repairs >= 2:
        reasons.append("🔧 Previous repairs indicate a history of maintenance issues")

    return reasons


# --------------------------------
# APP TITLE
# --------------------------------
st.title("🚧 InfraVision AI")

st.subheader(
    "AI-Powered Road Damage Detection & Maintenance Priority System"
)

st.write(
    "Upload one or multiple road images. "
    "InfraVision AI analyzes visual road patterns, "
    "predicts maintenance risk, explains the risk factors, "
    "and ranks roads by maintenance priority."
)

st.divider()


# --------------------------------
# IMAGE UPLOAD
# --------------------------------
uploaded_files = st.file_uploader(
    "📤 Upload Road Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# --------------------------------
# ROAD INFORMATION
# --------------------------------
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


traffic_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}


# --------------------------------
# ANALYZE BUTTON
# --------------------------------
if st.button("🚀 Analyze Infrastructure", type="primary"):

    if not uploaded_files:

        st.warning("Please upload at least one road image.")

    else:

        results = []
        analyzed_images = []

        for uploaded_file in uploaded_files:

            # Analyze image
            image, edge_density, damage_level, damage_value = (
                analyze_road_image(uploaded_file)
            )

            # ML model input
            input_data = pd.DataFrame({
                "road_age": [road_age],
                "traffic_level": [traffic_map[traffic]],
                "rainfall": [rainfall],
                "previous_repairs": [previous_repairs],
                "damage_severity": [damage_value]
            })

            # Predict risk
            risk_score = float(model.predict(input_data)[0])

            # Add visual analysis influence
            visual_risk = min(edge_density * 5, 25)

            risk_score = min(
                100,
                risk_score + visual_risk
            )

            # Determine priority
            if risk_score >= 80:

                priority = "🔴 Critical"
                recommendation = (
                    "Immediate inspection and repair recommended."
                )

            elif risk_score >= 60:

                priority = "🟠 High"
                recommendation = (
                    "Schedule maintenance as soon as possible."
                )

            elif risk_score >= 40:

                priority = "🟡 Medium"
                recommendation = (
                    "Monitor the road and plan maintenance."
                )

            else:

                priority = "🟢 Low"
                recommendation = (
                    "Regular monitoring is recommended."
                )

            # Get AI explanation
            reasons = get_risk_reasons(
                damage_level,
                traffic,
                road_age,
                rainfall,
                previous_repairs
            )

            # Store result
            results.append({
                "Image": uploaded_file.name,
                "Visual Damage": damage_level,
                "Risk Score": round(risk_score, 2),
                "Priority": priority,
                "Recommendation": recommendation,
                "Risk Explanation": " | ".join(reasons)
            })

            analyzed_images.append({
                "name": uploaded_file.name,
                "image": image,
                "edge_density": edge_density,
                "damage_level": damage_level,
                "risk_score": risk_score,
                "priority": priority,
                "reasons": reasons
            })


        # --------------------------------
        # RESULTS DATAFRAME
        # --------------------------------
        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="Risk Score",
            ascending=False
        )

        results_df.index = (
            range(1, len(results_df) + 1)
        )


        # --------------------------------
        # SUCCESS MESSAGE
        # --------------------------------
        st.success(
            "AI analysis completed successfully!"
        )


        # --------------------------------
        # DASHBOARD
        # --------------------------------
        st.subheader(
            "📊 Infrastructure Analysis Dashboard"
        )

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


        # --------------------------------
        # PRIORITY RANKING
        # --------------------------------
        st.subheader(
            "🏆 Maintenance Priority Ranking"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )


        # --------------------------------
        # DOWNLOAD REPORT
        # --------------------------------
        st.subheader(
            "📥 Download Maintenance Report"
        )

        csv = results_df.to_csv(
            index=True
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Maintenance Report (CSV)",
            data=csv,
            file_name="infravision_maintenance_report.csv",
            mime="text/csv"
        )


        # --------------------------------
        # AI EXPLANATION
        # --------------------------------
        st.subheader(
            "🧠 Why Is This Road Risky?"
        )

        for item in analyzed_images:

            with st.expander(
                f"{item['priority']} — {item['name']} "
                f"({item['risk_score']:.1f}% Risk)"
            ):

                st.write(
                    f"### Risk Factors for {item['name']}"
                )

                for reason in item["reasons"]:
                    st.write(reason)

                st.info(
                    "InfraVision AI combines visual road analysis "
                    "with traffic, road age, rainfall, and maintenance "
                    "history to prioritize infrastructure maintenance."
                )


        # --------------------------------
        # IMAGE RESULTS
        # --------------------------------
        st.subheader(
            "📷 AI Image Analysis"
        )

        image_columns = st.columns(3)

        for index, item in enumerate(analyzed_images):

            with image_columns[index % 3]:

                st.image(
                    item["image"],
                    caption=(
                        f"{item['name']} | "
                        f"{item['damage_level']} Damage | "
                        f"Risk: {item['risk_score']:.1f}%"
                    ),
                    use_container_width=True
                )