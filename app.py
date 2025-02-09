import streamlit as st
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import joblib
import os
import cv2
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor
from crewai import Agent, Crew
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from matplotlib.lines import Line2D  

# Set Page Config
st.set_page_config(page_title="AstroCleanAI 🚀", page_icon="🌌", layout="wide")

# Load YOLOv5 Model for Space Debris Detection
@st.cache_resource
def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_yolov5_model()

# Load AI Models for Debris Classification
debris_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
processor = AutoProcessor.from_pretrained("microsoft/resnet-50")

# Ensure Collision Model Exists
MODEL_PATH = "collision_model.pkl"

def train_collision_model():
    """ Train and save an XGBoost model if it doesn't exist """
    X_train = np.random.rand(1000, 3) * [1000, 10, 50]  # (Altitude, Speed, Size)
    y_train = np.random.randint(0, 2, 1000)  # Binary labels (0 = No collision, 1 = Collision)
    
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    return model

# Load or Train Collision Model
if os.path.exists(MODEL_PATH):
    try:
        collision_model = joblib.load(MODEL_PATH)
    except:
        collision_model = train_collision_model()
else:
    collision_model = train_collision_model()

# AI Agents Setup (CrewAI)
agent1 = Agent(
    role="Debris Analyzer",
    goal="Classify space debris images.",
    backstory="An advanced AI specialist in space observation and debris analysis."
)

agent2 = Agent(
    role="Collision Predictor",
    goal="Predict space debris collision risks.",
    backstory="An AI researcher with expertise in orbital mechanics and risk assessment."
)

crew = Crew(agents=[agent1, agent2])

# Streamlit UI
st.title("🚀 AstroCleanAI: AI-Powered Space Agent 🤖")
st.markdown("### *Monitor, Predict, and Prevent Space Debris Collisions in Real-Time! 🌌*")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Collision Prediction", "🛰️ Space Debris Simulation", "📡 AI Debris Detection & Classification"])

# 📊 **Tab 1: Collision Prediction**
with tab1:
    st.markdown("### 📊 *Collision Risk Prediction*")

    # User inputs
    altitude = st.slider("Select Altitude (km)", 300, 1200, 500)
    speed = st.slider("Select Speed (km/s)", 5, 15, 7)
    size = st.slider("Select Debris Size (cm)", 1, 50, 10)

    # ✅ Improved Altitude Weight (Risk highest in 600-1200 km range)
    if altitude < 600:
        altitude_weight = 1.2  # Lower risk below dense LEO
    elif 600 <= altitude <= 1200:
        altitude_weight = 2.0  # Highest risk in dense LEO
    else:
        altitude_weight = 1.0  # Reduced risk at higher altitudes

    # ✅ Improved Speed Weight (Nonlinear Scaling)
    speed_weight = min(1 + np.log1p(speed - 5), 4.5)  # Capped at 4.5 for stability

    # ✅ Improved Size Weight (Square Root Scaling)
    size_weight = min(1 + np.sqrt(size / 10), 4.5)  # Capped at 4.5

    # ✅ Normalize and adjust risk (ensuring valid ML input)
    adjusted_risk = (altitude_weight * speed_weight * size_weight) / 10  # Normalize

    # **Prepare Input Data for Model**
    input_data = np.array([[altitude / 1200, speed / 15, size / 50]])  # Normalize inputs

    # ✅ Ensure Model Predicts Properly
    try:
        base_risk = collision_model.predict_proba(input_data)[0][1]  # ML prediction
        final_risk = min(base_risk * adjusted_risk, 1.0)  # Ensure max 100%
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        final_risk = 0.0  # Default risk in case of failure

    # 🚀 Display Risk
    st.metric("🚀 Collision Risk", f"{final_risk:.2%}")


# 🛰️ **Tab 2: Space Debris Simulation**
with tab2:
    st.markdown("### 🛰️ *Interactive Space Debris Simulation*")

    class CelestialBody:
        def __init__(self, position, velocity, size, mass, name, color):
            self.position = np.array(position, dtype=np.float64)
            self.velocity = np.array(velocity, dtype=np.float64)
            self.size = float(size)
            self.mass = float(mass)
            self.name = name
            self.color = color  # Color for plotting
            self.trajectory_history = [self.position.copy()]

        def update_position(self, dt):
            dt = float(dt)
            self.position = self.position + self.velocity * dt
            self.trajectory_history.append(self.position.copy())

    class CollisionTracker:
        def __init__(self):
            self.objects = []
            self.collision_threshold = 1.0  # km

        def add_object(self, space_object):
            self.objects.append(space_object)

        def calculate_collision_risk(self, obj1, obj2):
            distance = np.linalg.norm(obj1.position - obj2.position)
            relative_velocity = np.linalg.norm(obj1.velocity - obj2.velocity)
            risk = 1.0 / (distance * relative_velocity + 1e-6)
            return float(risk)

        def find_collision_risks(self):
            risks = []
            for i, obj1 in enumerate(self.objects):
                for j, obj2 in enumerate(self.objects[i + 1:], i + 1):
                    risk = self.calculate_collision_risk(obj1, obj2)
                    if risk > 0.1:  # Risk threshold
                        risks.append((obj1, obj2, risk))
            return risks

    def create_simulation_scenario():
        satellite = CelestialBody(
            position=[42164.0, 0.0, 0.0],
            velocity=[0.0, 3.075, 0.0],
            size=10.0,
            mass=1000.0,
            name="🛰 Active Satellite",
            color="green"
        )

        debris = CelestialBody(
            position=[42164.0, 100.0, 50.0],
            velocity=[0.1, 3.0, 0.1],
            size=0.1,
            mass=1.0,
            name="🪐 Debris",
            color="red"
        )

        return satellite, debris

    time_step = st.slider("⏱ Time Step (hours)", 0.1, 24.0, step=0.1)
    num_steps = st.slider("🔢 Number of Steps", 10, 1000, step=10)

    if st.button("🎮 Run Simulation"):
        tracker = CollisionTracker()
        satellite, debris = create_simulation_scenario()
        tracker.add_object(satellite)
        tracker.add_object(debris)

        fig = plt.figure(figsize=(12, 9))  # 🔧 Increased figure size
        ax = fig.add_subplot(111, projection='3d')

        for step in range(num_steps):
            for obj in tracker.objects:
                obj.update_position(time_step * 3600)

        ax.clear()

        x_vals, y_vals, z_vals = [], [], []  # To dynamically adjust limits

        for obj in tracker.objects:
            trajectory = np.array(obj.trajectory_history, dtype=np.float64)
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=obj.color, linewidth=1.5, alpha=0.6, label=obj.name
            )

            ax.scatter(
                trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                color=obj.color, marker='o', s=100, edgecolors="black", label=f"{obj.name} (Current)"
            )

            # Collecting min/max values for proper frame adjustment
            x_vals.extend(trajectory[:, 0])
            y_vals.extend(trajectory[:, 1])
            z_vals.extend(trajectory[:, 2])

        # 🔧 Adjust limits dynamically for full-frame view
        ax.set_xlim(min(x_vals) - 500, max(x_vals) + 500)
        ax.set_ylim(min(y_vals) - 500, max(y_vals) + 500)
        ax.set_zlim(min(z_vals) - 500, max(z_vals) + 500)  # Ensure Z is fully visible

        ax.set_xlabel("X Position (km)")
        ax.set_ylabel("Y Position (km)")
        ax.set_zlabel("Z Position (km)")
        ax.set_title("Space Debris Trajectory Simulation")

        # ✅ Fixed: Import `Line2D` from `matplotlib.lines`
        from matplotlib.lines import Line2D  

        # ✅ Custom legend with only pin-head markers
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label="Active Satellite"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="Debris"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()  # 🔧 Prevents label cutoff
        st.pyplot(fig)

# 🛰 **Debris Detection & Classification**
with tab3:
    st.markdown("### 🛰 *AI-Powered Debris Classification & Detection*")
    
    uploaded_file = st.file_uploader("Upload a Space Image for Debris Detection", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Perform detection
        with torch.no_grad():
            results = model(image)

        img_with_boxes = img_rgb.copy()
        debris_info = []

        for detection in results.xyxy[0].tolist():
            x1, y1, x2, y2, conf, class_id = detection
            width = x2 - x1
            height = y2 - y1
            size = width * height

            # Size category classification
            size_category = "Small"
            if size > 3000:
                size_category = "Large"
            elif size > 1000:
                size_category = "Medium"

            # Risk level classification
            risk_level = "Low"
            if size_category == "Large" and conf > 0.8:
                risk_level = "High"
            elif size_category == "Medium" and conf > 0.6:
                risk_level = "Moderate"

            debris_info.append({
                "coordinates": (int(x1), int(y1), int(x2), int(y2)),
                "size": round(size, 2),
                "size_category": size_category,
                "confidence": round(conf, 2),
                "risk_level": risk_level
            })

            # Draw bounding boxes
            cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{size_category} ({risk_level})"
            cv2.putText(img_with_boxes, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert image back for display
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        st.image(img_with_boxes_rgb, caption="🛰 Detected Debris", use_container_width=True)

        # Show debris information
        if debris_info:
            st.markdown("### 📝 Detected Debris Information")
            debris_df = pd.DataFrame(debris_info)
            st.dataframe(debris_df)
        else:
            st.warning("🌠 No debris detected in the image!")

st.markdown("--- 🌍 *Built by Space Ninjas 🚀 | © 2025 AstroCleanAI. All Rights Reserved.*")
