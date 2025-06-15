import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_classifier.pkl")

# Feature order must match model training
feature_order = [
    'Sh/90', 'SoT/90', 'Cmp%', 'xG/90', 'npxG/90',
    'xAG/90', 'xA/90', 'Touches/90', 'KP/90', 'Tkl/90',
    'Pass/90', 'PrgP/90', 'PrgC/90', 'PrgR/90'
]

# Label mapping (must match your LabelEncoder)
label_mapping = {0: 'DF', 1: 'FW', 2: 'GK', 3: 'MF'}

# Streamlit UI
st.set_page_config(page_title="Player Position Predictor", layout="centered")
st.title("⚽ Predict Player Position")
st.markdown("Enter a player's per-90 stats to predict their position.")

# Collect inputs
input_data = {}
for feat in feature_order:
    # Convert to a readable label for the UI (e.g. 'Sh/90' → 'Sh_90')
    label = feat.replace("/", "_").replace("%", "_perc")
    input_data[feat] = st.number_input(label, min_value=0.0, step=0.01)

# Predict button
if st.button("Predict Position"):
    # Prepare input DataFrame in the correct order
    input_df = pd.DataFrame([[input_data[feat] for feat in feature_order]], columns=feature_order)

    # Predict
    pred_class = model.predict(input_df)[0]
    pred_position = label_mapping.get(pred_class, "Unknown")

    st.success(f"🟢 Predicted Position: **{pred_position}**")
