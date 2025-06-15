from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/random_forest_classifier.pkl")

# (Optional) If you saved it, load feature order
# Otherwise, manually define the correct order based on training
feature_order = [
    'Sh/90', 'SoT/90', 'Cmp%', 'xG/90', 'npxG/90',
    'xAG/90', 'xA/90', 'Touches/90', 'KP/90', 'Tkl/90',
    'Pass/90', 'PrgP/90', 'PrgC/90', 'PrgR/90'
]

# Position label mapping (adjust as per LabelEncoder used)
label_mapping = {0: 'DF', 1: 'FW', 2: 'GK', 3: 'MF'}

# Define FastAPI app
app = FastAPI(title="Player Position Prediction API")

# Define the expected input structure
class PlayerStats(BaseModel):
    Sh_90: float
    SoT_90: float
    Cmp_perc: float
    xG_90: float
    npxG_90: float
    xAG_90: float
    xA_90: float
    Touches_90: float
    KP_90: float
    Tkl_90: float
    Pass_90: float
    PrgP_90: float
    PrgC_90: float
    PrgR_90: float

    class Config:
        schema_extra = {
            "example": {
                "Sh_90": 2.1,
                "SoT_90": 1.2,
                "Cmp_perc": 87.5,
                "xG_90": 0.32,
                "npxG_90": 0.30,
                "xAG_90": 0.15,
                "xA_90": 0.22,
                "Touches_90": 45.3,
                "KP_90": 1.8,
                "Tkl_90": 2.0,
                "Pass_90": 34.5,
                "PrgP_90": 5.6,
                "PrgC_90": 3.9,
                "PrgR_90": 1.4
            }
        }

# Health check
@app.get("/health")
def health_check():
    return {"status": "API is running ✅"}

# Prediction endpoint
@app.post("/predict")
def predict_position(stats: PlayerStats):
    input_dict = stats.dict()

    # Convert to DataFrame and enforce feature order
    input_df = pd.DataFrame([[input_dict[col.replace("/", "_").replace("%", "_perc")]] for col in feature_order]).T
    input_df.columns = feature_order

    # Predict
    prediction = model.predict(input_df)[0]
    position = label_mapping.get(prediction, "Unknown")

    return {
        "predicted_class": int(prediction),
        "predicted_position": position
    }
