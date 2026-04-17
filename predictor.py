from typing import Union

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _safe_transform(encoder: LabelEncoder, value: str, field_name: str) -> int:
    classes = set(encoder.classes_)
    if value not in classes:
        preview = ", ".join(list(encoder.classes_)[:10])
        raise ValueError(f"Unknown {field_name}: '{value}'. Available examples: {preview}")
    return int(encoder.transform([value])[0])


def predict_yield(
    crop: str,
    state: str,
    rainfall: Union[int, float],
    temperature: Union[int, float],
    model_path: str = "model.pkl",
    crop_encoder_path: str = "crop_encoder.pkl",
    state_encoder_path: str = "state_encoder.pkl",
) -> float:
    model = joblib.load(model_path)
    crop_encoder: LabelEncoder = joblib.load(crop_encoder_path)
    state_encoder: LabelEncoder = joblib.load(state_encoder_path)

    crop_clean = str(crop).strip().lower()
    state_clean = str(state).strip().upper()
    rainfall_val = float(rainfall)
    temperature_val = float(temperature)

    crop_encoded = _safe_transform(crop_encoder, crop_clean, "crop")
    state_encoded = _safe_transform(state_encoder, state_clean, "state")
    rainfall_squared = rainfall_val ** 2

    features = pd.DataFrame(
        [[crop_encoded, state_encoded, rainfall_val, temperature_val, rainfall_squared]],
        columns=[
            "crop_encoded",
            "state_encoded",
            "annual_rainfall",
            "avg_temperature",
            "rainfall_squared",
        ],
    )
    prediction = model.predict(features)[0]
    # Model is trained directly on kg/ha, so return as-is (no extra scaling).
    return float(prediction)
