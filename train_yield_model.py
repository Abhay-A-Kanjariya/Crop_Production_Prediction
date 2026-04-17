import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_and_save(data_path: str) -> None:
    df = pd.read_csv(data_path)

    required_columns = {
        "state_name",
        "crop",
        "annual_rainfall",
        "avg_temperature",
        "yield",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Encode categorical columns
    crop_encoder = {value: idx for idx, value in enumerate(sorted(df["crop"].dropna().unique()))}
    state_encoder = {
        value: idx for idx, value in enumerate(sorted(df["state_name"].dropna().unique()))
    }

    df["crop_encoded"] = df["crop"].map(crop_encoder)
    df["state_encoded"] = df["state_name"].map(state_encoder)

    # Engineered feature
    df["rainfall_squared"] = df["annual_rainfall"] ** 2

    # Features and target
    feature_cols = [
        "crop_encoded",
        "state_encoded",
        "annual_rainfall",
        "avg_temperature",
        "rainfall_squared",
    ]
    X = df[feature_cols]
    y = df["yield"]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Save artifacts
    joblib.dump(model, "yield_random_forest_model.pkl")
    joblib.dump(crop_encoder, "crop_encoder.pkl")
    joblib.dump(state_encoder, "state_encoder.pkl")

    print("Saved: yield_random_forest_model.pkl, crop_encoder.pkl, state_encoder.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV dataset")
    args = parser.parse_args()
    train_and_save(args.data)
