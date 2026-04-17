import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from data_preparation import prepare_final_dataset


def train_and_save(
    dataset_path: str = "final_dataset.csv",
    model_path: str = "model.pkl",
    crop_encoder_path: str = "crop_encoder.pkl",
    state_encoder_path: str = "state_encoder.pkl",
) -> None:
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        df = prepare_final_dataset(output_path=dataset_path)

    required = ["state_name", "crop", "annual_rainfall", "avg_temperature", "yield"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training dataset missing columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["state_name"] = df["state_name"].astype(str).str.upper().str.strip()
    df["crop"] = df["crop"].astype(str).str.lower().str.strip()
    df["annual_rainfall"] = pd.to_numeric(df["annual_rainfall"], errors="coerce")
    df["avg_temperature"] = pd.to_numeric(df["avg_temperature"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    # Ensure yield is correctly expressed as kg/ha.
    if "production" in df.columns and "area" in df.columns:
        prod_num = pd.to_numeric(df["production"], errors="coerce")
        area_num = pd.to_numeric(df["area"], errors="coerce")
        valid = area_num.notna() & (area_num != 0) & prod_num.notna()
        df.loc[valid, "yield"] = (prod_num[valid] / area_num[valid]) * 1000.0

    df = df.dropna(subset=["annual_rainfall", "avg_temperature", "yield"]).copy()

    crop_encoder = LabelEncoder()
    state_encoder = LabelEncoder()

    df["crop_encoded"] = crop_encoder.fit_transform(df["crop"])
    df["state_encoded"] = state_encoder.fit_transform(df["state_name"])
    df["rainfall_squared"] = df["annual_rainfall"] ** 2

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

    joblib.dump(model, model_path)
    joblib.dump(crop_encoder, crop_encoder_path)
    joblib.dump(state_encoder, state_encoder_path)

    print(f"Saved model: {model_path}")
    print(f"Saved crop encoder: {crop_encoder_path}")
    print(f"Saved state encoder: {state_encoder_path}")
    print(f"Target yield min (kg/ha): {df['yield'].min():.2f}")
    print(f"Target yield max (kg/ha): {df['yield'].max():.2f}")
    print(f"Target yield mean (kg/ha): {df['yield'].mean():.2f}")

    # Validation scenario requested: rice, Punjab, rainfall=800-1000, temperature=30
    if "rice" in set(crop_encoder.classes_) and "PUNJAB" in set(state_encoder.classes_):
        rice_enc = int(crop_encoder.transform(["rice"])[0])
        punjab_enc = int(state_encoder.transform(["PUNJAB"])[0])
        test_rows = pd.DataFrame(
            [
                [rice_enc, punjab_enc, 800.0, 30.0, 800.0**2],
                [rice_enc, punjab_enc, 900.0, 30.0, 900.0**2],
                [rice_enc, punjab_enc, 1000.0, 30.0, 1000.0**2],
            ],
            columns=[
                "crop_encoded",
                "state_encoded",
                "annual_rainfall",
                "avg_temperature",
                "rainfall_squared",
            ],
        )
        preds = model.predict(test_rows)
        print(
            "Validation (rice, PUNJAB, rainfall 800/900/1000, temp 30) kg/ha:",
            [round(float(p), 2) for p in preds],
        )
        if any((p < 2000.0) or (p > 4000.0) for p in preds):
            print("WARNING: Validation prediction is outside expected 2000-4000 kg/ha range.")
    else:
        print("WARNING: Could not run Punjab rice validation; crop/state not found in encoders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="final_dataset.csv")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--crop_encoder", default="crop_encoder.pkl")
    parser.add_argument("--state_encoder", default="state_encoder.pkl")
    args = parser.parse_args()

    train_and_save(
        dataset_path=args.dataset,
        model_path=args.model,
        crop_encoder_path=args.crop_encoder,
        state_encoder_path=args.state_encoder,
    )
