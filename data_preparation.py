import argparse
from typing import Dict, List

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    df = df.copy()
    df.columns = cleaned
    return df


def build_long_crop_dataframe(icrisat_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["state_name", "dist_name", "year"]
    missing_base = [c for c in base_cols if c not in icrisat_df.columns]
    if missing_base:
        raise ValueError(f"Missing required base columns in ICRISAT data: {missing_base}")

    crop_names: List[str] = []
    for col in icrisat_df.columns:
        if col.endswith("_area_(1000_ha)"):
            crop_name = col.replace("_area_(1000_ha)", "")
            prod_col = f"{crop_name}_production_(1000_tons)"
            if prod_col in icrisat_df.columns:
                crop_names.append(crop_name)

    if not crop_names:
        raise ValueError("No crop area/production column pairs found in ICRISAT data.")

    frames = []
    for crop in sorted(set(crop_names)):
        area_col = f"{crop}_area_(1000_ha)"
        production_col = f"{crop}_production_(1000_tons)"
        temp_df = icrisat_df[base_cols + [area_col, production_col]].copy()
        temp_df.columns = ["state_name", "dist_name", "year", "area", "production"]
        temp_df["crop"] = crop
        frames.append(temp_df)

    long_df = pd.concat(frames, ignore_index=True)
    return long_df


def merge_rainfall(long_df: pd.DataFrame, rainfall_df: pd.DataFrame) -> pd.DataFrame:
    month_cols = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    missing_months = [c for c in month_cols if c not in rainfall_df.columns]
    if missing_months:
        raise ValueError(f"Rainfall file missing month columns: {missing_months}")
    if "subdivision" not in rainfall_df.columns or "year" not in rainfall_df.columns:
        raise ValueError("Rainfall file must contain 'subdivision' and 'year'.")

    rain_df = rainfall_df.copy()
    rain_df["annual_rainfall"] = rain_df[month_cols].sum(axis=1)
    rain_df["state_match"] = rain_df["subdivision"].astype(str).str.upper().str.strip()

    # Minimal mapping to improve common match issues.
    state_mapping: Dict[str, str] = {
        "ANDHRA PRADESH": "COASTAL ANDHRA PRADESH",
    }
    data_df = long_df.copy()
    data_df["state_match"] = (
        data_df["state_name"].astype(str).str.upper().str.strip().replace(state_mapping)
    )

    merged = data_df.merge(
        rain_df[["state_match", "year", "annual_rainfall"]],
        on=["state_match", "year"],
        how="left",
    )
    return merged.drop(columns=["state_match"])


def merge_temperature(data_df: pd.DataFrame, temperature_df: pd.DataFrame) -> pd.DataFrame:
    if "year" not in temperature_df.columns:
        raise ValueError("Temperature file must contain 'year'.")
    if "annual" in temperature_df.columns:
        temperature_df = temperature_df.rename(columns={"annual": "avg_temperature"})
    if "avg_temperature" not in temperature_df.columns:
        raise ValueError("Temperature file must contain 'annual' or 'avg_temperature'.")

    final_df = data_df.merge(temperature_df[["year", "avg_temperature"]], on="year", how="left")
    return final_df


def prepare_final_dataset(
    icrisat_path: str = "ICRISAT-District Level Data.csv",
    rainfall_path: str = "rainfall in india 1901-2015.csv",
    temperature_path: str = "temperatures.csv",
    output_path: str = "final_dataset.csv",
) -> pd.DataFrame:
    df = pd.read_csv(icrisat_path, low_memory=False)
    df = normalize_columns(df)

    long_df = build_long_crop_dataframe(df)

    rain_df = normalize_columns(pd.read_csv(rainfall_path, low_memory=False))
    merged_df = merge_rainfall(long_df, rain_df)

    temp_df = normalize_columns(pd.read_csv(temperature_path, low_memory=False))
    final_df = merge_temperature(merged_df, temp_df)

    required = ["state_name", "crop", "annual_rainfall", "avg_temperature", "production", "area"]
    missing = [c for c in required if c not in final_df.columns]
    if missing:
        raise ValueError(f"Final dataframe missing required columns: {missing}")

    final_df["area"] = pd.to_numeric(final_df["area"], errors="coerce")
    final_df["production"] = pd.to_numeric(final_df["production"], errors="coerce")
    final_df["annual_rainfall"] = pd.to_numeric(final_df["annual_rainfall"], errors="coerce")
    final_df["avg_temperature"] = pd.to_numeric(final_df["avg_temperature"], errors="coerce")

    # production is in (1000 tons) and area is in (1000 hectares)
    # so convert to kg/ha by multiplying ratio by 1000
    final_df["yield"] = (final_df["production"] / final_df["area"]) * 1000.0

    # Row cleanup
    final_df = final_df[(final_df["area"] > 0) & (final_df["production"] > 0)]
    final_df = final_df.dropna(subset=required + ["yield"]).copy()
    # Keep realistic agronomic range in kg/ha.
    final_df = final_df[final_df["yield"].between(1000.0, 6000.0)]

    # Standardization
    final_df["state_name"] = final_df["state_name"].astype(str).str.upper().str.strip()
    final_df["crop"] = final_df["crop"].astype(str).str.lower().str.strip()

    final_df = final_df[
        [
            "state_name",
            "crop",
            "annual_rainfall",
            "avg_temperature",
            "production",
            "area",
            "yield",
        ]
    ].reset_index(drop=True)

    final_df.to_csv(output_path, index=False)
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--icrisat", default="ICRISAT-District Level Data.csv")
    parser.add_argument("--rainfall", default="rainfall in india 1901-2015.csv")
    parser.add_argument("--temperature", default="temperatures.csv")
    parser.add_argument("--output", default="final_dataset.csv")
    args = parser.parse_args()

    final = prepare_final_dataset(
        icrisat_path=args.icrisat,
        rainfall_path=args.rainfall,
        temperature_path=args.temperature,
        output_path=args.output,
    )
    print(f"Saved cleaned dataset: {args.output}")
    print(f"Rows: {len(final)}")
    print(f"Yield min (kg/ha): {final['yield'].min():.2f}")
    print(f"Yield max (kg/ha): {final['yield'].max():.2f}")
    print(f"Yield mean (kg/ha): {final['yield'].mean():.2f}")
    realistic = final["yield"].between(1000, 6000).mean() * 100.0
    print(f"Yield in 1000-6000 kg/ha range: {realistic:.2f}%")
