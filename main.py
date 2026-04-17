import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# --- Data Loading & Cleaning ---
df = pd.read_csv("ICRISAT-District Level Data.csv")
df.columns = df.columns.str.lower().str.replace(' ', '_')
thresh = len(df) * 0.5
df = df.dropna(axis=1, thresh=thresh)
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# --- Wide to Long Formatting ---
crops = set()
for col in df.columns:
    if col.endswith('_area_(1000_ha)'):
        crop_name = col.replace('_area_(1000_ha)', '')
        if f'{crop_name}_production_(1000_tons)' in df.columns and f'{crop_name}_yield_(kg_per_ha)' in df.columns:
            crops.add(crop_name)

crops = sorted(list(crops))
base_cols = ['state_name', 'dist_name', 'year']
long_frames = []

for crop in crops:
    area_col = f'{crop}_area_(1000_ha)'
    prod_col = f'{crop}_production_(1000_tons)'
    yield_col = f'{crop}_yield_(kg_per_ha)'
    
    temp_df = df[base_cols + [area_col, prod_col, yield_col]].copy()
    temp_df.columns = ['state_name', 'dist_name', 'year', 'area', 'production', 'yield']
    temp_df['crop'] = crop
    long_frames.append(temp_df)

long_df = pd.concat(long_frames, ignore_index=True)
long_df = long_df[long_df['area'] > 0]
long_df = long_df[['state_name', 'dist_name', 'year', 'crop', 'area', 'production', 'yield']]

# --- Merging Rainfall Data ---
rain_df = pd.read_csv("rainfall in india 1901-2015.csv")
rain_df.columns = rain_df.columns.str.lower().str.replace(' ', '_')
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
rain_df['calculated_annual_rainfall'] = rain_df[months].sum(axis=1)

rain_df['state_match'] = rain_df['subdivision'].str.upper()
long_df['state_match'] = long_df['state_name'].str.upper()

merged_df = pd.merge(
    long_df, 
    rain_df[['state_match', 'year', 'calculated_annual_rainfall']], 
    on=['state_match', 'year'], 
    how='left'
)
merged_df.drop(columns=['state_match'], inplace=True)
merged_df.rename(columns={'calculated_annual_rainfall': 'annual_rainfall'}, inplace=True)

# --- Merging Temperature Data ---
temp_df = pd.read_csv("temperatures.csv")
temp_df.columns = temp_df.columns.str.lower().str.replace(' ', '_')
temp_df.rename(columns={'annual': 'avg_temperature'}, inplace=True)

final_df = pd.merge(
    merged_df,
    temp_df[['year', 'avg_temperature']],
    on='year',
    how='left'
)

# ==========================================
# --- Prepare Dataset for ML ---
# ==========================================
print("\n--- ML Data Preparation ---")

# Fill/drop NaNs that might have been introduced during left joins
final_df = final_df.dropna(subset=['annual_rainfall', 'avg_temperature', 'yield'])

# 1. Select features & target
X = final_df[['crop', 'area', 'annual_rainfall', 'avg_temperature']].copy()
y = final_df['yield']

# 2. Encode categorical column: crop using Label Encoding
le = LabelEncoder()
X['crop'] = le.fit_transform(X['crop'])

# 3. Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Print information
print("Feature matrix (X) shape:", X.shape)
print("Target vector (y) shape:", y.shape)

print("\nTrain-Test Split Sizes:")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")

# ==========================================
# --- Model Training & Evaluation ---
# ==========================================
print("\n--- Model Training & Evaluation ---")

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

print("\nModel: Linear Regression")
print(f"R2 score: {lr_r2:.4f}")
print(f"MAE:      {lr_mae:.4f}")

# 2. Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print("\nModel: Random Forest Regressor")
print(f"R2 score: {rf_r2:.4f}")
print(f"MAE:      {rf_mae:.4f}")

# ==========================================
# --- Feature Importance ---
# ==========================================
print("\n--- Feature Importance (Random Forest) ---")
feature_names = ['crop', 'area', 'annual_rainfall', 'avg_temperature']
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

for index, row in importance_df.iterrows():
    print(f"{row['Feature']:<15} : {row['Importance']:.4f}")

most_important = importance_df.iloc[0]['Feature']
least_important = importance_df.iloc[-1]['Feature']

print("\n--- Interpretation ---")
print(f"- The factor that affects yield the most is '{most_important}'.")
print(f"- The factor that is least important is '{least_important}'.")

# ==========================================
# --- Custom Input Testing ---
# ==========================================
print("\n--- Testing Model with Custom Inputs ---")
test_crop = le.classes_[0]
if 'rice' in le.classes_:
    test_crop = 'rice'
elif 'wheat' in le.classes_:
    test_crop = 'wheat'

test_crop_encoded = le.transform([test_crop])[0]

def predict_yield(crop_enc, area, rain, temp):
    input_data = pd.DataFrame(
        [[crop_enc, area, rain, temp]], 
        columns=['crop', 'area', 'annual_rainfall', 'avg_temperature']
    )
    return rf_model.predict(input_data)[0]

base_area = 1.0
base_rain = 800
base_temp = 30
base_pred = predict_yield(test_crop_encoded, base_area, base_rain, base_temp)

print(f"Base Case: Crop='{test_crop}', Area={base_area}, Rainfall={base_rain}, Temp={base_temp}")
print(f"Predicted Yield: {base_pred:.4f} kg/ha\n")

print("--- Varying Rainfall (Fixed Area/Temp) ---")
for r in [500, 1000]:
    pred = predict_yield(test_crop_encoded, base_area, r, base_temp)
    print(f"Rainfall: {r:>4} -> Predicted Yield: {pred:8.4f} kg/ha (Change: {pred - base_pred:>9.4f})")

print("\n--- Varying Temperature (Fixed Area/Rainfall) ---")
for t in [25, 35]:
    pred = predict_yield(test_crop_encoded, base_area, base_rain, t)
    print(f"Temperature: {t:>2} -> Predicted Yield: {pred:8.4f} kg/ha (Change: {pred - base_pred:>9.4f})")

# ==========================================
# --- Compare Yield Across Crops ---
# ==========================================
print("\n--- Comparing Yield Across Crops ---")
print("Conditions: Area=1.0, Rainfall=800, Temp=30")

crop_yields = []

for crop_name in le.classes_:
    encoded_c = le.transform([crop_name])[0]
    yield_pred = predict_yield(encoded_c, 1.0, 800, 30)
    crop_yields.append({'Crop': crop_name, 'Predicted Yield': yield_pred})

yields_df = pd.DataFrame(crop_yields)
yields_df = yields_df.sort_values(by='Predicted Yield', ascending=False).reset_index(drop=True)

print("\nTop 5 Crops with Highest Predicted Yield:")
for index, row in yields_df.head(5).iterrows():
    print(f"{index+1}. {row['Crop']:<15} : {row['Predicted Yield']:.4f} kg/ha")

# ==========================================
# --- Single Crop (Rice) Model Testing ---
# ==========================================
print("\n==========================================")
print("--- Single Crop Model Testing (Rice) ---")
print("==========================================")

df_rice = final_df[final_df['crop'] == 'rice'].copy()

if len(df_rice) > 0:
    X_rice = df_rice[['area', 'annual_rainfall', 'avg_temperature']]
    y_rice = df_rice['yield']
    
    X_rice_train, X_rice_test, y_rice_train, y_rice_test = train_test_split(X_rice, y_rice, test_size=0.20, random_state=42)
    
    rf_rice_model = RandomForestRegressor(random_state=42)
    rf_rice_model.fit(X_rice_train, y_rice_train)
    
    rice_pred = rf_rice_model.predict(X_rice_test)
    rice_r2 = r2_score(y_rice_test, rice_pred)
    rice_mae = mean_absolute_error(y_rice_test, rice_pred)
    
    print("\n--- Model Evaluation (Rice Only) ---")
    print(f"R2 score: {rice_r2:.4f}")
    print(f"MAE:      {rice_mae:.4f}")
    
    def predict_rice_yield(area, rain, temp):
        input_df = pd.DataFrame([[area, rain, temp]], columns=['area', 'annual_rainfall', 'avg_temperature'])
        return rf_rice_model.predict(input_df)[0]
    
    base_area_rice = 1.0
    base_rain_rice = 800
    base_temp_rice = 30
    base_pred_rice = predict_rice_yield(base_area_rice, base_rain_rice, base_temp_rice)
    
    print("\n--- Scenario Testing (Rice Only) ---")
    print(f"Base Case: Area={base_area_rice}, Rainfall={base_rain_rice}, Temp={base_temp_rice}")
    print(f"Predicted Yield: {base_pred_rice:.4f} kg/ha\n")
    
    print("--- Varying Rainfall ---")
    for r in [500, 1000]:
        pred = predict_rice_yield(base_area_rice, r, base_temp_rice)
        print(f"Rainfall: {r:>4} -> Predicted Yield: {pred:8.4f} kg/ha (Change: {pred - base_pred_rice:>9.4f})")
    
    print("\n--- Varying Temperature ---")
    for t in [25, 35]:
        pred = predict_rice_yield(base_area_rice, base_rain_rice, t)
        print(f"Temperature: {t:>2} -> Predicted Yield: {pred:8.4f} kg/ha (Change: {pred - base_pred_rice:>9.4f})")

    # ==========================================
    # --- Rice Model Testing (NO AREA FEATURE) ---
    # ==========================================
    print("\n==========================================")
    print("--- Rice Model Testing (No 'Area' Feature) ---")
    print("==========================================")
    
    X_rice_2 = df_rice[['annual_rainfall', 'avg_temperature']]
    y_rice_2 = df_rice['yield']
    
    X_r2_train, X_r2_test, y_r2_train, y_r2_test = train_test_split(X_rice_2, y_rice_2, test_size=0.20, random_state=42)
    
    rf_rice2_model = RandomForestRegressor(random_state=42)
    rf_rice2_model.fit(X_r2_train, y_r2_train)
    
    rice2_pred = rf_rice2_model.predict(X_r2_test)
    rice2_r2 = r2_score(y_r2_test, rice2_pred)
    rice2_mae = mean_absolute_error(y_r2_test, rice2_pred)
    
    print("\n--- Model Evaluation (Rice Only - No Area) ---")
    print(f"R2 score: {rice2_r2:.4f}")
    print(f"MAE:      {rice2_mae:.4f}")
    
    def predict_rice_no_area(rain, temp):
        input_df = pd.DataFrame([[rain, temp]], columns=['annual_rainfall', 'avg_temperature'])
        return rf_rice2_model.predict(input_df)[0]
    
    print("\n--- Scenario Testing (Rice Only - No Area) ---")
    print("Varying Rainfall (Temp Fixed at 30):")
    for r in [500, 800, 1000]:
        pred = predict_rice_no_area(r, 30)
        print(f"  Rainfall: {r:>4} -> Predicted Yield: {pred:8.4f} kg/ha")
        
    print("\nVarying Temperature (Rainfall Fixed at 800):")
    for t in [25, 30, 35]:
        pred = predict_rice_no_area(800, t)
        print(f"  Temperature: {t:>2} -> Predicted Yield: {pred:8.4f} kg/ha")

else:
    print("'rice' not found in dataset for single-crop testing.")

# ==========================================
# --- Rice Model Testing (Specific State) ---
# ==========================================
print("\n==========================================")
print("--- Rice Model Testing (Specific State) ---")
print("==========================================")

# Debug: Print available states and crops
print("Available States (first 10):", final_df['state_name'].unique()[:10])
print("Available Crops:", final_df['crop'].unique())

target_state = 'ANDHRA PRADESH'
df_state_rice = final_df[(final_df['crop'] == 'rice') & (final_df['state_name'].str.upper() == target_state)].copy()

if len(df_state_rice) > 0:
    X_state = df_state_rice[['annual_rainfall', 'avg_temperature']]
    y_state = df_state_rice['yield']
    
    if len(df_state_rice) > 5:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_state, y_state, test_size=0.20, random_state=42)
        
        rf_state = RandomForestRegressor(random_state=42)
        rf_state.fit(X_train_s, y_train_s)
        
        y_pred = rf_state.predict(X_test_s)
        print(f"\n--- Model Evaluation (Rice in {target_state}) ---")
        print(f"R2 score: {r2_score(y_test_s, y_pred):.4f}")
        print(f"MAE:      {mean_absolute_error(y_test_s, y_pred):.4f}")
        
        def predict_state_rice(rain, temp):
            return rf_state.predict(pd.DataFrame([[rain, temp]], columns=['annual_rainfall', 'avg_temperature']))[0]
            
        print(f"\n--- Scenario Testing (Rice in {target_state}) ---")
        print("Varying Rainfall (Temp Fixed at 30):")
        for r in [500, 800, 1000]:
            print(f"  Rainfall: {r:>4} -> Predicted Yield: {predict_state_rice(r, 30):8.4f} kg/ha")
    else:
        print(f"Not enough data for {target_state} specifically.")
else:
    print(f"'rice' in '{target_state}' not found.")


import matplotlib.pyplot as plt
import numpy as np

# Convert to 1D numpy arrays (works for pandas Series/DataFrame too)
actual = np.ravel(y_test)
predicted = np.ravel(y_pred)

# Scatter plot: Actual (x) vs Predicted (y)
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.7)

# Perfect prediction diagonal line
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

# Labels and title
plt.title("Actual vs Predicted Yield")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()