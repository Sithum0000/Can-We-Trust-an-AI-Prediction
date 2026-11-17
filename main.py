import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
data = pd.read_csv("hour.csv")

print("\n=== DATASET INFORMATION ===")
print("Dataset size:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

# 2. Understand the target
print("\n=== BIKE RENTAL SUMMARY ===")
print(data["cnt"].describe())

# 3. Prepare the data
# Do not use casual or registered because
# casual + registered = cnt (data leakage).
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed"
]

X = data[features]
y = data["cnt"]

# 4. Create a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2

results = []

# 5. Experiment A: Random split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y, test_size=0.20, random_state=42
)

random_models = {
    "Random Baseline": DummyRegressor(strategy="median"),
    "Random Forest - Random Split": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

for name, model in random_models.items():
    scores = evaluate_model(
        model, X_train_r, X_test_r, y_train_r, y_test_r
    )
    results.append([name, *scores])

# 6. Experiment B: Chronological split
# Earlier records train the model; later records test it.
split_index = int(len(data) * 0.80)

X_train_t = X.iloc[:split_index]
X_test_t = X.iloc[split_index:]
y_train_t = y.iloc[:split_index]
y_test_t = y.iloc[split_index:]

time_models = {
    "Time Baseline": DummyRegressor(strategy="median"),
    "Random Forest - Time Split": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

for name, model in time_models.items():
    scores = evaluate_model(
        model, X_train_t, X_test_t, y_train_t, y_test_t
    )
    results.append([name, *scores])

# 7. Show and save results
results_df = pd.DataFrame(
    results,
    columns=["Experiment", "MAE", "RMSE", "R2"]
)

print("\n=== MODEL RESULTS ===")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("model_results.csv", index=False)

print("\nResults saved to model_results.csv")
print("First research experiment completed!")


# 8. Create research charts

# Chart 1: MAE
plt.figure(figsize=(10, 5))
plt.bar(results_df["Experiment"], results_df["MAE"])
plt.title("Model Comparison: MAE")
plt.ylabel("MAE (lower is better)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("mae_comparison.png", dpi=150)
plt.close()

# Chart 2: RMSE
plt.figure(figsize=(10, 5))
plt.bar(results_df["Experiment"], results_df["RMSE"])
plt.title("Model Comparison: RMSE")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("rmse_comparison.png", dpi=150)
plt.close()

# Chart 3: R-squared
plt.figure(figsize=(10, 5))
plt.bar(results_df["Experiment"], results_df["R2"])
plt.title("Model Comparison: R-squared")
plt.ylabel("R-squared (higher is generally better)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("r2_comparison.png", dpi=150)
plt.close()

print("All three charts saved successfully!")



# ==========================================
# EXPERIMENT 2: TESTING DIFFERENT TIME PERIODS
# ==========================================

print("\n=== EXPERIMENT 2: TIME PERIOD TESTING ===")

# Ensure the data is in chronological order
time_data = data.sort_values(
    ["dteday", "hr"]
).reset_index(drop=True)

features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed"
]

X_time = time_data[features]
y_time = time_data["cnt"]

# Divide the timeline into 60%, 20%, and 20%
n = len(time_data)
split_60 = int(n * 0.60)
split_80 = int(n * 0.80)

# Test period 1:
# Train on the first 60%, test on the next 20%
X_train_1 = X_time.iloc[:split_60]
y_train_1 = y_time.iloc[:split_60]

X_test_1 = X_time.iloc[split_60:split_80]
y_test_1 = y_time.iloc[split_60:split_80]

model_1 = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

mae_1, rmse_1, r2_1 = evaluate_model(
    model_1,
    X_train_1,
    X_test_1,
    y_train_1,
    y_test_1
)

# Test period 2:
# Train on the first 80%, test on the final 20%
X_train_2 = X_time.iloc[:split_80]
y_train_2 = y_time.iloc[:split_80]

X_test_2 = X_time.iloc[split_80:]
y_test_2 = y_time.iloc[split_80:]

model_2 = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

mae_2, rmse_2, r2_2 = evaluate_model(
    model_2,
    X_train_2,
    X_test_2,
    y_train_2,
    y_test_2
)

# Display results
period_results = pd.DataFrame({
    "Test Period": [
        "Middle 20% of timeline",
        "Final 20% of timeline"
    ],
    "Training Data (%)": [60, 80],
    "MAE": [mae_1, mae_2],
    "RMSE": [rmse_1, rmse_2],
    "R2": [r2_1, r2_2]
})

print(period_results.round(3).to_string(index=False))

# Save results
period_results.to_csv(
    "time_period_results.csv",
    index=False
)

# Create a chart comparing MAE
plt.figure(figsize=(9, 5))
plt.bar(
    period_results["Test Period"],
    period_results["MAE"]
)
plt.title("Prediction Error Across Time Periods")
plt.ylabel("MAE (lower is better)")
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.savefig("time_period_mae.png", dpi=150)
plt.close()

print("\nExperiment 2 completed!")
print("Saved: time_period_results.csv")
print("Saved: time_period_mae.png")


# FINAL COMPARISON EXPERIMENT

print("\n=== FINAL COMPARISON ===")

# Load the results from our previous experiment
main_results = pd.read_csv("model_results.csv")

# Select the Random Forest results
random_row = main_results[
    main_results["Experiment"] == "Random Forest - Random Split"
].iloc[0]

time_row = main_results[
    main_results["Experiment"] == "Random Forest - Time Split"
].iloc[0]

# Calculate the percentage increase in MAE
mae_increase = (
    (time_row["MAE"] - random_row["MAE"])
    / random_row["MAE"]
) * 100

# Create the final comparison table
final_summary = pd.DataFrame({
    "Evaluation": ["Random split", "Chronological time split"],
    "MAE": [random_row["MAE"], time_row["MAE"]],
    "RMSE": [random_row["RMSE"], time_row["RMSE"]],
    "R2": [random_row["R2"], time_row["R2"]]
})

print(final_summary.round(3).to_string(index=False))

print(
    f"\nTime-split MAE is {mae_increase:.1f}% "
    "higher than random-split MAE."
)

# Save the comparison table
final_summary.to_csv("final_comparison.csv", index=False)

# Create and save a comparison chart
plt.figure(figsize=(8, 5))
plt.bar(
    final_summary["Evaluation"],
    final_summary["MAE"]
)
plt.title("Random Split vs Chronological Time Split")
plt.ylabel("MAE (lower is better)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("final_mae_comparison.png", dpi=150)
plt.close()

print("\nFinal comparison table and chart saved successfully.")




# ==========================================
# EXPERIMENT 3: DAILY WALK-FORWARD EVALUATION
# ==========================================

print("\n=== DAILY WALK-FORWARD EVALUATION ===")

time_data = data.sort_values(
    ["dteday", "hr"]
).reset_index(drop=True)

# Split unique dates, not individual hourly records
unique_dates = time_data["dteday"].drop_duplicates().to_numpy()

tscv = TimeSeriesSplit(n_splits=5)
daily_results = []

for fold, (train_idx, test_idx) in enumerate(
    tscv.split(unique_dates), start=1
):
    train_dates = unique_dates[train_idx]
    test_dates = unique_dates[test_idx]

    train_data = time_data[
        time_data["dteday"].isin(train_dates)
    ]

    test_data = time_data[
        time_data["dteday"].isin(test_dates)
    ]

    X_train = train_data[features]
    y_train = train_data["cnt"]

    X_test = test_data[features]
    y_test = test_data["cnt"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    mae, rmse, r2 = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )

    daily_results.append({
        "Fold": fold,
        "Train_Start": train_dates[0],
        "Train_End": train_dates[-1],
        "Test_Start": test_dates[0],
        "Test_End": test_dates[-1],
        "Training_Rows": len(train_data),
        "Testing_Rows": len(test_data),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

daily_df = pd.DataFrame(daily_results)

print("\n=== DAILY WALK-FORWARD RESULTS ===")
print(daily_df.round(3).to_string(index=False))

daily_df.to_csv(
    "daily_walk_forward_results.csv",
    index=False
)

print("\nSaved: daily_walk_forward_results.csv")

# Average performance across all 5 folds
print("\n=== AVERAGE DAILY WALK-FORWARD PERFORMANCE ===")

print("Average MAE:", round(daily_df["MAE"].mean(), 3))
print("Average RMSE:", round(daily_df["RMSE"].mean(), 3))
print("Average R2:", round(daily_df["R2"].mean(), 3))

# Plot MAE for each fold
plt.figure(figsize=(8, 5))

plt.plot(
    daily_df["Fold"],
    daily_df["MAE"],
    marker="o"
)

plt.xticks(daily_df["Fold"])
plt.xlabel("Fold")
plt.ylabel("MAE (Bikes)")
plt.title("Daily Walk-Forward Evaluation: MAE by Fold")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("daily_walk_forward_mae.png", dpi=150)
plt.close()

print("Saved: daily_walk_forward_mae.png")

# ==========================================
# EXPERIMENT 4: HOUR-BY-HOUR ERROR ANALYSIS
# ==========================================

print("\n=== EXPERIMENT 4: HOUR-BY-HOUR ERROR ANALYSIS ===")

# Use the final 20% of chronological data as unseen test data
split_index = int(len(time_data) * 0.80)

train_data = time_data.iloc[:split_index]
test_data = time_data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["cnt"]

X_test = test_data[features]
y_test = test_data["cnt"]

# Train the model using earlier data only
hour_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

hour_model.fit(X_train, y_train)

# Predict bike rentals for the test period
predictions = hour_model.predict(X_test)

# Create a table of actual and predicted values
hour_analysis = pd.DataFrame({
    "Hour": test_data["hr"].to_numpy(),
    "Actual": y_test.to_numpy(),
    "Predicted": predictions
})

# Calculate absolute prediction error
hour_analysis["Absolute Error"] = (
    hour_analysis["Actual"] - hour_analysis["Predicted"]
).abs()

# Calculate average error separately for each hour
hour_results = (
    hour_analysis.groupby("Hour")
    .agg(
        Average_Actual=("Actual", "mean"),
        Average_Predicted=("Predicted", "mean"),
        MAE=("Absolute Error", "mean"),
        Number_of_Records=("Actual", "size")
    )
    .reset_index()
)

print("\n=== ERROR FOR EACH HOUR ===")
print(hour_results.round(2).to_string(index=False))

# Save the results
hour_results.to_csv("hourly_error_results.csv", index=False)

# Plot hourly prediction errors
plt.figure(figsize=(10, 5))
plt.bar(hour_results["Hour"], hour_results["MAE"])
plt.title("Prediction Error by Hour of Day")
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("MAE (lower is better)")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig("hourly_error_chart.png", dpi=150)
plt.close()

# Identify hours with the highest and lowest errors
highest_error = hour_results.loc[hour_results["MAE"].idxmax()]
lowest_error = hour_results.loc[hour_results["MAE"].idxmin()]

print("\nHighest-error hour:", int(highest_error["Hour"]))
print("Highest MAE:", round(highest_error["MAE"], 2))

print("\nLowest-error hour:", int(lowest_error["Hour"]))
print("Lowest MAE:", round(lowest_error["MAE"], 2))

print("\nExperiment 4 completed!")
print("Saved: hourly_error_results.csv")
print("Saved: hourly_error_chart.png")



# ==========================================
# EXPERIMENT 5: MODEL CONFIGURATION TEST
# ==========================================

print("\n=== EXPERIMENT 5: MODEL CONFIGURATION TEST ===")

# Use the same chronological training and testing data
split_index = int(len(time_data) * 0.80)

X_train = time_data[features].iloc[:split_index]
X_test = time_data[features].iloc[split_index:]

y_train = time_data["cnt"].iloc[:split_index]
y_test = time_data["cnt"].iloc[split_index:]

# Test different Random Forest configurations
configurations = {
    "RF_50_Tree": RandomForestRegressor(
        n_estimators=50,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    "RF_100_Tree": RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    "RF_200_Tree": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    "RF_Depth_10": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

config_results = []

for name, model in configurations.items():
    mae, rmse, r2 = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )

    config_results.append({
        "Configuration": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

config_df = pd.DataFrame(config_results)

print("\n=== CONFIGURATION COMPARISON ===")
print(config_df.round(3).to_string(index=False))

config_df.to_csv("configuration_results.csv", index=False)

# Compare MAE across configurations
plt.figure(figsize=(9, 5))
plt.bar(config_df["Configuration"], config_df["MAE"])
plt.title("Random Forest Configuration Comparison")
plt.xlabel("Model Configuration")
plt.ylabel("MAE (lower is better)")
plt.tight_layout()
plt.savefig("configuration_comparison.png", dpi=150)
plt.close()

print("\nExperiment 5 completed!")
print("Saved: configuration_results.csv")
print("Saved: configuration_comparison.png")



# ==========================================
# EXPERIMENT 6: INVESTIGATE FOLD 3
# ==========================================

print("\n=== INVESTIGATING WALK-FORWARD FOLDS ===")

time_data = data.sort_values(
    ["dteday", "hr"]
).reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(
    tscv.split(time_data), start=1
):
    test_period = time_data.iloc[test_idx]

    start_date = test_period["dteday"].iloc[0]
    end_date = test_period["dteday"].iloc[-1]

    average_rentals = test_period["cnt"].mean()
    maximum_rentals = test_period["cnt"].max()

    print(f"\nFold {fold}")
    print("Test start date:", start_date)
    print("Test end date:", end_date)
    print("Average actual rentals:", round(average_rentals, 2))
    print("Maximum actual rentals:", maximum_rentals)



# ==========================================
# EXPERIMENT 6: WEATHER FEATURE COMPARISON
# ==========================================

print("\n=== WEATHER FEATURE COMPARISON ===")

# Sort data chronologically
weather_data = data.sort_values(
    ["dteday", "hr"]
).reset_index(drop=True)

# Features including weather information
features_with_weather = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed"
]

# Features excluding weather information
features_without_weather = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday"
]

# Use the same chronological 80/20 split
split_point = int(len(weather_data) * 0.8)

train_data = weather_data.iloc[:split_point]
test_data = weather_data.iloc[split_point:]

y_train = train_data["cnt"]
y_test = test_data["cnt"]

weather_comparison = []

# Test 1: With weather features
model_with_weather = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

mae, rmse, r2 = evaluate_model(
    model_with_weather,
    train_data[features_with_weather],
    test_data[features_with_weather],
    y_train,
    y_test
)

weather_comparison.append({
    "Model": "With_Weather",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
})

# Test 2: Without weather features
model_without_weather = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

mae, rmse, r2 = evaluate_model(
    model_without_weather,
    train_data[features_without_weather],
    test_data[features_without_weather],
    y_train,
    y_test
)

weather_comparison.append({
    "Model": "Without_Weather",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
})

weather_df = pd.DataFrame(weather_comparison)

print("\n=== WEATHER COMPARISON RESULTS ===")
print(weather_df.round(3).to_string(index=False))

weather_df.to_csv(
    "weather_comparison_results.csv",
    index=False
)

print("\nSaved: weather_comparison_results.csv")



# ==========================================
# EXPERIMENT 7: FEATURE IMPORTANCE ANALYSIS
# ==========================================

print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

importance_df = pd.DataFrame({
    "Feature": features_with_weather,
    "Importance": model_with_weather.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.round(4).to_string(index=False))

importance_df.to_csv(
    "feature_importance_results.csv",
    index=False
)

# Plot feature importance
plt.figure(figsize=(9, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance_chart.png", dpi=150)
plt.close()

print("\nSaved: feature_importance_results.csv")
print("Saved: feature_importance_chart.png")
