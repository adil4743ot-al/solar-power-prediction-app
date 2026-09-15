from pathlib import Path
import pandas as pd
from load_data import load_raw
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
TABLES = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
raw = load_raw()
gen1 = raw["gen1"].copy()
sensor1 = raw["sensor1"].copy()
raw_generation_rows = len(gen1)
raw_sensor_rows = len(sensor1)
print("================================================")
print("TASK 1.2 – DATE TIME CONVERSION")
print("================================================")
print("\nRaw Generation DATE_TIME:")
print(gen1["DATE_TIME"].head())
print("\nRaw Sensor DATE_TIME:")
print(sensor1["DATE_TIME"].head())
gen1["datetime"] = pd.to_datetime(
    gen1["DATE_TIME"],
    format="%d-%m-%Y %H:%M"
)
sensor1["datetime"] = pd.to_datetime(
    sensor1["DATE_TIME"],
    format="%Y-%m-%d %H:%M:%S"
)
print("\nPlant 1 Generation Data")
print("First timestamp:", gen1["datetime"].min())
print("Last timestamp :", gen1["datetime"].max())
print("\nPlant 1 Weather Sensor Data")
print("First timestamp:", sensor1["datetime"].min())
print("Last timestamp :", sensor1["datetime"].max())
print("\nGeneration datetime datatype:")
print(gen1["datetime"].dtype)
print("\nSensor datetime datatype:")
print(sensor1["datetime"].dtype)
print("\n================================================")
print("TASK 1.3 – PLANT LEVEL POWER")
print("================================================")
plant_power = (
    gen1.groupby("datetime")[["AC_POWER", "DC_POWER"]]
    .sum()
    .reset_index()
)
print("\nPlant-level power:")
print(plant_power.head())
print(
    "\nNumber of generation timestamps:",
    len(plant_power)
)
print("\n================================================")
print("TASK 1.4 – MERGE GENERATION AND SENSOR DATA")
print("================================================")
sensor_data = sensor1[
    [
        "datetime",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
].copy()
merged = pd.merge(
    plant_power,
    sensor_data,
    on="datetime",
    how="outer",
    indicator=True
)
generation_only = (
    merged["_merge"] == "left_only"
).sum()
sensor_only = (
    merged["_merge"] == "right_only"
).sum()
both_files = (
    merged["_merge"] == "both"
).sum()
timestamps_only_one_file = (
    generation_only + sensor_only
)
print(
    "\nTimestamps in both files:",
    both_files
)
print(
    "Timestamps only in generation file:",
    generation_only
)
print(
    "Timestamps only in sensor file:",
    sensor_only
)
print(
    "Timestamps in only one file:",
    timestamps_only_one_file
)
merged = merged.drop(columns=["_merge"])
merged = merged.sort_values("datetime")
print("\nMerged data:")
print(merged.head())

print("\n================================================")
print("TASK 1.5 – HOURLY RESAMPLING")
print("================================================")
merged = merged.set_index("datetime")
hourly = merged.resample("1h").mean()
hourly = hourly.reset_index()
hourly = hourly[
    [
        "datetime",
        "AC_POWER",
        "DC_POWER",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
]
hourly_rows_after_resampling = len(hourly)
print("\nHourly data:")
print(hourly.head())
print(
    "\nNumber of hourly rows:",
    hourly_rows_after_resampling
)
print("\n================================================")
print("TASK 1.6 – MISSING VALUES")
print("================================================")
print(
    "\nMissing values in each column BEFORE handling:"
)
print(hourly.isna().sum())
missing_rows = int(
    hourly.isna().any(axis=1).sum()
)
hourly_rows_with_missing = missing_rows
print(
    "\nHourly rows with missing values:",
    hourly_rows_with_missing
)

if missing_rows > 0:
    hourly = hourly.dropna().reset_index(drop=True)
    print("\nMissing rows were removed.")
else:
    print("\nNo missing values found.")
    print("No rows were removed.")
output_file = DATA / "plant1 hourly.csv"
hourly.to_csv(
    output_file,
    index=False
)
final_hourly_rows = len(hourly)

final_missing_rows = int(
    hourly.isna().any(axis=1).sum()
)
print("\n================================================")
print("TASK 1 COMPLETED")
print("================================================")
print(
    "Final hourly rows:",
    final_hourly_rows
)
print(
    "Final missing-value rows:",
    final_missing_rows
)
print("\nFinal columns:")
print(hourly.columns.tolist())
print("\nFile saved successfully:")
print(output_file)
print("\n================================================")
print("GENERATING TABLE 1")
print("================================================")
table1 = pd.DataFrame({
    "Item": [
        "Raw generation rows (Plant 1)",
        "Raw sensor rows (Plant 1)",
        "Timestamps in only one file",
        "Hourly rows after resampling",
        "Hourly rows with missing values"
    ],
    "Value": [
        raw_generation_rows,
        raw_sensor_rows,
        timestamps_only_one_file,
        hourly_rows_after_resampling,
        hourly_rows_with_missing
    ]
})
table1_file = TABLES / "table1_data_preparation.csv"
table1.to_csv(
    table1_file,
    index=False
)
print("\nTable 1 saved successfully:")
print(table1_file)
print("\n-----------------------------")
print("TABLE 1 - DATA PREPARATION")
print("-----------------------------")
print(table1.to_string(index=False))
print("\n================================================")
print("TABLE 1 TASK 1 PART COMPLETED")
print("================================================")
