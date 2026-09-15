import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
latitude = 14.82
longitude = 78.28
start_date = "2020-05-15"
end_date = "2020-06-17"
timezone = "Asia/Kolkata"
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": timezone
}
print("================================================")
print("TASK 3 – DOWNLOAD PUBLIC WEATHER DATA")
print("================================================")
response = requests.get(url, params=params)
response.raise_for_status()
weather_data = response.json()
print("\nWeather data downloaded successfully.")
print("\nReturned latitude:",
      weather_data["latitude"])
print("Returned longitude:",
      weather_data["longitude"])
print("Returned timezone:",
      weather_data["timezone"])
hourly_data = weather_data["hourly"]
weather_df = pd.DataFrame({
    "datetime": hourly_data["time"],
    "sw_radiation": hourly_data["shortwave_radiation"],
    "temp_2m": hourly_data["temperature_2m"],
    "cloud_cover": hourly_data["cloud_cover"]
})
weather_df["datetime"] = pd.to_datetime(
    weather_df["datetime"]
)
print("\nDownloaded weather data:")
print(weather_df.head())
print("\nNumber of Open-Meteo rows:",
      len(weather_df))
print("\nWeather data columns:")
print(weather_df.columns.tolist())
output_file = DATA / "plant1 openmeteo.csv"
weather_df.to_csv(
    output_file,
    index=False
)
print("\nOpen-Meteo data saved to:")
print(output_file)
plant_df = pd.read_csv(
    DATA / "plant1 hourly.csv"
)
plant_df["datetime"] = pd.to_datetime(
    plant_df["datetime"]
)
merged_df = pd.merge(
    plant_df,
    weather_df,
    on="datetime",
    how="inner"
)
print("\n================================================")
print("MERGED DATA")
print("================================================")
print("\nMerged data:")
print(merged_df.head())
print("\nNumber of merged rows:",
      len(merged_df))
merged_df["sw_radiation_kw_m2"] = (
    merged_df["sw_radiation"] / 1000
)
print("\nRadiation units:")
print("Sensor irradiation = kW/m²")
print("Open-Meteo radiation = W/m²")
print("Converted Open-Meteo radiation = kW/m²")
selected_days = [
    "2020-05-15",
    "2020-05-16",
    "2020-05-17"
]
print("\n================================================")
print("TASK 3.4 – LOCATION VERIFICATION")
print("================================================")
peak_comparison = []
for day in selected_days:
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    day_data = merged_df[
        (merged_df["datetime"] >= day_start)
        &
        (merged_df["datetime"] < day_end)
    ].copy()
    plt.figure(figsize=(10, 5))
    plt.plot(
        day_data["datetime"],
        day_data["IRRADIATION"],
        marker="o",
        label="Sensor Irradiation"
    )
    plt.plot(
        day_data["datetime"],
        day_data["sw_radiation_kw_m2"],
        marker="o",
        label="Open-Meteo Radiation"
    )
    plt.xlabel("Datetime")
    plt.ylabel("Radiation (kW/m²)")
    plt.title(
        "Sensor vs Open-Meteo Radiation - " + day
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    figure_name = (
        "task3_4_radiation_comparison_"
        + day
        + ".png"
    )
    plt.savefig(
        FIGURES / figure_name,
        dpi=300
    )
    plt.show()
    sensor_peak_index = (
        day_data["IRRADIATION"].idxmax()
    )
    sensor_peak_time = (
        day_data.loc[
            sensor_peak_index,
            "datetime"
        ]
    )
    weather_peak_index = (
        day_data["sw_radiation_kw_m2"].idxmax()
    )

    weather_peak_time = (
        day_data.loc[
            weather_peak_index,
            "datetime"
        ]
    )

    difference = abs(
        (
            sensor_peak_time - weather_peak_time
        ).total_seconds()
    ) / 3600
    print("\nDate:", day)
    print(
        "Sensor peak hour:",
        sensor_peak_time.strftime("%H:%M")
    )
    print(
        "Open-Meteo peak hour:",
        weather_peak_time.strftime("%H:%M")
    )
    print(
        "Peak time difference:",
        difference,
        "hours"
    )
    peak_comparison.append({
        "Date": day,
        "Sensor peak hour":
            sensor_peak_time.strftime("%H:%M"),
        "Open-Meteo peak hour":
            weather_peak_time.strftime("%H:%M"),
        "Peak time difference (hours)":
            difference
    })
print("\n================================================")
print("UPDATING TABLE 1")
print("================================================")
table1_file = TABLES / "table1_data_preparation.csv"
if table1_file.exists():

    table1 = pd.read_csv(table1_file)
else:
    table1 = pd.DataFrame(
        columns=[
            "Item",
            "Value",
            "Sensor peak hour",
            "Open-Meteo peak hour",
            "Peak time difference (hours)"
        ]
    )
required_columns = [
    "Item",
    "Value",
    "Sensor peak hour",
    "Open-Meteo peak hour",
    "Peak time difference (hours)"
]
for column in required_columns:

    if column not in table1.columns:
        table1[column] = ""
table1 = table1[
    table1["Item"] != "Open-Meteo rows downloaded"
].copy()
openmeteo_row = pd.DataFrame([{
    "Item": "Open-Meteo rows downloaded",
    "Value": len(weather_df),
    "Sensor peak hour": "",
    "Open-Meteo peak hour": "",
    "Peak time difference (hours)": ""
}])
table1 = pd.concat(
    [table1, openmeteo_row],
    ignore_index=True
)
table1 = table1[
    ~table1["Item"].str.startswith(
        "Peak hour comparison",
        na=False
    )
].copy()
for result in peak_comparison:
    peak_row = pd.DataFrame([{
        "Item":
            "Peak hour comparison " + result["Date"],
        "Value": "",
        "Sensor peak hour":
            result["Sensor peak hour"],
        "Open-Meteo peak hour":
            result["Open-Meteo peak hour"],
        "Peak time difference (hours)":
            result["Peak time difference (hours)"]
    }])
    table1 = pd.concat(
        [table1, peak_row],
        ignore_index=True
    )
table1.to_csv(
    table1_file,
    index=False
)
print("\nTable 1 updated successfully.")
print("Saved to:")
print(table1_file)
print("\n================================================")
print("TASK 3 COMPLETED")
print("================================================")
print("\nFile created:")
print(DATA / "plant1 openmeteo.csv")
print("\nFigures created in:")
print(FIGURES)
print("\nTable 1 created/updated:")
print(table1_file)
