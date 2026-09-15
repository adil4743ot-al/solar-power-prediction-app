from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
file_path = DATA / "plant1 hourly.csv"
df = pd.read_csv(file_path)
df["datetime"] = pd.to_datetime(df["datetime"])
print("================================================")
print("TASK 2 – EXPLORATORY PLOTS")
print("================================================")
print("\nHourly data:")
print(df.head())
print("\nNumber of rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())
plt.figure(figsize=(8, 5))
plt.scatter(
    df["IRRADIATION"],
    df["AC_POWER"],
    alpha=0.6
)
plt.xlabel("Irradiation (kW/m²)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Irradiation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "2_1_ac_power_vs_irradiation.png",
    dpi=300
)
plt.show()
plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    df["AMBIENT_TEMPERATURE"],
    df["MODULE_TEMPERATURE"],
    c=df["IRRADIATION"],
    alpha=0.7
)
plt.xlabel("Ambient Temperature (°C)")
plt.ylabel("Module Temperature (°C)")
plt.title("Module Temperature vs Ambient Temperature")
plt.colorbar(
    scatter,
    label="Irradiation (kW/m²)"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "2_2_module_temp_vs_ambient_temp.png",
    dpi=300
)
plt.show()
plt.scatter(
    df["DC_POWER"],
    df["AC_POWER"],
    alpha=0.6
)
plt.xlabel("DC Power (kW)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs DC Power")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "2_3_ac_power_vs_dc_power.png",
    dpi=300
)
plt.show()
df["hour"] = df["datetime"].dt.hour
hourly_ac_power = (
    df.groupby("hour")["AC_POWER"]
    .mean()
)
hourly_ac_power = hourly_ac_power.reindex(range(24))
plt.figure(figsize=(8, 5))
plt.plot(
    hourly_ac_power.index,
    hourly_ac_power.values,
    marker="o"
)
plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power (kW)")
plt.title("Average AC Power for Each Hour of the Day")
plt.xticks(range(24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "2_4_average_ac_power_by_hour.png",
    dpi=300
)
plt.show()
print("\n================================================")
print("AVERAGE AC POWER BY HOUR")
print("================================================")

print(hourly_ac_power)


print("\n================================================")
print("TASK 2 COMPLETED")
print("================================================")

print("\nFour figures have been saved in:")
print(FIGURES)
