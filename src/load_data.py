from pathlib import Path
import pandas as pd

FILES = {
    "gen1": "Plant_1_Generation_Data.csv",
    "sensor1": "Plant_1_Weather_Sensor_Data.csv",
}
def load_raw(data_dir=None):
    """Return a dict of the raw DataFrames.
    Keys: 'gen1', 'sensor1'.
    DATE_TIME is left as a plain string on purpose.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    frames = {}
    for key, name in FILES.items():
        path = data_dir / name

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the dataset first."
            )
        frames[key] = pd.read_csv(path)
    return frames
if __name__ == "__main__":
    raw = load_raw()
    for key, df in raw.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head(3).to_string(), end="\n\n")
