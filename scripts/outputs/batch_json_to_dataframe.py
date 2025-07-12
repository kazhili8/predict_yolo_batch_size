import json
import pandas as pd
from pathlib import Path
import argparse

def json_to_df(json_file):
    "Read data from a JSON file and convert it to a DataFrame"
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    power_series = data.get("power_series", [])

    df = pd.DataFrame(steps)
    if len(power_series) >= len(df):
        df["power"] = power_series[:len(df)]
    else:
        df["power"] = [None] * len(df)

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        required=True,
        help="The folder path containing all JSON files, such as D:\\... \\outputs"
    )
    args = parser.parse_args()

    input_dir = Path(args.json_dir)
    output_dir = input_dir / "dataframe"
    output_dir.mkdir(exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print("No.json file was found!")
        return

    for json_file in json_files:
        try:
            df = json_to_df(json_file)
            out_path = output_dir / (json_file.stem + ".csv")
            df.to_csv(out_path, index=False)
            print(f"Converted:{json_file.name} → {out_path.name}")
        except Exception as e:
            print(f"Error handling file{json_file.name}: {e}")

if __name__ == "__main__":
    main()
