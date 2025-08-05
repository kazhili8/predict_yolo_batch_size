import argparse
import pandas as pd
from experiments.config import FEATURES_CSV

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=FEATURES_CSV)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.features)
    print("shape:", df.shape)
    print(df.head())
    print("\nnull ratio:\n", df.isna().mean())
    print("\nby group counts:\n", df.groupby(["model", "epochs", "tag"]).size())

if __name__ == "__main__":
    main()
