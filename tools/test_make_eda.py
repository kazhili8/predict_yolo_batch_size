import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv(r"scripts/outputs/dataframe/features_v2.csv")
profile = ProfileReport(df, title="Feature EDA", minimal=True)
profile.to_file("eda_report.html")
print("EDA report generated → eda_report.html")