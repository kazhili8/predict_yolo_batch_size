import joblib, pandas as pd, matplotlib.pyplot as plt
model = joblib.load("models/model_ExtraTrees.pkl")

FEATS = list(model.feature_names_in_)

model = joblib.load("models/model_ExtraTrees.pkl")
imp = pd.Series(model.feature_importances_, index=FEATS).sort_values()

plt.figure(figsize=(6,4))
imp.plot.barh()
plt.title("ExtraTrees feature importance")
plt.tight_layout()
plt.savefig("fig_feat_importance.png", dpi=140)
print("picture saved as： fig_feat_importance.png")