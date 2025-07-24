import numpy as np, xgboost as xgb, joblib, json, pathlib

X = np.load("rank_features.npy")
y = np.load("rank_labels.npy")
groups = np.load("rank_group.npy")

dtrain = xgb.DMatrix(X, label=y)
dtrain.set_group(groups)

params = {
    "objective": "rank:pairwise",
    "eta": 0.1,
    "max_depth": 6,
    "seed": 42,
}

watchlist = [(dtrain, "train")]

model = xgb.train(params,dtrain,num_boost_round=300)

joblib.dump(model, "models/model_ranker.pkl")

best_ndcg = model.best_score if hasattr(model, "best_score") else "N/A"
print(f"\nFinal ndcg@1 = {best_ndcg}")
print("Ranker training finished to models/model_ranker.pkl")