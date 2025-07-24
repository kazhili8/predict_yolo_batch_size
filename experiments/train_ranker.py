import numpy as np, xgboost as xgb, joblib

X = np.load("rank_features.npy")
y = np.load("rank_labels.npy")
group = np.load("rank_group.npy")

dtrain = xgb.DMatrix(X, label=y)
dtrain.set_group(group)

params = {
    "objective": "rank:pairwise",
    "eta": 0.1,
    "max_depth": 6,
    "eval_metric": "ndcg",
    "verbosity": 1,
    "seed": 42
}
model = xgb.train(params, dtrain, num_boost_round=300)
joblib.dump(model, "models/model_ranker.pkl")
print("Ranker trainning has finished → models/model_ranker.pkl")