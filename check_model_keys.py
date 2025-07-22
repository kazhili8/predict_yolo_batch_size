import joblib, pprint
mdl = joblib.load("models/model.pkl")
pprint.pprint(mdl.keys())
