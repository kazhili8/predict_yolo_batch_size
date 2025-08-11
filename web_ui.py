import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib, pickle, subprocess, sys, time, io, os
from pathlib import Path

def pick_map_col(df):
    for c in ["map50","avg_map50","mAP50","mAP@0.5"]:
        if c in df.columns:
            return c
    raise RuntimeError("no mAP column")

def ensure_feats(df, pref):
    cols = list(pref) if pref else [
        "throughput","inv_step_time","avg_power","avg_mem",
        "pwr_mean","pwr_std","pwr_cv","energy_per_img",
        "eff_tp_watt","eff_tp_mem","throughput_var_ratio","vram_total_mb"
    ]
    feats = [c for c in cols if c in df.columns]
    if not feats:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feats = [c for c in num_cols if c not in {"batch","n_logs"}]
    return feats

def unwrap_model_object(obj):
    m = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if isinstance(m, xgb.Booster):
        kind = "xgb_booster"
    elif hasattr(m, "get_booster") and callable(getattr(m, "get_booster")):
        kind = "xgb_sklearn"
    elif "lightgbm" in type(m).__module__ or isinstance(m, lgb.LGBMRanker):
        kind = "lgbm"
    else:
        raise RuntimeError("unsupported model object")
    feats = None
    if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], (list, tuple)):
        feats = list(obj["features"])
    return (m, kind, feats)

def load_models_list(pkl1, pkl2=""):
    models = []
    feats = None
    for p in [pkl1, pkl2] if pkl2 else [pkl1]:
        if not p:
            continue
        try:
            obj = joblib.load(p)
        except Exception:
            obj = pickle.load(open(p, "rb"))
        m, kind, f = unwrap_model_object(obj)
        models.append((m, kind))
        if feats is None and f is not None:
            feats = f
    return models, feats

def predict_one(model_tuple, X_df, feats):
    m, kind = model_tuple
    X_pd = X_df[feats] if feats is not None else X_df
    if kind == "xgb_booster":
        dm = xgb.DMatrix(X_pd.values, feature_names=(feats if feats is not None else None))
        return m.predict(dm)
    elif kind == "xgb_sklearn":
        try:
            dm = xgb.DMatrix(X_pd.values, feature_names=(feats if feats is not None else None))
            return m.predict(dm)
        except Exception:
            return m.predict(X_pd.values)
    else:
        return m.predict(X_pd)

def make_true_score(df, weights, oracle, map_col):
    if oracle == "throughput":
        return df["throughput"].astype(float)
    T,P,M,A = weights
    dmap = df[map_col].max() - df[map_col]
    return T*df["throughput"] - P*df["avg_power"] - M*df["avg_mem"] - A*dmap

def feasible_mask(df, cap_w, delta_map, map_col):
    if cap_w is None:
        return pd.Series(True, index=df.index)
    b1 = df[df["batch"]==1]
    if len(b1)==0:
        return pd.Series(False, index=df.index)
    base = float(b1[map_col].iloc[0])
    return (df["pwr_mean"]<=cap_w) & (df[map_col] >= base*(1.0-delta_map))

def recommend_table(df, models, feats, cap_w, delta_map, weights, oracle):
    map_col = pick_map_col(df)
    feats_use = [c for c in ensure_feats(df, feats) if c in df.columns]
    rows = []
    for key, g0 in df.groupby(["model","epochs","tag"], sort=False):
        need = list(set(feats_use + [map_col,"avg_power","avg_mem","throughput","pwr_mean","batch"]))
        g = g0.dropna(subset=[c for c in need if c in g0.columns]).copy()
        if g.empty:
            continue
        fm = feasible_mask(g, cap_w, delta_map, map_col)
        g = g[fm].copy()
        if g.empty:
            continue
        preds = []
        for mt in models:
            preds.append(predict_one(mt, g[feats_use], feats_use))
        if len(preds) == 1:
            g["pred_score"] = preds[0]
        else:
            g["pred_score"] = np.mean(np.column_stack(preds), axis=1)
        g["oracle_score"] = make_true_score(g, weights, oracle, map_col)
        b_pred = int(g.sort_values("pred_score", ascending=False)["batch"].iloc[0])
        b_opt = int(g.sort_values("oracle_score", ascending=False)["batch"].iloc[0])
        rows.append({"model": key[0],"epochs": key[1],"tag": key[2],"batch_pred": b_pred,"batch_oracle": b_opt})
    out = pd.DataFrame(rows)
    return out

def eval_policies(df, recs, cap_w, delta_map, weights, oracle, seed=0):
    rng = np.random.default_rng(seed)
    map_col = pick_map_col(df)
    rows = []
    for key, g0 in df.groupby(["model","epochs","tag"], sort=False):
        g = g0.copy()
        fm = feasible_mask(g, cap_w, delta_map, map_col)
        g = g[fm].copy()
        if g.empty:
            continue
        g["oracle_score"] = make_true_score(g, weights, oracle, map_col)
        b_orc = int(g.sort_values("oracle_score", ascending=False)["batch"].iloc[0])
        if "eff_tp_watt" in g.columns:
            b_greedy = int(g.sort_values("eff_tp_watt", ascending=False)["batch"].iloc[0])
        else:
            b_greedy = int(g.sort_values("throughput", ascending=False)["batch"].iloc[0])
        b_zeus = int(g.sort_values("throughput", ascending=False)["batch"].iloc[0])
        b_rand = int(rng.choice(g["batch"].astype(int).values))
        line = recs[(recs["model"]==key[0])&(recs["epochs"]==key[1])&(recs["tag"]==key[2])]
        if len(line)>0:
            b_tab = int(line["batch_pred"].iloc[0])
        else:
            b_tab = b_greedy
        s_orc = float(g[g["batch"]==b_orc]["oracle_score"].iloc[0])
        def s_of(b):
            return float(g[g["batch"]==b]["oracle_score"].iloc[0])
        rows.append({"model":key[0],"epochs":key[1],"tag":key[2],
                     "top1_table":float(b_tab==b_orc),"top1_zeus":float(b_zeus==b_orc),
                     "top1_greedy":float(b_greedy==b_orc),"top1_random":float(b_rand==b_orc),
                     "regret_table":s_orc - s_of(b_tab),"regret_zeus":s_orc - s_of(b_zeus),
                     "regret_greedy":s_orc - s_of(b_greedy),"regret_random":s_orc - s_of(b_rand),
                     "vio_table":0.0,"vio_zeus":0.0,"vio_greedy":0.0,"vio_random":0.0})
    out = pd.DataFrame(rows)
    return out

def agg_means(df):
    out = {}
    for who in ["table","zeus","greedy","random"]:
        out[f"Top1_{who}"] = float(df[f"top1_{who}"].mean()) if len(df)>0 else np.nan
        out[f"Regret_{who}"] = float(df[f"regret_{who}"].mean()) if len(df)>0 else np.nan
        out[f"Violation_{who}"] = float(df[f"vio_{who}"].mean()) if len(df)>0 else np.nan
    return out

st.set_page_config(page_title="Batch Recommender", layout="wide")
st.title("YOLO Batch Recommender")

tab1, tab2 = st.tabs(["Offline CSV","Probe (one-click)"])

with tab1:
    with st.sidebar:
        uploaded = st.file_uploader("features CSV", type=["csv"])
        default_csv = st.text_input("or CSV path", "scripts/outputs/dataframe/features_v7_agg.csv")
        oracle = st.selectbox("oracle", ["score","throughput"], index=0)
        colw = st.columns(4)
        wT = colw[0].number_input("T", value=0.60, step=0.01, format="%.2f")
        wP = colw[1].number_input("P", value=0.20, step=0.01, format="%.2f")
        wM = colw[2].number_input("M", value=0.10, step=0.01, format="%.2f")
        wA = colw[3].number_input("A", value=0.10, step=0.01, format="%.2f")
        cap_w = st.selectbox("cap_w", [None,65,115], index=2)
        delta_map = st.number_input("delta_map", value=0.01, step=0.005, format="%.3f")
        model_mode = st.selectbox("model", ["xgb","lgbm","ensemble"], index=2)
        xgb_pkl = st.text_input("xgb pkl", "scripts/outputs/tune_v7_w/xgb_ranker_tuned.pkl")
        lgbm_pkl = st.text_input("lgbm pkl", "scripts/outputs/tune_v7_w/lgbm_ranker_tuned.pkl")
        seed = st.number_input("seed", value=0, step=1)
        run_btn = st.button("Recommend and Evaluate")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(default_csv)
    if run_btn:
        weights = (wT,wP,wM,wA)
        if model_mode=="xgb":
            models, feats = load_models_list(xgb_pkl, "")
        elif model_mode=="lgbm":
            models, feats = load_models_list(lgbm_pkl, "")
        else:
            models, feats = load_models_list(xgb_pkl, lgbm_pkl)
        recs = recommend_table(df, models, feats, cap_w, delta_map, weights, oracle)
        st.subheader("Recommendations")
        st.dataframe(recs, use_container_width=True)
        csv_bytes = recs.to_csv(index=False).encode("utf-8")
        st.download_button("Download recommendations.csv", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")
        eval_df = eval_policies(df, recs, cap_w, delta_map, weights, oracle, seed=int(seed))
        st.subheader("Policy evaluation")
        st.dataframe(eval_df, use_container_width=True)
        agg = agg_means(eval_df)
        st.metric("N groups", len(eval_df))
        cols = st.columns(4)
        cols[0].metric("Top1 table", f"{agg['Top1_table']:.3f}")
        cols[1].metric("Top1 zeus", f"{agg['Top1_zeus']:.3f}")
        cols[2].metric("Top1 greedy", f"{agg['Top1_greedy']:.3f}")
        cols[3].metric("Top1 random", f"{agg['Top1_random']:.3f}")
        cols = st.columns(4)
        cols[0].metric("Regret table", f"{agg['Regret_table']:.3f}")
        cols[1].metric("Regret zeus", f"{agg['Regret_zeus']:.3f}")
        cols[2].metric("Regret greedy", f"{agg['Regret_greedy']:.3f}")
        cols[3].metric("Regret random", f"{agg['Regret_random']:.3f}")

with tab2:
    st.subheader("One-click: table-or-probe")
    col = st.columns(2)
    pt_path = col[0].text_input("YOLO weights (.pt)", "yolo11m.pt")
    tag_mode = col[1].selectbox("power tag", ["auto","65W","115W"], index=0)
    rec115 = st.text_input("recommendations_115W.csv", "scripts/outputs/recommendations_115W_constrained.csv")
    rec65  = st.text_input("recommendations_65W.csv",  "scripts/outputs/recommendations_65W_constrained.csv")
    go = st.button("Run")
    if go:
        t0 = time.perf_counter()
        cmd = [sys.executable, "scripts/predict_batch_size.py", "--model", pt_path, "--tag", tag_mode, "--rec115", rec115, "--rec65", rec65]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=True)
            out = p.stdout
            err = p.stderr
            took = time.perf_counter() - t0
            st.metric("Elapsed (s)", f"{took:.2f}")
            st.text_area("stdout", out, height=300)
            if err:
                st.text_area("stderr", err, height=200)
        except subprocess.CalledProcessError as e:
            took = time.perf_counter() - t0
            st.metric("Elapsed (s)", f"{took:.2f}")
            st.error("predict_batch_size.py failed")
            st.text_area("stdout", e.stdout or "", height=200)
            st.text_area("stderr", e.stderr or "", height=200)
