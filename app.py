"""
Streamlit demo for the finance-sector shock propagation model.

    streamlit run app.py

The app serves trained_models/finance_gatv2.pth (written by
`python -m experiments.run_static_comparison`). It builds the model's inputs
with the same functions used in training (data/finance_universe.py), so the
graph, features and universe match what the model was evaluated on.
"""

import os

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf

from data.finance_universe import FINANCE_TICKERS, SUPPLY_CHAIN, build_graph, node_features
from models.baselines import StaticGATv2

MODEL_PATH = "trained_models/finance_gatv2.pth"
RESULTS = "experiments/results"
FIGURES = "experiments/figures"

st.set_page_config(page_title="Shock Propagation GNN", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model = StaticGATv2(len(ckpt["feature_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data(ttl=3600, show_spinner="Downloading recent prices from Yahoo Finance...")
def recent_prices(tickers):
    out = {}
    for t in tickers:
        df = yf.Ticker(t).history(period="3mo", auto_adjust=True)
        if not df.empty:
            df.index = df.index.tz_localize(None).normalize()
            out[t] = df[["Open", "High", "Low", "Close", "Volume"]]
    return out


def read_csv(name):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def fmt(df, mean, std):
    return df[mean].map("{:.3f}".format) + " ± " + df[std].map("{:.3f}".format)


def results_table(df):
    return pd.DataFrame({
        "Model": df.model,
        "Test ROC-AUC": fmt(df, "roc_auc_mean", "roc_auc_std"),
        "Test PR-AUC": fmt(df, "pr_auc_mean", "pr_auc_std"),
        "Test F1": fmt(df, "f1_mean", "f1_std"),
        "Seeds": df.seeds,
    })


model, ckpt = load_model()
page = st.sidebar.radio("Page", ["Overview", "Shock scenario", "Results"])
st.sidebar.divider()
if ckpt:
    m = ckpt["test_metrics"]
    st.sidebar.caption(f"Served model: GATv2 (seed 0), trained {ckpt['trained_at'][:10]}")
    st.sidebar.caption(f"Test ROC-AUC {m['roc_auc']:.3f} · PR-AUC {m['pr_auc']:.3f} "
                       f"(base rate {m['base_rate']:.3f})")
else:
    st.sidebar.warning("No trained model. Run `python -m experiments.run_static_comparison`.")

# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Financial shock propagation with graph neural networks")
    st.markdown("""
When one stock has an abnormal move (a *shock*), which related stocks also make a large
move shortly after? This project frames that as **node classification on a stock graph**
and tests whether graph neural networks beat simple baselines on held-out, later-in-time events.

**Two experiments** (details in the README):

1. **Finance sector** — 10 large US banks, documented peer links, 1-day horizon.
   This is the model served on the *Shock scenario* page.
2. **CascadeNet ablation** — ~200 stocks across the S&P 500, NIFTY 50, FTSE 100 and Nikkei 225,
   12 weekly graph snapshots, DCC-GARCH correlation edges and a temporal-attention GNN.

All numbers below are on **test events that come after the training and validation periods**,
averaged over several random seeds.
""")
    for title, name in [("Finance sector", "static_comparison.csv"),
                        ("CascadeNet ablation (S&P 500 targets)", "ablation_results.csv")]:
        df = read_csv(name)
        st.subheader(title)
        if df is None:
            st.info(f"`{RESULTS}/{name}` not found — run the experiment first.")
        else:
            st.dataframe(results_table(df), hide_index=True, use_container_width=True)
            st.caption(f"Positive base rate on test: {df.base_rate.iloc[0]:.3f} "
                       f"(PR-AUC of a random scorer). ROC-AUC of a random scorer: 0.5.")

# ---------------------------------------------------------------------------
elif page == "Shock scenario":
    st.title("Shock scenario — US banks")
    if model is None:
        st.stop()
    st.markdown(f"""
The model scores each bank's probability of moving more than
**{ckpt['config']['impact_threshold']:.0%} over the next {ckpt['config']['horizon']} trading day**,
given the latest market data and a shock at the chosen bank.
Optionally override the shock bank's latest daily return to explore a hypothetical shock.
""")
    c1, c2 = st.columns([1, 2])
    with c1:
        source = st.selectbox("Shock source", FINANCE_TICKERS)
        hypothetical = st.checkbox("Override latest return of the shock source")
        shock_ret = st.slider("Hypothetical return (%)", -20.0, 20.0, -8.0, 0.5,
                              disabled=not hypothetical)
        run = st.button("Score", type="primary")

    if run:
        prices = recent_prices(tuple(FINANCE_TICKERS))
        missing = [t for t in FINANCE_TICKERS if t not in prices]
        if missing:
            st.error(f"Could not download: {missing}")
            st.stop()
        if hypothetical:
            df = prices[source].copy()
            df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-2] * (1 + shock_ret / 100)
            prices = {**prices, source: df}
        feats = {t: node_features(prices[t], t, source) for t in FINANCE_TICKERS}
        graph = build_graph(FINANCE_TICKERS, feats, source)
        with torch.no_grad():
            probs = model(graph.x, graph.edge_index, graph.edge_attr).numpy()

        table = pd.DataFrame({
            "Ticker": FINANCE_TICKERS,
            "Latest return (%)": [feats[t][2] for t in FINANCE_TICKERS],
            "P(large move)": probs,
        })
        table = table[table.Ticker != source].sort_values("P(large move)", ascending=False)
        table["Flagged"] = table["P(large move)"] >= ckpt["threshold"]
        as_of = prices[source].index[-1].date()
        with c2:
            st.caption(f"Data as of {as_of}. Flag threshold {ckpt['threshold']:.3f} "
                       f"(chosen on validation to maximise F1).")
            st.dataframe(table.style.format({"Latest return (%)": "{:+.2f}", "P(large move)": "{:.3f}"}),
                         hide_index=True, use_container_width=True)
            st.download_button("Download CSV", table.to_csv(index=False).encode(),
                               file_name=f"shock_{source}_{as_of}.csv")

        G = nx.DiGraph()
        G.add_nodes_from(FINANCE_TICKERS)
        G.add_edges_from((s, d) for s, ds_ in SUPPLY_CHAIN.items() for d in ds_)
        pos = nx.circular_layout(G)
        ex, ey = [], []
        for s, d in G.edges:
            ex += [pos[s][0], pos[d][0], None]
            ey += [pos[s][1], pos[d][1], None]
        p = dict(zip(FINANCE_TICKERS, probs))
        fig = go.Figure([
            go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#bbbbbb", width=1), hoverinfo="skip"),
            go.Scatter(
                x=[pos[t][0] for t in FINANCE_TICKERS], y=[pos[t][1] for t in FINANCE_TICKERS],
                mode="markers+text", text=FINANCE_TICKERS, textposition="top center",
                marker=dict(size=28, color=[p[t] for t in FINANCE_TICKERS],
                            colorscale="Reds", cmin=0, cmax=1, showscale=True,
                            line=dict(color=["black" if t == source else "white" for t in FINANCE_TICKERS],
                                      width=3),
                            colorbar=dict(title="P(large move)")),
                hovertext=[f"{t}: {p[t]:.3f}" for t in FINANCE_TICKERS], hoverinfo="text"),
        ])
        fig.update_layout(showlegend=False, height=480, xaxis_visible=False, yaxis_visible=False,
                          title="Documented peer links (the model also links the shock source to every bank)")
        st.plotly_chart(fig, use_container_width=True)
        m = ckpt["test_metrics"]
        st.info(f"On held-out test events this model reached ROC-AUC {m['roc_auc']:.3f} and "
                f"PR-AUC {m['pr_auc']:.3f} against a base rate of {m['base_rate']:.3f}. "
                "Treat the scores as a research demo, not a trading signal.")

# ---------------------------------------------------------------------------
else:
    st.title("Results")
    for title, name, figs in [
        ("Finance sector", "static_comparison.csv", ["static_comparison"]),
        ("CascadeNet ablation", "ablation_results.csv",
         ["ablation_metrics", "ablation_roc", "training_curves", "temporal_attention",
          "cross_market_attention"]),
    ]:
        st.header(title)
        df = read_csv(name)
        if df is not None:
            st.dataframe(df, hide_index=True, use_container_width=True)
        for f in figs:
            path = os.path.join(FIGURES, f"{f}.png")
            if os.path.exists(path):
                st.image(path)
