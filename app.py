"""
Sri Lankan used-car price estimator - ensemble learning dashboard (Streamlit)

Run:  python -m streamlit run app.py

Needs only: models/ and artifacts/ (both written by the notebook) and static/ (bundled font).
AI usage: generated with the help of Claude (Anthropic) and reviewed by the author - see the notebook, section 10.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
import xgboost
from sklearn.base import BaseEstimator, RegressorMixin

ROOT = Path(__file__).parent
MODELS, ART = ROOT / "models", ROOT / "artifacts"

# ------------------------------------------------------------------ colours
INK, MUTED, LINE, GRID, ACCENT = "#1c2230", "#5d6676", "#dde1e8", "#e7eaef", "#0e5e66"
FAMILY_COLORS = {"Bagging": "#2a78d6", "Boosting": "#eb6834", "Stacking": "#1baf7a", "Hybrid": "#eda100"}
SEQ = ["#e8f1f2", "#7fb0b5", "#0e5e66", "#073236"]        # one hue, light -> dark (errors / importance)
FAMILY_ORDER = list(FAMILY_COLORS)
FUEL_COLORS = {"Petrol": "#5d6676", "Hybrid": "#0e5e66", "Diesel": "#9a5aa6"}
MAIN_MODELS = {
    "Random Forest": ("random_forest.pkl", "Bagging"),
    "XGBoost": ("xgboost.pkl", "Boosting"),
    "Stacking": ("stacking.pkl", "Stacking"),
    "Hybrid RF+ANN (blend)": ("hybrid_rf_ann_blend.pkl", "Hybrid"),
    "Hybrid RF→ANN (residual)": ("hybrid_rf_ann_residual.pkl", "Hybrid"),
}


# ------------------------------------------------------------------ model classes
# The saved models contain three small classes that were written in the notebook.
# Python needs classes with the same names to load them, so their prediction part is repeated here.
class RFANNBlendHybrid(BaseEstimator, RegressorMixin):
    def predict(self, X):
        return self.weight_rf_ * self.rf_.predict(X) + (1 - self.weight_rf_) * self.ann_.predict(X)


class RFANNResidualHybrid(BaseEstimator, RegressorMixin):
    def predict(self, X):
        return self.rf_.predict(X) + self.ann_.predict(X)


class PortableXGBRegressor(xgboost.XGBRegressor):
    def __setstate__(self, state):
        raw = state.pop("_booster_json", None)
        self.__dict__.update(state)
        if raw is not None:
            booster = xgboost.Booster()
            booster.load_model(bytearray(raw))
            self._Booster = booster


for _cls in (RFANNBlendHybrid, RFANNResidualHybrid, PortableXGBRegressor):
    setattr(sys.modules["__main__"], _cls.__name__, _cls)  # the notebook saved them as __main__.<name>


st.set_page_config(page_title="Sri Lankan used-car price estimator", page_icon="🚗", layout="wide")

st.markdown("""
<style>
@font-face { font-family: "Archivo Board"; src: url("app/static/archivo-latin-wdth-normal.woff2") format("woff2");
             font-weight: 100 900; font-stretch: 62% 125%; }
.block-container { padding-top: 2.2rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton { display: none; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }

.mast h1 { font-size: 2.15rem; font-weight: 760; margin: 0 0 .35rem 0; padding: 0; color: #1c2230; }
.mast p  { font-size: 1.02rem; color: #5d6676; max-width: 72ch; margin: 0 0 .6rem 0; line-height: 1.5; }

button[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }

/* the one memorable element: a car-window "for sale" board */
.board { background: #fff; border: 4px solid #1c2230; border-radius: 12px; padding: 14px 22px 16px;
         box-shadow: inset 0 0 0 4px #fff, inset 0 0 0 6px #1c2230; max-width: 520px; }
.board-top { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
             border-bottom: 2px solid #1c2230; padding-bottom: 6px; margin-bottom: 4px; }
.board-sale { font-family: "Archivo Board", "Archivo", sans-serif; font-variation-settings: "wdth" 112, "wght" 850;
              font-size: 1.05rem; letter-spacing: .14em; color: #1c2230; }
.board-car { font-size: .95rem; font-weight: 650; color: #1c2230; text-align: right; }
.board-price { font-family: "Archivo Board", "Archivo", sans-serif; font-variation-settings: "wdth" 62, "wght" 820;
               font-size: clamp(64px, 7.4vw, 104px); line-height: .98; color: #1c2230; letter-spacing: -.005em;
               animation: settle .28s ease-out; }
.board-price small { font-variation-settings: "wdth" 80, "wght" 700; font-size: .27em; letter-spacing: 0; }
.board-detail { font-size: .95rem; color: #5d6676; }
.board-note { font-size: .92rem; color: #5d6676; margin: .55rem 0 1.1rem 2px; max-width: 520px; }
@keyframes settle { from { opacity: .3; transform: translateY(5px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .board-price { animation: none; } }

.facts { max-width: 520px; margin: 0; }
.facts div { display: grid; grid-template-columns: 10rem 1fr; gap: 12px; padding: 9px 0; border-top: 1px solid #dde1e8; }
.facts dt { color: #5d6676; font-size: .93rem; }
.facts dd { margin: 0; color: #1c2230; font-size: 1rem; font-weight: 600; }
.facts dd span { font-weight: 400; color: #5d6676; }

.verdict { font-size: 1.12rem; line-height: 1.5; color: #1c2230; max-width: 78ch; margin: .2rem 0 1rem 0; }

.steps { display: flex; flex-wrap: wrap; gap: 0; margin: .4rem 0 1.4rem 0; }
.step { flex: 1 1 150px; padding: 10px 16px 12px 14px; border-left: 3px solid #dde1e8; }
.step:first-child { border-left-color: #1c2230; } .step:last-child { border-left-color: #0e5e66; }
.step b { display: block; font-size: 1.55rem; font-weight: 760; color: #1c2230; line-height: 1.15; }
.step span { font-size: .9rem; color: #5d6676; }
.step i { font-style: normal; font-size: .8rem; color: #8a93a3; }

.foot { color: #8a93a3; font-size: .85rem; margin-top: 2rem; border-top: 1px solid #dde1e8; padding-top: .7rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the trained models…")
def load_models():
    return {name: joblib.load(MODELS / f) for name, (f, _) in MAIN_MODELS.items()}


@st.cache_data
def load_artifacts():
    a = {
        "test": pd.read_csv(ART / "metrics_test.csv", index_col=0),
        "cv": pd.read_csv(ART / "metrics_cv.csv", index_col=0),
        "seg": pd.read_csv(ART / "segment_metrics.csv"),
        "imp": pd.read_csv(ART / "permutation_importance.csv"),
        "curve": pd.read_csv(ART / "hybrid_weight_curve.csv"),
        "preds": pd.read_csv(ART / "test_predictions.csv"),
        "opts": json.load(open(ART / "options.json")),
        "info": json.load(open(ART / "training_info.json")),
    }
    a["data"] = add_features(pd.read_csv(ART / "clean_listings.csv"), a["opts"]["reference_year"])
    return a


def add_features(df, reference_year):
    # identical to add_features() in the notebook
    out = df.copy()
    out["Age"] = (reference_year - out["YOM"]).clip(lower=0)
    out["km_per_year"] = out["Mileage (km)"] / np.maximum(out["Age"], 1)
    out["Brand_Model"] = out["Brand"] + " " + out["Model"]
    out["Brand_Model_Edition"] = out["Brand_Model"] + " | " + out["Edition"]
    return out


FEATURES = ["Age", "Mileage (km)", "Engine (cc)", "km_per_year", "Transmission", "Fuel Type", "Body Type",
            "Negotiable", "Brand", "Brand_Model", "Brand_Model_Edition"]

A = load_artifacts()
test_df, cv_df, info, opts = A["test"], A["cv"], A["info"], A["opts"]
REF_YEAR = opts["reference_year"]

_bad = {k: v for k, v in {"scikit-learn": (info["sklearn"], sklearn.__version__),
                          "xgboost": (info["xgboost"], xgboost.__version__)}.items() if v[0] != v[1]}
if _bad:
    st.error("The saved models need different library versions:\n\n" +
             "\n".join(f"* {k}: models need **{need}**, installed **{have}**" for k, (need, have) in _bad.items()) +
             "\n\nRun `python -m pip install -r requirements.txt`, then restart with `python -m streamlit run app.py`.")
    st.stop()

best_model = cv_df["CV MAE"].idxmin()          # lowest 5-fold cross-validation error = most reliable
most_stable = cv_df["CV MAE std"].idxmin()


def fmt(v, d=1):
    return f"{v:,.{d}f}"


def style(fig, h=380, legend=True):
    fig.update_layout(
        height=h, margin=dict(l=6, r=26, t=48, b=6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Archivo, sans-serif", size=13, color=INK),
        title=dict(font=dict(size=16, color=INK), x=0, xanchor="left", y=0.97),
        showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.14, x=0, title_text="", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family="Archivo, sans-serif", color=INK)),
        barcornerradius=4, bargap=0.42,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, tickfont=dict(color=INK), title_font=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ------------------------------------------------------------------ header
st.markdown(f"""
<div class="mast">
  <h1>Sri Lankan used-car price estimator</h1>
  <p>Five ensemble models (bagging, boosting, stacking and two hybrids) trained on {info['n_rows_clean']:,} ikman.lk adverts.
  Prices are in lakhs: 1 lakh = Rs. 100,000.</p>
</div>""", unsafe_allow_html=True)

tab_price, tab_cmp, tab_imp, tab_data = st.tabs(["Price a car", "Compare the models", "What drives the price", "The data"])

# ================================================================== PRICE A CAR
with tab_price:
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("#### Car details")
        brands = sorted(opts["brands"].keys())
        c1, c2 = st.columns(2)
        brand = c1.selectbox("Brand", brands, index=brands.index("TOYOTA"), key="brand")
        models_for_brand = opts["brands"][brand]
        model_names = sorted(models_for_brand, key=lambda m: -models_for_brand[m]["n"])
        model = c2.selectbox("Model", model_names, help="Most-listed models are at the top.", key="model")
        d = models_for_brand[model]
        c1, c2 = st.columns(2)
        yom = c1.number_input("Year of manufacture", opts["yom_min"], opts["yom_max"], value=d["yom"], step=1)
        km = c2.number_input("Mileage (km)", 0, 1_000_000, value=int(round(d["km"] / 1000) * 1000), step=1000,
                             help="Odometer reading. Use 0 for a brand-new, unregistered car.")
        editions = d["editions"] + (["UNKNOWN"] if "UNKNOWN" not in d["editions"] else [])
        edition = st.selectbox("Edition / grade", editions, help="Grade as written in adverts. Choose UNKNOWN if unsure.")
        c1, c2 = st.columns(2)
        engine = c1.number_input("Engine capacity (cc)", 0, 8000, value=int(d["engine"]), step=50,
                                 help="Use 0 for electric cars.")
        fuel = c2.selectbox("Fuel", opts["fuel"], index=opts["fuel"].index(d["fuel"]))
        c1, c2, c3 = st.columns(3)
        trans = c1.selectbox("Transmission", opts["transmission"], index=opts["transmission"].index(d["transmission"]))
        body = c2.selectbox("Body type", opts["body"], index=opts["body"].index(d["body"]))
        nego = c3.selectbox("Negotiable", ["Yes", "No"])

    row = pd.DataFrame([{
        "Brand": brand, "Model": model, "Edition": edition, "YOM": int(yom), "Mileage (km)": float(km),
        "Engine (cc)": 0.0 if fuel == "Electric" else float(engine), "Transmission": trans,
        "Fuel Type": fuel, "Body Type": body, "Negotiable": nego}])
    X_row = add_features(row, REF_YEAR)[FEATURES]

    models = load_models()
    preds = {name: float(m.predict(X_row)[0]) for name, m in models.items()}
    est, lo, hi = preds[best_model], min(preds.values()), max(preds.values())

    data = A["data"]
    real = data[(data.Brand == brand) & (data.Model == model) & (data.YOM == int(yom))]["Price"]

    with right:
        edition_txt = "" if edition == "UNKNOWN" else f"{edition.title()}, "
        st.markdown(f"""
<div class="board" role="img" aria-label="Estimated asking price {fmt(est)} lakhs">
  <div class="board-top"><span class="board-sale">FOR SALE</span><span class="board-car">{int(yom)} {brand} {model}</span></div>
  <div class="board-price"><small>Rs.</small> {fmt(est)} <small>lakhs</small></div>
  <div class="board-detail">{edition_txt}{int(km):,} km, {fuel.lower()}, {trans.lower()}</div>
</div>
<p class="board-note">Estimated by {best_model}, the model with the lowest error in 5-fold cross-validation.</p>
""", unsafe_allow_html=True)
        if len(real):
            real_txt = (f"{fmt(real.min())} to {fmt(real.max())} lakhs <span>({len(real)} adverts, median {fmt(real.median())})</span>"
                        if len(real) > 1 else f"{fmt(real.iloc[0])} lakhs <span>(1 advert)</span>")
        else:
            real_txt = f"<span>No {int(yom)} {brand} {model} adverts in the data</span>"
        st.markdown(f"""
<dl class="facts">
  <div><dt>All five models</dt><dd>{fmt(lo)} to {fmt(hi)} lakhs <span>(they differ by {fmt(hi - lo)})</span></dd></div>
  <div><dt>Real adverts, same year</dt><dd>{real_txt}</dd></div>
  <div><dt>This model, any year</dt><dd>{d['n']:,} <span>(median {fmt(d['price_median'])} lakhs)</span></dd></div>
</dl>""", unsafe_allow_html=True)
        if int(yom) <= REF_YEAR - 4 and km < 1000:
            st.warning(f"0 km is unusual for a {int(yom)} car. In the data, 0 km appears only on cars from "
                       f"{REF_YEAR - 3} onwards (new or unregistered imports). For older cars it was treated as a missing "
                       "reading during training, so enter the real mileage for a reliable estimate.")
        if (hi - lo) / est > 0.15:
            st.warning("The models disagree by more than 15 %, so treat this estimate as rough.")
        if d["n"] < 5:
            st.info(f"Only {d['n']} advert(s) for {brand} {model} in the data. The models lean on brand, age and "
                    "engine size instead, so expect a larger error.")

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        pdf = pd.DataFrame({"Model": list(preds), "Estimate": list(preds.values()),
                            "Family": [MAIN_MODELS[n][1] for n in preds]}).sort_values("Estimate")
        fig = go.Figure()
        if len(real) > 1:
            fig.add_vrect(x0=real.min(), x1=real.max(), fillcolor=INK, opacity=0.07, line_width=0)
            fig.add_annotation(x=(real.min() + real.max()) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                               text=f"real {int(yom)} adverts", font=dict(color=MUTED, size=12))
        for fam in FAMILY_ORDER:
            g = pdf[pdf.Family == fam]
            if len(g):
                fig.add_scatter(x=g["Estimate"], y=g["Model"], mode="markers+text", name=fam,
                                marker=dict(size=15, color=FAMILY_COLORS[fam], line=dict(color="white", width=2)),
                                text=g["Estimate"].map(fmt), textposition="middle right", textfont=dict(color=INK),
                                hovertemplate="%{y}: %{x:.1f} lakhs<extra></extra>")
        xs_all = list(pdf["Estimate"]) + (list(real) if len(real) else [])
        pad = (max(xs_all) - min(xs_all)) * 0.25 + est * 0.03
        fig.update_xaxes(range=[min(xs_all) - pad, max(xs_all) + pad * 1.6], title=None)
        fig.update_yaxes(categoryorder="array", categoryarray=list(pdf["Model"]), title=None, showgrid=False)
        fig.update_layout(title="Each model's estimate (lakhs)")
        show(style(fig, 330))
    with c2:
        what = st.radio("What if this car had a different", ["year", "mileage"], horizontal=True, key="whatif")
        if what == "year":
            xs = np.arange(max(opts["yom_min"], int(yom) - 10), min(opts["yom_max"], int(yom) + 4) + 1)
            grid = pd.concat([row.assign(YOM=int(v)) for v in xs], ignore_index=True)
            xlab, now = "Year of manufacture", int(yom)
        else:
            xs = np.linspace(0, max(200_000, km * 2), 21).round(-3)
            grid = pd.concat([row.assign(**{"Mileage (km)": float(v)}) for v in xs], ignore_index=True)
            xlab, now = "Mileage (km)", km
        Xg = add_features(grid, REF_YEAR)[FEATURES]
        curve = pd.DataFrame({xlab: xs})
        for name in ["XGBoost", "Random Forest", "Hybrid RF+ANN (blend)"]:
            curve[name] = models[name].predict(Xg)
        long = curve.melt(xlab, var_name="Model", value_name="Estimate")
        fig = px.line(long, x=xlab, y="Estimate", color="Model", markers=True,
                      color_discrete_map={"Random Forest": FAMILY_COLORS["Bagging"], "XGBoost": FAMILY_COLORS["Boosting"],
                                          "Hybrid RF+ANN (blend)": FAMILY_COLORS["Hybrid"]},
                      title=f"Estimate for this car by {what} (lakhs)")
        fig.update_traces(line_width=2, marker_size=7)
        fig.add_vline(x=now, line_dash="dot", line_color=MUTED, annotation_text="this car",
                      annotation_font=dict(color=MUTED, size=12))
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(title=None)
        fig.update_xaxes(title=None)
        show(style(fig, 318))

# ================================================================== COMPARE
with tab_cmp:
    stable_txt = (" and the smallest variation between splits" if most_stable == best_model else "")
    st.markdown(f"""<p class="verdict"><b>{best_model}</b> is the most reliable model: it has the lowest average error
    across five different train/test splits ({fmt(cv_df.loc[best_model, 'CV MAE'], 2)} lakhs){stable_txt}.
    On the single 20 % hold-out split, <b>{test_df['MAE'].idxmin()}</b> scores best, but one split can be lucky.</p>""",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        d = cv_df.reset_index().sort_values("CV MAE", ascending=False)
        fig = go.Figure()
        for fam in FAMILY_ORDER:
            g = d[d.Family == fam]
            fig.add_bar(x=g["CV MAE"], y=g["Model"], orientation="h", name=fam, marker_color=FAMILY_COLORS[fam],
                        error_x=dict(type="data", array=g["CV MAE std"], color=MUTED, thickness=1.5, width=5),
                        hovertemplate="%{y}<br>average error %{x:.2f} ± %{error_x.array:.2f} lakhs<extra></extra>")
        for _, r in d.iterrows():
            fig.add_annotation(x=r["CV MAE"] + r["CV MAE std"], y=r["Model"], text=fmt(r["CV MAE"], 2), xshift=6,
                               xanchor="left", showarrow=False, font=dict(color=INK))
        fig.update_layout(title="Average error across 5 splits (MAE, lakhs; bar = ± 1 std)", barmode="overlay",
                          yaxis=dict(categoryorder="array", categoryarray=list(d["Model"])))
        show(style(fig, 380))
    with c2:
        metric = st.radio("Hold-out metric", ["MAE", "RMSE", "R2", "MAPE_%"], horizontal=True,
                          format_func=lambda m: {"MAPE_%": "MAPE %", "R2": "R²"}.get(m, m))
        asc = metric == "R2"
        d = test_df.reset_index().sort_values(metric, ascending=not asc)
        fig = px.bar(d, x=metric, y="Model", color="Family", orientation="h", color_discrete_map=FAMILY_COLORS,
                     category_orders={"Family": FAMILY_ORDER},
                     text=d[metric].map(lambda v: f"{v:.3f}" if metric == "R2" else f"{v:.2f}"),
                     title=f"One 80/20 split: {metric.replace('_%', ' %').replace('R2', 'R²')} "
                           f"({'higher' if asc else 'lower'} is better)")
        fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=INK)
        fig.update_yaxes(categoryorder="array", categoryarray=list(d["Model"])[::-1], title=None)
        fig.update_xaxes(title=None)
        if metric == "R2":
            fig.update_xaxes(range=[max(0, d[metric].min() - 0.03), 1])
        show(style(fig, 330))

    st.markdown(f"##### Hold-out test set ({info['n_test']:,} adverts never seen during training)")
    tbl = test_df[["Family", "MAE", "RMSE", "R2", "MAPE_%", "Fit time (s)"]].sort_values("MAE")
    st.dataframe(tbl.rename(columns={"R2": "R²", "MAPE_%": "MAPE %"}).style
                 .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}", "MAPE %": "{:.1f}", "Fit time (s)": "{:.1f}"})
                 .highlight_min(subset=["MAE", "RMSE", "MAPE %"], color="#d6e8ea")
                 .highlight_max(subset=["R²"], color="#d6e8ea"), width="stretch")
    st.caption("MAE is the main metric: the average error in lakhs a buyer or seller would notice. "
               "RMSE punishes big misses on expensive cars, R² is the share of price variation explained, "
               "MAPE is the error as a percentage of the price. Shaded cells are the best in each column.")

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        seg = A["seg"]
        seg_metric = st.radio("Error by price band", ["MAPE_%", "MAE"], horizontal=True, key="segm",
                              format_func=lambda m: "Percentage error" if m == "MAPE_%" else "Error in lakhs")
        labels = ["< 50", "50–100", "100–200", "200–500", "> 500"]
        piv = seg.pivot(index="Model", columns="Segment (lakhs)", values=seg_metric)[labels]
        piv = piv.loc[test_df.sort_values("MAE").index]
        fig = px.imshow(piv, text_auto=".1f", color_continuous_scale=SEQ, aspect="auto",
                        labels=dict(x="Actual price band (lakhs)", y="", color=""),
                        title="Where the errors are, by price band")
        fig.update_layout(coloraxis_colorbar=dict(thickness=10, len=0.8))
        show(style(fig, 360, legend=False))
        n_seg = seg.drop_duplicates("Segment (lakhs)").set_index("Segment (lakhs)")["n"].reindex(labels)
        st.caption("Test adverts per band: " + ", ".join(f"{k}: {v}" for k, v in n_seg.items()) +
                   ". Cheap cars have small errors in lakhs but large percentage errors; luxury cars the opposite.")
    with c2:
        pr = A["preds"]
        sel = st.selectbox("Predicted vs actual for", list(test_df.sort_values("MAE").index), key="pva")
        fam = test_df.loc[sel, "Family"]
        fig = px.scatter(pr, x="Actual", y=sel, log_x=True, log_y=True, opacity=0.4,
                         hover_data={"Brand_Model_Edition": True, "Age": True, "Mileage (km)": ":,.0f"},
                         color_discrete_sequence=[FAMILY_COLORS[fam]], labels={sel: "Predicted", "Actual": "Actual"},
                         title="Each test advert (log scale; on the line = perfect)")
        fig.update_traces(marker_size=6)
        fig.update_xaxes(tickvals=[3, 10, 30, 100, 300, 1000, 3000], ticktext=["3", "10", "30", "100", "300", "1k", "3k"])
        fig.update_yaxes(tickvals=[3, 10, 30, 100, 300, 1000, 3000], ticktext=["3", "10", "30", "100", "300", "1k", "3k"])
        lim = [pr["Actual"].min() * 0.8, pr["Actual"].max() * 1.2]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color=MUTED, width=1.5),
                                 name="perfect prediction", hoverinfo="skip"))
        show(style(fig, 360, legend=False))

# ================================================================== WHAT DRIVES THE PRICE
with tab_imp:
    imp = A["imp"]
    top3 = imp.groupby("Feature")["Importance (MAE increase)"].mean().nlargest(3).index.tolist()
    st.markdown(f"""<p class="verdict">Across the models, <b>{top3[0].lower()}</b>, <b>{top3[1].lower()}</b> and
    <b>{top3[2].lower()}</b> move the price the most. Each bar shows how much the average error grows (in lakhs)
    when that input is shuffled at random, so the model can no longer use it.</p>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        imp_models = [m for m in cv_df.sort_values("CV MAE").index if m in set(imp.Model)]
        msel = st.selectbox("Show the model", imp_models, key="imp")
        fam = MAIN_MODELS[msel][1]
        d = imp[imp.Model == msel].sort_values("Importance (MAE increase)")
        fig = px.bar(d, x="Importance (MAE increase)", y="Feature", orientation="h",
                     color_discrete_sequence=[FAMILY_COLORS[fam]],
                     text=d["Importance (MAE increase)"].map(lambda v: fmt(v)),
                     title=f"{msel}: error increase when shuffled (lakhs)")
        fig.update_traces(textposition="outside",
                          cliponaxis=False, textfont_color=INK)
        fig.update_yaxes(title=None)
        fig.update_xaxes(title=None)
        show(style(fig, 560, legend=False))
    with c2:
        st.markdown("##### Inside the combined models")
        w = pd.DataFrame({"Base model": list(info["stack_meta_weights"]),
                          "Weight": list(info["stack_meta_weights"].values())})
        fig = px.bar(w, x="Weight", y="Base model", orientation="h", color="Base model",
                     text=w["Weight"].map(lambda v: f"{v:.2f}"),
                     color_discrete_map={"Random Forest": FAMILY_COLORS["Bagging"], "XGBoost": FAMILY_COLORS["Boosting"],
                                         "Ridge": "#8a93a3"},
                     title="Stacking: how much the meta-learner trusts each model")
        fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=INK)
        fig.update_yaxes(categoryorder="total ascending", title=None)
        fig.update_xaxes(title=None)
        show(style(fig, 230, legend=False))
        st.caption("Random Forest gets little weight because its mistakes are almost the same as XGBoost's. "
                   "Ridge is weaker on its own but makes different mistakes, so it adds more.")
        cv = A["curve"]
        fig = px.line(cv, x="w_rf", y="oof_rmse_log", markers=True, color_discrete_sequence=[FAMILY_COLORS["Hybrid"]],
                      labels={"w_rf": "share of Random Forest in the blend (0 = only the neural network)",
                              "oof_rmse_log": "error (log RMSE)"},
                      title=f"Hybrid blend: best mix is {info['blend_weight_rf']:.0%} Random Forest")
        fig.update_traces(line_width=2, marker_size=7)
        fig.add_vline(x=info["blend_weight_rf"], line_dash="dot", line_color=MUTED)
        show(style(fig, 250, legend=False))
        st.caption("Mixing the two gives a lower error than either one alone (the ends of the curve), "
                   "because the forest and the neural network make different mistakes.")

# ================================================================== THE DATA
with tab_data:
    data = A["data"]
    cl = info.get("cleaning_log", {})
    steps = [
        (cl.get("raw_rows", 0), "adverts scraped from ikman.lk", ""),
        (cl.get("duplicates_removed", 0), "duplicate adverts removed", "the same advert scraped twice"),
        (cl.get("price_typos_removed", 0), "price typo removed", "a car listed at Rs. 21.5 billion"),
        (cl.get("mileage_placeholders_imputed", 0), "mileage placeholders fixed", "1,000,000 km or 0 km on old cars"),
        (cl.get("engine_typos_imputed", 0), "engine-size typos fixed", "e.g. 1 cc or 10,000 cc"),
        (cl.get("clean_rows", len(data)), "adverts used for training", ""),
    ]
    st.markdown('<p class="verdict">How the raw scrape became training data:</p><div class="steps">' + "".join(
        f'<div class="step"><b>{n:,}</b><span>{label}</span><br><i>{note}</i></div>' for n, label, note in steps) +
        "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.histogram(data, x="Price", nbins=90, color_discrete_sequence=[ACCENT],
                           title="Asking prices: most cars under 200 lakhs, a long tail of luxury cars")
        fig.update_xaxes(title="Price (lakhs)")
        fig.update_yaxes(title="Adverts")
        show(style(fig, 340, legend=False))
    with c2:
        top = data.Brand.value_counts().head(12).index
        med = data[data.Brand.isin(top)].groupby("Brand")["Price"].median().sort_values().reset_index()
        fig = px.bar(med, x="Price", y="Brand", orientation="h", color_discrete_sequence=[ACCENT],
                     text=med["Price"].map(lambda v: fmt(v, 0)), title="Median price of the 12 most-listed brands (lakhs)")
        fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=INK)
        fig.update_yaxes(title=None)
        fig.update_xaxes(title=None)
        show(style(fig, 340, legend=False))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        main = data[data["Fuel Type"].isin(list(FUEL_COLORS))]
        fig = px.scatter(main, x="YOM", y="Price", color="Fuel Type", log_y=True, opacity=0.45,
                         color_discrete_map=FUEL_COLORS, hover_data={"Brand_Model": True},
                         title="Newer cars cost more (price on a log scale)")
        fig.update_traces(marker_size=5)
        fig.update_xaxes(title="Year of manufacture")
        fig.update_yaxes(title="Price (lakhs)", tickvals=[3, 10, 30, 100, 300, 1000, 3000],
                         ticktext=["3", "10", "30", "100", "300", "1,000", "3,000"])
        fig = style(fig, 420)
        fig.update_layout(legend=dict(y=-0.2, itemsizing="constant"))
        show(fig)
    with c2:
        used = data[data.Age >= 1]
        fig = px.scatter(used, x="Age", y="Mileage (km)", opacity=0.25, hover_data={"Brand_Model": True},
                         color_discrete_sequence=[ACCENT],
                         title="Real odometer readings: same age, very different mileage")
        fig.update_traces(marker_size=5)
        fig.update_xaxes(title="Age (years)")
        show(style(fig, 400, legend=False))
    with st.expander("Browse the cleaned adverts"):
        st.dataframe(data.drop(columns=["Brand_Model", "Brand_Model_Edition", "km_per_year"]), width="stretch", height=360)

st.markdown(f"""<p class="foot">Data: {cl.get('raw_rows', 0):,} adverts scraped from ikman.lk in 2026.
These are asking prices, not final sale prices.</p>""", unsafe_allow_html=True)
