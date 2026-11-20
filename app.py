import os
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Market Dashboard", page_icon="📈", layout="wide")
CSV = "data/daily_update.csv"

GROUPS = {
 "Equities": ["S&P 500","Nasdaq-100","Dow Jones","Russell 2000","STOXX Europe 600","DAX","FTSE 100",
              "CAC 40","Nikkei 225","Hang Seng","CSI 300","Shanghai Composite","KOSPI","Nifty 50",
              "MSCI World (proxy URTH)","MSCI EM (proxy EEM)"],
 "Bonds (ETFs)": ["US Aggregate (AGG)","Global Agg ex-US (BNDX)","US Treasury 7-10Y (IEF)",
              "US Treasury 20Y+ (TLT)","US Investment Grade (LQD)","US High Yield (HYG)",
              "EM USD Bonds (EMB)","EM Local Bonds (EMLC)"],
 "Metals": ["Gold","Silver","Platinum","Palladium","Copper","Aluminum","Iron Ore"],
 "Energy": ["WTI Crude","Brent Crude","Henry Hub Nat Gas","TTF Europe Nat Gas","Gasoline (RBOB)",
            "Heating Oil / Diesel","Uranium"],
 "Agriculture": ["Corn","Wheat","Soybeans","Rice","Sugar","Coffee","Cocoa","Cotton","Orange Juice",
                 "Live Cattle","Lean Hogs","Soybean Oil","Soybean Meal"],
 "FX": ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD","USDCNY","USDHKD","USDSGD",
        "EURGBP","EURJPY","GBPJPY","AUDJPY","EURCHF"],
 "Rates (%)": ["Fed Funds","ECB Deposit Rate","BoJ Policy Rate","SOFR","US 2Y Treasury",
               "US 10Y Treasury","US 30Y Treasury","German Bund 10Y","JGB 10Y","UK Gilt 10Y",
               "US 10Y-2Y Spread"],
}
RATES = set(GROUPS["Rates (%)"])
YRS = {"1M":1/12,"3M":0.25,"6M":0.5,"1Y":1,"2Y":2,"3Y":3,"5Y":5,"10Y":10,"20Y":20}


@st.cache_data(ttl=3600)
def load(mtime):
    df = pd.read_csv(CSV, parse_dates=["Date"], index_col="Date").sort_index()
    return df.dropna(axis=1, how="all")


def derived(df):
    d = pd.DataFrame(index=df.index)
    def r(a, b, name):
        if a in df and b in df: d[name] = df[a] / df[b]
    def s(a, b, name):
        if a in df and b in df: d[name] = df[a] - df[b]
    r("Gold","Silver","Gold/Silver"); r("Copper","Gold","Copper/Gold")
    r("Gold","WTI Crude","Gold/Oil"); r("S&P 500","Gold","S&P/Gold")
    r("Nasdaq-100","S&P 500","Nasdaq/S&P"); r("Russell 2000","S&P 500","Small/Large cap")
    r("MSCI EM (proxy EEM)","MSCI World (proxy URTH)","EM/World")
    r("US High Yield (HYG)","US Treasury 7-10Y (IEF)","HY/Treasury (risk appetite)")
    r("Soybeans","Corn","Soybean/Corn"); r("Platinum","Gold","Platinum/Gold")
    s("Brent Crude","WTI Crude","Brent-WTI"); s("US 30Y Treasury","US 2Y Treasury","US 30Y-2Y")
    s("US 10Y Treasury","German Bund 10Y","US-Bund 10Y"); s("US 10Y Treasury","JGB 10Y","US-JGB 10Y")
    s("US 2Y Treasury","Fed Funds","US 2Y-Fed Funds")
    if all(c in df for c in ["Soybeans","Soybean Oil","Soybean Meal"]):   # crush: $/bu
        d["Soy crush"] = df["Soybean Meal"]*0.022 + df["Soybean Oil"]*0.11 - df["Soybeans"]/100
    return d


def pca_input(df, mode, log_lv):
    """Returns DataFrame of PCA inputs (not yet standardized)."""
    pcols = [c for c in df if c not in RATES]
    rcols = [c for c in df if c in RATES]
    if mode == "Daily returns":
        X = pd.concat([df[pcols].pct_change(fill_method=None), df[rcols].diff() / 100], axis=1)
    else:
        lv = df[pcols].where(df[pcols] > 0)
        X = pd.concat([np.log(lv) if log_lv else lv, df[rcols]], axis=1)
    return X.iloc[1:]


def run_pca(X, min_cov=0.9):
    """Standardize within the window and do PCA. Drops columns with poor coverage."""
    X = X.dropna(axis=1, thresh=int(len(X) * min_cov)).ffill().bfill()
    sd = X.std().replace(0, np.nan)
    Z = ((X - X.mean()) / sd).dropna(axis=1)
    vals, vecs = np.linalg.eigh(np.cov(Z.T))
    o = vals.argsort()[::-1]
    vals, vecs = vals[o], vecs[:, o]
    if "S&P 500" in Z:                                  # sign convention: S&P loads positive
        sgn = np.sign(vecs[list(Z.columns).index("S&P 500")]); sgn[sgn == 0] = 1
        vecs = vecs * sgn
    return Z, vals / vals.sum(), pd.DataFrame(vecs, index=Z.columns,
                                              columns=[f"PC{i+1}" for i in range(len(vals))])


@st.cache_data(ttl=3600, show_spinner="Computing rolling PCA…")
def rolling_pc1(mtime, mode, log_lv, win, step):
    X = pca_input(load(mtime), mode, log_lv)
    X = X.dropna(axis=1, thresh=int(len(X) * 0.5))
    out = {}
    for t in range(win, len(X), step):
        w = X.iloc[t - win:t]
        w = w.dropna(axis=1, thresh=int(win * 0.9)).ffill().bfill()
        w = ((w - w.mean()) / w.std().replace(0, np.nan)).dropna(axis=1)
        if w.shape[1] < 5: continue
        e = np.linalg.eigvalsh(np.cov(w.T))
        out[X.index[t]] = e.max() / e.sum()
    return pd.Series(out)


if not os.path.exists(CSV):
    st.error("No data yet - run the GitHub Action once."); st.stop()
mtime = os.path.getmtime(CSV)
df = load(mtime)
last = df.index.max()

# ---------- sidebar ----------
st.sidebar.title("📈 Market Dashboard")
st.sidebar.caption(f"Data {df.index.min():%b %Y} – {last:%d %b %Y} · {len(df):,} rows")
period = st.sidebar.selectbox("Chart period", ["1M","3M","6M","YTD","1Y","3Y","5Y","10Y","20Y","All"], index=4)
if period == "YTD":   start = pd.Timestamp(last.year, 1, 1)
elif period == "All": start = df.index.min()
else:                 start = last - pd.Timedelta(days=round(365.25 * YRS[period]))
view = df[df.index >= start]

tabs = st.tabs(["Overview","Charts","Ratios & Spreads","Rates","Correlation","PCA / Regime"])

# ---------- Overview ----------
with tabs[0]:
    def back(days): return df[df.index <= last - pd.Timedelta(days=days)].iloc[-1]
    cur = df.iloc[-1]
    ytd0 = df[df.index < pd.Timestamp(last.year, 1, 1)]
    ytd0 = ytd0.iloc[-1] if len(ytd0) else df.iloc[0]
    rows = []
    for g, cols in GROUPS.items():
        for c in [c for c in cols if c in df]:
            f = (lambda a: cur[c]-a) if c in RATES else (lambda a: (cur[c]/a-1)*100)
            rows.append([g, c, cur[c], f(df[c].iloc[-2]), f(back(7)[c]), f(back(30)[c]),
                         f(ytd0[c]), f(back(365)[c])])
    tbl = pd.DataFrame(rows, columns=["Group","Asset","Last","1D","1W","1M","YTD","1Y"])
    st.caption("% change for prices; absolute change (percentage points) for rates")
    grp = st.selectbox("Group", ["All"] + list(GROUPS))
    t = tbl if grp == "All" else tbl[tbl.Group == grp]
    chg = ["1D","1W","1M","YTD","1Y"]
    st.dataframe(
        t.style.format({"Last":"{:,.4g}", **{k:"{:+.2f}" for k in chg}})
         .background_gradient(subset=chg, cmap="RdYlGn", vmin=-5, vmax=5),
        hide_index=True, width="stretch", height=600)
    bar = tbl[~tbl.Asset.isin(RATES)].sort_values("1M")
    st.plotly_chart(px.bar(bar, x="1M", y="Asset", color="1M", color_continuous_scale="RdYlGn",
                    range_color=[-10,10], height=1400, title="1-month % change"), width="stretch")

# ---------- Charts ----------
with tabs[1]:
    g = st.selectbox("Asset class", list(GROUPS), key="cg")
    opts = [c for c in GROUPS[g] if c in df]
    sel = st.multiselect("Assets", opts, default=opts[:4])
    mode = st.radio("Show", ["Rebased to 100","Raw level"], horizontal=True)
    logy = st.checkbox("Log scale", value=period in ("10Y","20Y","All"))
    if sel:
        v = view[sel].dropna(how="all")
        if mode.startswith("Rebased") and g != "Rates (%)":
            v = v / v.bfill().iloc[0] * 100
        fig = px.line(v, height=500)
        if logy and g != "Rates (%)": fig.update_yaxes(type="log")
        st.plotly_chart(fig, width="stretch")

# ---------- Ratios ----------
with tabs[2]:
    d = derived(df)
    pick = st.selectbox("Ratio / spread", list(d.columns))
    s = d[pick][d.index >= start].dropna()
    z = (d[pick] - d[pick].rolling(252).mean()) / d[pick].rolling(252).std()
    if len(s):
        c1, c2, c3 = st.columns(3)
        c1.metric(pick, f"{s.iloc[-1]:,.4g}", f"{s.iloc[-1]-s.iloc[0]:+.4g} over period")
        c2.metric("1Y z-score", f"{z.iloc[-1]:+.2f}")
        full = d[pick].dropna()
        c3.metric("Percentile (full history)", f"{(full < full.iloc[-1]).mean()*100:.0f}%")
        fig = px.line(s, height=420); fig.add_hline(s.mean(), line_dash="dot")
        st.plotly_chart(fig, width="stretch")
    zt = ((d - d.rolling(252).mean()) / d.rolling(252).std()).iloc[-1].sort_values()
    st.plotly_chart(px.bar(zt, orientation="h", title="All ratios: 1Y z-score (extremes = stretched)",
                    height=550), width="stretch")

# ---------- Rates ----------
with tabs[3]:
    curve = [c for c in ["Fed Funds","US 2Y Treasury","US 10Y Treasury","US 30Y Treasury"] if c in df]
    fig = go.Figure()
    for lab, dd in [("Today", last), ("1M ago", last-pd.Timedelta(days=30)),
                    ("1Y ago", last-pd.Timedelta(days=365)), ("5Y ago", last-pd.Timedelta(days=1826))]:
        prev = df[df.index <= dd]
        if len(prev):
            fig.add_scatter(x=["FF","2Y","10Y","30Y"][:len(curve)], y=prev.iloc[-1][curve],
                            mode="lines+markers", name=lab)
    fig.update_layout(title="US yield curve", height=380)
    st.plotly_chart(fig, width="stretch")
    if "US 10Y-2Y Spread" in df:
        f2 = px.area(view["US 10Y-2Y Spread"], title="10Y-2Y spread (below 0 = inverted)", height=350)
        f2.add_hline(0, line_color="red"); st.plotly_chart(f2, width="stretch")
    g10 = [c for c in ["US 10Y Treasury","German Bund 10Y","JGB 10Y","UK Gilt 10Y"] if c in df]
    st.plotly_chart(px.line(view[g10], title="Global 10Y yields", height=380), width="stretch")

# ---------- Correlation ----------
with tabs[4]:
    win = st.slider("Window (trading days)", 20, 252, 60)
    gsel = st.multiselect("Groups", list(GROUPS), default=["Equities","Bonds (ETFs)","Metals","Energy"])
    cols = [c for g in gsel for c in GROUPS[g] if c in df and c not in RATES]
    rets = df[cols].pct_change(fill_method=None).iloc[-win:]
    st.plotly_chart(px.imshow(rets.corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    height=750, title=f"{win}-day return correlation"), width="stretch")
    if "S&P 500" in df and "US Treasury 7-10Y (IEF)" in df:
        r = df[["S&P 500","US Treasury 7-10Y (IEF)"]].pct_change(fill_method=None)
        rc = r["S&P 500"].rolling(win).corr(r["US Treasury 7-10Y (IEF)"])
        f = px.line(rc[rc.index >= start], title="Rolling stock-bond correlation (S&P vs IEF)", height=350)
        f.add_hline(0); st.plotly_chart(f, width="stretch")

# ---------- PCA ----------
with tabs[5]:
    c1, c2 = st.columns(2)
    pmode = c1.radio("PCA input", ["Daily returns", "Levels (standardized)"], horizontal=True)
    log_lv = c2.checkbox("Use log prices for levels", value=True,
                         disabled=(pmode == "Daily returns"),
                         help="Log makes % moves comparable across assets. Rates always stay in %.")
    pwin = st.select_slider("PCA window", ["3M","6M","1Y","2Y","3Y","5Y","10Y","20Y"], value="1Y")
    min_cov = st.slider("Min data coverage to include an asset", 0.5, 1.0, 0.9, 0.05,
                        help="Assets with less history than this share of the window are dropped "
                             "(e.g. BNDX starts 2013, SOFR 2018).")
    if pmode == "Daily returns":
        st.caption("Returns PCA = what drives day-to-day co-movement. Rates use daily changes. "
                   "Each series is z-scored within the window.")
    else:
        st.caption("Levels PCA = long-term trends / regimes. Each series is z-scored within the window, "
                   "so different units are comparable. Levels are trending, so PC1 is usually very large "
                   "— read it as 'common trend', not risk.")

    X = pca_input(df, pmode, log_lv)
    X = X[X.index >= last - pd.Timedelta(days=round(365.25 * YRS[pwin]))]
    Z, ev, L = run_pca(X, min_cov)
    dropped = [c for c in X.columns if c not in Z.columns]
    st.caption(f"{Z.shape[1]} assets × {len(Z):,} days used"
               + (f" · dropped {len(dropped)}: {', '.join(dropped)}" if dropped else ""))

    m = st.columns(4)
    for i in range(3): m[i].metric(f"PC{i+1} variance", f"{ev[i]*100:.1f}%")
    m[3].metric("PC1–3 total", f"{ev[:3].sum()*100:.1f}%")
    st.plotly_chart(px.bar(x=[f"PC{i+1}" for i in range(10)], y=ev[:10]*100,
                    labels={"x":"", "y":"% variance"}, title="Scree plot", height=300), width="stretch")

    k = st.selectbox("Loadings of", [f"PC{i+1}" for i in range(5)])
    ld = L[k].sort_values()
    top = pd.concat([ld.head(12), ld.tail(12)]) if len(ld) > 24 else ld
    st.plotly_chart(px.bar(top, orientation="h", height=650, title=f"{k} largest loadings"),
                    width="stretch")

    scores = pd.DataFrame(Z.values @ L.iloc[:, :3].values, index=Z.index, columns=["PC1","PC2","PC3"])
    if pmode == "Daily returns":
        scores = scores.cumsum()
        ttl = "Cumulative PC factor returns (PC1–3)"
    else:
        ttl = "PC scores over time (PC1–3) — where the market sits in each factor"
    st.plotly_chart(px.line(scores, title=ttl, height=400), width="stretch")

    st.markdown("**Rolling PC1 share** — how much one factor dominates (rising = concentrated / fragile market)")
    rw = 126 if pmode == "Daily returns" else 252
    ar = rolling_pc1(mtime, pmode, log_lv, rw, 10)
    ar = ar[ar.index >= start]
    st.plotly_chart(px.line(ar * 100, height=350, labels={"value":"%"},
                    title=f"Rolling {rw}-day PC1 share % ({pmode.lower()})"), width="stretch")
