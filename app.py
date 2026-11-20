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


if not os.path.exists(CSV):
    st.error("No data yet - run the GitHub Action once."); st.stop()
df = load(os.path.getmtime(CSV))
last = df.index.max()

# ---------- sidebar ----------
st.sidebar.title("📈 Market Dashboard")
st.sidebar.caption(f"Data to {last:%d %b %Y} · {len(df):,} rows")
period = st.sidebar.selectbox("Chart period", ["1M","3M","6M","YTD","1Y","3Y","5Y","All"], index=4)
offs = {"1M":30,"3M":91,"6M":182,"1Y":365,"3Y":1095,"5Y":1826}
start = pd.Timestamp(last.year,1,1) if period=="YTD" else (df.index.min() if period=="All"
        else last - pd.Timedelta(days=offs[period]))
view = df[df.index >= start]

tabs = st.tabs(["Overview","Charts","Ratios & Spreads","Rates","Correlation","PCA / Regime"])

# ---------- Overview ----------
with tabs[0]:
    def chg(days):
        past = df[df.index <= last - pd.Timedelta(days=days)].iloc[-1]
        return past
    cur = df.iloc[-1]
    ytd0 = df[df.index < pd.Timestamp(last.year,1,1)]
    ytd0 = ytd0.iloc[-1] if len(ytd0) else df.iloc[0]
    rows = []
    for g, cols in GROUPS.items():
        for c in [c for c in cols if c in df]:
            p1, p7, p30 = df[c].iloc[-2], chg(7)[c], chg(30)[c]
            f = (lambda a: cur[c]-a) if c in RATES else (lambda a: (cur[c]/a-1)*100)
            rows.append([g, c, cur[c], f(p1), f(p7), f(p30), f(ytd0[c])])
    tbl = pd.DataFrame(rows, columns=["Group","Asset","Last","1D","1W","1M","YTD"])
    st.caption("% change for prices; absolute change (percentage points) for rates")
    grp = st.selectbox("Group", ["All"] + list(GROUPS))
    t = tbl if grp=="All" else tbl[tbl.Group==grp]
    st.dataframe(
        t.style.format({"Last":"{:,.4g}","1D":"{:+.2f}","1W":"{:+.2f}","1M":"{:+.2f}","YTD":"{:+.2f}"})
         .background_gradient(subset=["1D","1W","1M","YTD"], cmap="RdYlGn", vmin=-5, vmax=5),
        hide_index=True, width="stretch", height=600)
    # quick heat bar for 1M moves (non-rates)
    bar = tbl[~tbl.Asset.isin(RATES)].sort_values("1M")
    st.plotly_chart(px.bar(bar, x="1M", y="Asset", color="1M", color_continuous_scale="RdYlGn",
                    range_color=[-10,10], height=1400, title="1-month % change"),
                    width="stretch")

# ---------- Charts ----------
with tabs[1]:
    g = st.selectbox("Asset class", list(GROUPS), key="cg")
    opts = [c for c in GROUPS[g] if c in df]
    sel = st.multiselect("Assets", opts, default=opts[:4])
    mode = st.radio("Show", ["Rebased to 100","Raw level"], horizontal=True)
    if sel:
        v = view[sel].dropna(how="all")
        if mode.startswith("Rebased") and g != "Rates (%)":
            v = v / v.bfill().iloc[0] * 100
        st.plotly_chart(px.line(v, height=500), width="stretch")

# ---------- Ratios ----------
with tabs[2]:
    d = derived(df)
    pick = st.selectbox("Ratio / spread", list(d.columns))
    s = d[pick][d.index >= start].dropna()
    z = (d[pick] - d[pick].rolling(252).mean()) / d[pick].rolling(252).std()
    c1, c2 = st.columns(2)
    c1.metric(pick, f"{s.iloc[-1]:,.4g}", f"{s.iloc[-1]-s.iloc[0]:+.4g} over period")
    c2.metric("1Y z-score", f"{z.iloc[-1]:+.2f}")
    fig = px.line(s, height=420); fig.add_hline(s.mean(), line_dash="dot")
    st.plotly_chart(fig, width="stretch")
    zt = ((d - d.rolling(252).mean()) / d.rolling(252).std()).iloc[-1].sort_values()
    st.plotly_chart(px.bar(zt, orientation="h", title="All ratios: 1Y z-score (extremes = stretched)",
                    height=550), width="stretch")

# ---------- Rates ----------
with tabs[3]:
    curve = [c for c in ["Fed Funds","US 2Y Treasury","US 10Y Treasury","US 30Y Treasury"] if c in df]
    fig = go.Figure()
    for lab, dd in [("Today", last), ("1M ago", last-pd.Timedelta(days=30)), ("1Y ago", last-pd.Timedelta(days=365))]:
        row = df[df.index <= dd].iloc[-1]
        fig.add_scatter(x=["FF","2Y","10Y","30Y"][:len(curve)], y=row[curve], mode="lines+markers", name=lab)
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
    corr = rets.corr()
    st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    height=750, title=f"{win}-day return correlation"), width="stretch")
    if "S&P 500" in df and "US Treasury 7-10Y (IEF)" in df:
        r = df[["S&P 500","US Treasury 7-10Y (IEF)"]].pct_change(fill_method=None)
        rc = r["S&P 500"].rolling(win).corr(r["US Treasury 7-10Y (IEF)"])
        f = px.line(rc[rc.index >= start], title="Rolling stock-bond correlation (S&P vs IEF)", height=350)
        f.add_hline(0); st.plotly_chart(f, width="stretch")

# ---------- PCA ----------
with tabs[5]:
    st.caption("PCA on daily returns (rates use daily changes). Rising PC1 share = assets moving "
               "together = risk-on/off regime dominating.")
    win = st.slider("PCA window (trading days)", 60, 504, 252, key="pw")
    px_cols = [c for c in df if c not in RATES]
    R = pd.concat([df[px_cols].pct_change(fill_method=None), df[[c for c in df if c in RATES]].diff()/100], axis=1)
    R = R.iloc[-win:].dropna(axis=1, thresh=int(win*0.9)).fillna(0)
    Z = (R - R.mean()) / R.std().replace(0, np.nan)
    Z = Z.dropna(axis=1)
    vals, vecs = np.linalg.eigh(np.cov(Z.T))
    idx = vals.argsort()[::-1]; vals, vecs = vals[idx], vecs[:, idx]
    ev = vals / vals.sum()
    c1, c2, c3 = st.columns(3)
    for i, c in enumerate([c1, c2, c3]): c.metric(f"PC{i+1} variance", f"{ev[i]*100:.1f}%")
    k = st.selectbox("Loadings of", ["PC1","PC2","PC3"])
    j = int(k[-1]) - 1
    load_ = pd.Series(vecs[:, j], index=Z.columns)
    if load_.get("S&P 500", 1) < 0: load_ = -load_      # sign convention: S&P positive
    load_ = load_.sort_values()
    top = pd.concat([load_.head(12), load_.tail(12)])
    st.plotly_chart(px.bar(top, orientation="h", height=650, title=f"{k} largest loadings"),
                    width="stretch")

    # rolling PC1 share (absorption ratio) - computed every 5 days for speed
    Rall = pd.concat([df[px_cols].pct_change(fill_method=None), df[[c for c in df if c in RATES]].diff()/100], axis=1)
    Rall = Rall.dropna(axis=1, thresh=int(len(Rall)*0.8))
    out = {}
    for t in range(126, len(Rall), 5):
        w = Rall.iloc[t-126:t].fillna(0)
        w = (w - w.mean()) / w.std().replace(0, np.nan)
        w = w.dropna(axis=1)
        e = np.linalg.eigvalsh(np.cov(w.T))
        out[Rall.index[t]] = e.max() / e.sum()
    ar = pd.Series(out)
    st.plotly_chart(px.line(ar[ar.index >= start]*100, height=350,
                    title="Rolling 6M PC1 share % (market concentration / fragility)"),
                    width="stretch")
