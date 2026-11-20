"""Shared definitions and calculations used by app.py and insights.py."""
import numpy as np, pandas as pd

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


def load_csv(path="data/daily_update.csv"):
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
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


def zscore_1y(d):
    return (d - d.rolling(252).mean()) / d.rolling(252).std()


def pca_input(df, mode, log_lv=True):
    """PCA inputs (not yet standardized). mode: 'Daily returns' or 'Levels (standardized)'."""
    pcols = [c for c in df if c not in RATES]
    rcols = [c for c in df if c in RATES]
    if mode == "Daily returns":
        X = pd.concat([df[pcols].pct_change(fill_method=None), df[rcols].diff() / 100], axis=1)
    else:
        lv = df[pcols].where(df[pcols] > 0)
        X = pd.concat([np.log(lv) if log_lv else lv, df[rcols]], axis=1)
    return X.iloc[1:]


def run_pca(X, min_cov=0.9):
    """Standardize within window, PCA. Returns Z, explained-variance ratios, loadings."""
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


def pc1_share(X, end_idx, win):
    w = X.iloc[max(0, end_idx - win):end_idx]
    w = w.dropna(axis=1, thresh=int(len(w) * 0.9)).ffill().bfill()
    w = ((w - w.mean()) / w.std().replace(0, np.nan)).dropna(axis=1)
    if w.shape[1] < 5: return np.nan
    e = np.linalg.eigvalsh(np.cov(w.T))
    return e.max() / e.sum()
