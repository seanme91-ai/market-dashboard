"""
update_data.py  -  builds / updates data/daily_update.csv (+ .xlsx copy)
First run  : back-fills HISTORY_YEARS of history (also re-runs automatically
             if the saved file starts later than HISTORY_YEARS ago).
Later runs : re-downloads the last 15 days and overwrites/extends those rows,
             so missed days and late revisions are fixed automatically.
"""
import io, os, datetime as dt
import pandas as pd, requests, yfinance as yf

HISTORY_YEARS = 20
CSV  = "data/daily_update.csv"
XLSX = "data/Daily Update.xlsx"

YAHOO = {
 "S&P 500":"^GSPC","Nasdaq-100":"^NDX","Dow Jones":"^DJI","Russell 2000":"^RUT",
 "STOXX Europe 600":"^STOXX","DAX":"^GDAXI","FTSE 100":"^FTSE","CAC 40":"^FCHI",
 "Nikkei 225":"^N225","Hang Seng":"^HSI","CSI 300":"000300.SS","Shanghai Composite":"000001.SS",
 "KOSPI":"^KS11","Nifty 50":"^NSEI","MSCI World (proxy URTH)":"URTH","MSCI EM (proxy EEM)":"EEM",
 "US Aggregate (AGG)":"AGG","Global Agg ex-US (BNDX)":"BNDX","US Treasury 7-10Y (IEF)":"IEF",
 "US Treasury 20Y+ (TLT)":"TLT","US Investment Grade (LQD)":"LQD","US High Yield (HYG)":"HYG",
 "EM USD Bonds (EMB)":"EMB","EM Local Bonds (EMLC)":"EMLC",
 "Gold":"GC=F","Silver":"SI=F","Platinum":"PL=F","Palladium":"PA=F","Copper":"HG=F",
 "Aluminum":"ALI=F","Iron Ore":"TIO=F","WTI Crude":"CL=F","Brent Crude":"BZ=F",
 "Henry Hub Nat Gas":"NG=F","TTF Europe Nat Gas":"TTF=F","Gasoline (RBOB)":"RB=F",
 "Heating Oil / Diesel":"HO=F","Uranium":"URA",
 "Corn":"ZC=F","Wheat":"ZW=F","Soybeans":"ZS=F","Rice":"ZR=F","Sugar":"SB=F","Coffee":"KC=F",
 "Cocoa":"CC=F","Cotton":"CT=F","Orange Juice":"OJ=F","Live Cattle":"LE=F","Lean Hogs":"HE=F",
 "Soybean Oil":"ZL=F","Soybean Meal":"ZM=F",
 "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","AUDUSD":"AUDUSD=X","NZDUSD":"NZDUSD=X",
 "USDJPY":"JPY=X","USDCHF":"CHF=X","USDCAD":"CAD=X","USDCNY":"CNY=X","USDHKD":"HKD=X",
 "USDSGD":"SGD=X","EURGBP":"EURGBP=X","EURJPY":"EURJPY=X","GBPJPY":"GBPJPY=X",
 "AUDJPY":"AUDJPY=X","EURCHF":"EURCHF=X",
}
FRED = {
 "Fed Funds":"DFF","ECB Deposit Rate":"ECBDFR","BoJ Policy Rate":"IRSTCB01JPM156N",
 "SOFR":"SOFR","US 2Y Treasury":"DGS2","US 10Y Treasury":"DGS10","US 30Y Treasury":"DGS30",
 "German Bund 10Y":"IRLTLT01DEM156N","JGB 10Y":"IRLTLT01JPM156N","UK Gilt 10Y":"IRLTLT01GBM156N",
 "US 10Y-2Y Spread":"T10Y2Y",
}
COLUMNS = list(YAHOO) + list(FRED)   # Nickel, Zinc, Coal, Palm Oil: no free daily source


def get_yahoo(start):
    raw = yf.download(list(YAHOO.values()), start=start, interval="1d",
                      auto_adjust=False, progress=False, threads=True)["Close"]
    inv = {v: k for k, v in YAHOO.items()}
    df = raw.rename(columns=inv)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def get_fred(sid, start):
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        s = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text))
        s.columns = ["Date", "v"]
        s["Date"] = pd.to_datetime(s["Date"])
        s["v"] = pd.to_numeric(s["v"], errors="coerce")
        s = s.dropna().set_index("Date")["v"]
        return s[s.index >= pd.Timestamp(start) - pd.Timedelta(days=120)]  # buffer for monthly data
    except Exception as e:
        print("FRED failed:", sid, e)
        return pd.Series(dtype=float)


def main():
    os.makedirs("data", exist_ok=True)
    old = pd.read_csv(CSV, parse_dates=["Date"], index_col="Date") if os.path.exists(CSV) else None
    full_start = dt.date.today() - dt.timedelta(days=round(365.25 * HISTORY_YEARS))
    # full back-fill if no file yet, or the file doesn't reach back far enough
    if old is None or old.empty or old.index.min() > pd.Timestamp(full_start) + pd.Timedelta(days=30):
        start, old = full_start, None
    else:
        start = dt.date.today() - dt.timedelta(days=15)
    print("Downloading from", start)

    new = get_yahoo(start)
    new = new[new.index.dayofweek < 5]                  # weekdays only
    for col, sid in FRED.items():
        s = get_fred(sid, start)
        new[col] = s.reindex(new.index.union(s.index)).ffill().reindex(new.index)

    new = new.reindex(columns=COLUMNS).ffill()          # carry forward market holidays
    new = new[new.index >= pd.Timestamp(start)].dropna(how="all")

    if old is not None:
        old = old.reindex(columns=COLUMNS)
        out = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    else:
        out = new
    out.index.name = "Date"
    out = out.round(6)
    out.to_csv(CSV)
    out.to_excel(XLSX, sheet_name="Sheet1")
    print(f"Saved {len(out)} rows, {out.index.min().date()} to {out.index.max().date()}")
    print("First available date per column:")
    print(out.apply(lambda c: c.first_valid_index()).dt.date.to_string())


if __name__ == "__main__":
    main()
