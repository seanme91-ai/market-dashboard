"""
insights.py - runs after update_data.py in GitHub Actions.
Builds statistical summaries, asks Gemini for insights, saves data/insights.json:
  ratios  : AI insight for the Ratios & Spreads tab
  pca     : AI insight for the PCA / Regime tab
  news    : AI summary of critical events (from RSS headlines)
  headlines: raw headline list (title, source, link, time)
Needs repository secret GEMINI_API_KEY.
"""
import os, json, time, datetime as dt, email.utils, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests
from analytics import load_csv, derived, zscore_1y, pca_input, run_pca, pc1_share, GROUPS, RATES

# Models tried in order (first that works is used). Change here to switch model.
MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS",
          "gemini-3.8-flash,gemini-3.5-flash,gemini-2.5-flash").split(",") if m.strip()]
API_KEY = os.getenv("GEMINI_API_KEY", "")
OUT = "data/insights.json"

FEEDS = {
 "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
 "CNBC World Markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
 "CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
 "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
 "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
 "Google News - Markets": "https://news.google.com/rss/search?q=(stocks+OR+%22central+bank%22+OR+fed+OR+oil+OR+tariff+OR+inflation)+when:2d&hl=en-US&gl=US&ceid=US:en",
 "Google News - Asia": "https://news.google.com/rss/search?q=(china+OR+japan+OR+asia)+markets+when:2d&hl=en-US&gl=US&ceid=US:en",
}

SYSTEM = (
 "You are a senior cross-asset macro strategist writing a short daily note for an experienced "
 "investor and finance lecturer. Use ONLY the data provided; never invent numbers, events or causes "
 "that are not supported. When you suggest a possible driver, say it is a possible explanation. "
 "No price predictions or trade recommendations; you may mention what to monitor. "
 "Write concise Markdown: short headed sections with bullet points, cite the actual numbers. "
 "Do not include a title or a disclaimer."
)


# ---------------------------------------------------------------- Gemini
def gemini(prompt, max_tokens=4096):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    body = {"system_instruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_tokens}}
    last_err = None
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, timeout=180,
                                  headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"})
                if r.status_code == 429 or r.status_code >= 500:      # rate limit / busy: wait, retry
                    last_err = f"{model}: HTTP {r.status_code}"; time.sleep(20 * (attempt + 1)); continue
                if r.status_code != 200:                               # e.g. model not available: next model
                    last_err = f"{model}: HTTP {r.status_code} {r.text[:200]}"; break
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                if text:
                    return text, model
                last_err = f"{model}: empty response"; break
            except Exception as e:
                last_err = f"{model}: {e}"; time.sleep(5)
    raise RuntimeError(last_err)


# ---------------------------------------------------------------- summaries
def fmt(x, nd=3):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def ratios_summary(df):
    d = derived(df)
    z = zscore_1y(d).iloc[-1]
    last = d.index.max()
    rows = []
    for c in d.columns:
        s = d[c].dropna()
        if len(s) < 30: continue
        cur = s.iloc[-1]
        def ago(days):
            p = s[s.index <= last - pd.Timedelta(days=days)]
            return p.iloc[-1] if len(p) else np.nan
        is_spread = "-" in c and "/" not in c or c == "Soy crush"
        ch = (lambda a: cur - a) if is_spread else (lambda a: (cur / a - 1) * 100)
        rows.append({"series": c, "type": "spread (abs change)" if is_spread else "ratio (% change)",
                     "last": round(cur, 4), "z_1y": round(z.get(c, np.nan), 2),
                     "pctile_full_hist": round((s < cur).mean() * 100),
                     "chg_1M": round(ch(ago(30)), 2), "chg_3M": round(ch(ago(91)), 2),
                     "chg_1Y": round(ch(ago(365)), 2),
                     "hist_from": str(s.index.min().date())})
    t = pd.DataFrame(rows).sort_values("z_1y")
    return t.to_csv(index=False)


def pca_summary(df):
    out = []
    for mode in ["Daily returns", "Levels (standardized)"]:
        X = pca_input(df, mode)
        Xw = X[X.index >= X.index.max() - pd.Timedelta(days=365)]
        Z, ev, L = run_pca(Xw, 0.9)
        out.append(f"### {mode} PCA, 1-year window ({Z.shape[1]} assets, {len(Z)} days)")
        out.append("Explained variance PC1-5 (%): " + ", ".join(f"{v*100:.1f}" for v in ev[:5]))
        for k in ["PC1", "PC2", "PC3"]:
            ld = L[k].sort_values()
            out.append(f"{k} most negative: " + "; ".join(f"{i} {v:+.2f}" for i, v in ld.head(7).items()))
            out.append(f"{k} most positive: " + "; ".join(f"{i} {v:+.2f}" for i, v in ld.tail(7)[::-1].items()))
        sc = pd.DataFrame(Z.values @ L.iloc[:, :3].values, index=Z.index, columns=["PC1","PC2","PC3"])
        if mode == "Daily returns":
            cum = sc.cumsum()
            out.append("Cumulative factor score change, last 1M / 3M: " + "; ".join(
                f"{k} {cum[k].iloc[-1]-cum[k].iloc[-22]:+.1f} / {cum[k].iloc[-1]-cum[k].iloc[-64]:+.1f}"
                for k in cum))
            n, win = len(X), 126
            shares = {lab: pc1_share(X, n - off, win) for lab, off in
                      [("now", 0), ("1M ago", 21), ("3M ago", 63), ("6M ago", 126), ("1Y ago", 252)]}
            hist = [pc1_share(X, t, win) for t in range(win, n, 21)]
            hist = pd.Series(hist).dropna()
            out.append("Rolling 126-day PC1 share (%): " + ", ".join(
                f"{k} {v*100:.1f}" for k, v in shares.items()) +
                f". Full-history range {hist.min()*100:.1f}-{hist.max()*100:.1f}, median {hist.median()*100:.1f}.")
        else:
            out.append("Current PC score vs its 1Y range (0=low,100=high): " + "; ".join(
                f"{k} {((sc[k] < sc[k].iloc[-1]).mean()*100):.0f}" for k in sc))
        out.append("")
    return "\n".join(out)


def movers_summary(df):
    last = df.index.max()
    cur = df.iloc[-1]
    def back(days): return df[df.index <= last - pd.Timedelta(days=days)].iloc[-1]
    rows = []
    for g, cols in GROUPS.items():
        for c in [c for c in cols if c in df]:
            f = (lambda a: cur[c] - a) if c in RATES else (lambda a: (cur[c] / a - 1) * 100)
            rows.append((c, g, f(df[c].iloc[-2]), f(back(7)[c])))
    t = pd.DataFrame(rows, columns=["asset", "group", "1D", "1W"]).dropna()
    px_ = t[~t.asset.isin(RATES)]
    lines = ["Biggest 1-day % moves: " + "; ".join(f"{r.asset} {r._3:+.2f}%" for r in
             px_.reindex(px_["1D"].abs().sort_values(ascending=False).index).head(12).itertuples()),
             "Biggest 1-week % moves: " + "; ".join(f"{r.asset} {r._4:+.2f}%" for r in
             px_.reindex(px_["1W"].abs().sort_values(ascending=False).index).head(12).itertuples()),
             "Rates 1-week change (pp): " + "; ".join(f"{r.asset} {r._4:+.2f}" for r in
             t[t.asset.isin(RATES)].itertuples())]
    return f"Data as of {last.date()}\n" + "\n".join(lines)


# ---------------------------------------------------------------- news
def fetch_headlines(max_age_h=48, per_feed=25):
    now = dt.datetime.now(dt.timezone.utc)
    items, seen = [], set()
    for src, url in FEEDS.items():
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 market-dashboard"})
            root = ET.fromstring(r.content)
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = it.findtext("pubDate")
                try: t = email.utils.parsedate_to_datetime(pub).astimezone(dt.timezone.utc)
                except Exception: t = None
                if not title or (t and (now - t).total_seconds() > max_age_h * 3600): continue
                key = title.lower()[:70]
                if key in seen: continue
                seen.add(key)
                srcname = src
                if src.startswith("Google News") and " - " in title:     # Google appends " - Publisher"
                    title, srcname = title.rsplit(" - ", 1)
                items.append({"title": title, "source": srcname, "link": link,
                              "time": t.isoformat() if t else None})
                if sum(1 for i in items if i["source"] == srcname) >= per_feed: break
        except Exception as e:
            print("Feed failed:", src, e)
    items.sort(key=lambda x: x["time"] or "", reverse=True)
    return items[:150]


# ---------------------------------------------------------------- main
def main():
    df = load_csv()
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
           "data_date": str(df.index.max().date())}

    jobs = {
     "ratios": ("Below are cross-asset ratios and spreads (CSV). z_1y = z-score vs last 252 trading days; "
                "pctile_full_hist = percentile vs all available history (up to 20 years); changes are % for "
                "ratios and absolute for spreads (rates spreads in percentage points).\n\n"
                + ratios_summary(df) +
                "\n\nWrite: 1) **Most stretched** - the 4-6 most extreme readings, combining z-score and "
                "long-history percentile, and whether they agree; 2) **What the pattern says** - the macro "
                "story these ratios tell together (risk appetite, growth vs defensives, inflation, "
                "commodities, rate differentials); 3) **Watch** - 3 things worth monitoring. Max ~250 words."),
     "pca": ("Below are PCA results on a cross-asset universe (equities, bond ETFs, commodities, FX, rates). "
             "Loadings sign convention: S&P 500 positive on every PC.\n\n" + pca_summary(df) +
             "\n\nWrite: 1) **What each main factor represents** - interpret PC1-PC3 from the loadings for "
             "BOTH returns and levels PCA (e.g. global risk-on, USD, rates/duration, commodity/inflation); "
             "2) **Regime read** - what the PC1 share and its trend say about market concentration and "
             "fragility versus history, and recent factor moves; 3) **Watch** - 2-3 signals. Max ~300 words."),
    }

    headlines = fetch_headlines()
    res["headlines"] = headlines[:40]
    print(f"Fetched {len(headlines)} headlines")
    if headlines:
        hl = "\n".join(f"- [{h['source']}] {h['title']} ({(h['time'] or '')[:16]})" for h in headlines)
        jobs["news"] = (
            "Recent market headlines from the last 48 hours:\n" + hl +
            "\n\nObserved market moves:\n" + movers_summary(df) +
            "\n\nSelect the 5-8 MOST market-critical events (central banks, macro data, geopolitics, "
            "policy/tariffs, major corporate or credit events, commodity supply). For each give a bold "
            "short title, one sentence on what happened (in your own words), and which assets it affects, "
            "linking to the observed moves where they match. Mark each with 🔴 high / 🟠 medium impact. "
            "Then a short **Upcoming / watch** list only if the headlines mention scheduled events. "
            "Use only events in the headlines; do not add outside events.")

    for key, prompt in jobs.items():
        try:
            text, model = gemini(prompt)
            res[key] = text; res["model"] = model
            print(f"{key}: OK ({model}, {len(text)} chars)")
        except Exception as e:
            print(f"{key}: FAILED - {e}")
            if key in old:                        # keep yesterday's text, flagged as stale
                res[key] = old[key]; res[f"{key}_stale"] = old.get("generated_utc")
        time.sleep(8)                             # stay well inside free-tier per-minute limits

    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
