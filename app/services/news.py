import time
import hashlib
import pandas as pd
import feedparser
import requests
from datetime import datetime, timedelta

WHITELIST = {
    "ecb.europa.eu",
    "bankofengland.co.uk",
    "boj.or.jp",
    "snb.ch",
    "oecd.org",
    "imf.org",
    "bis.org",
    "fredblog.stlouisfed.org",
    "bls.gov",
    "bea.gov",
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "forexlive.com",
    "cnbc.com",
    "www.cnbc.com",
    "dailyfx.com",
    "www.dailyfx.com",
    "investing.com",
    "finance.yahoo.com",
}

PAIR_DOMAINS = {
    "EUR/USD": ["ecb.europa.eu", "bankofengland.co.uk", "reuters.com"],
    "GBP/USD": ["bankofengland.co.uk", "reuters.com"],
    "USD/JPY": ["boj.or.jp", "reuters.com"],
    "USD/CHF": ["snb.ch", "reuters.com"],
    "XAU/USD": [
        "reuters.com",
        "ft.com",
        "forexlive.com",
        "cnbc.com",
        "dailyfx.com",
        "investing.com",
        "finance.yahoo.com",
    ],
    "AUD/USD": ["reuters.com"],
    "NZD/USD": ["reuters.com"],
    "USD/CAD": ["reuters.com"],
}

def _gdelt_query(hours: int):
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    ts = start.strftime("%Y%m%d%H%M%S")
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query=central+bank+OR+inflation+OR+rate&mode=TimelineVol&startdatetime={ts}&sourcelang=english&format=json"
    try:
        r = requests.get(url, timeout=20)
        return r.json()
    except Exception:
        return {}

def _rss_fetch(domains: list[str], hours: int):
    items = []
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    feeds = []
    for d in domains:
        if d == "ecb.europa.eu":
            feeds.append("https://www.ecb.europa.eu/press/govcdec/rss/govcdec.rss")
        if d == "bankofengland.co.uk":
            feeds.append("https://www.bankofengland.co.uk/news?NewsCategories=65c2802d1d6945b498c61577e9a33c81&Direction=Descending&Take=20&Offset=0&from=01/01/2000&to=12/31/2099&format=rss")
        if d == "boj.or.jp":
            feeds.append("https://www.boj.or.jp/en/rss/press_all.xml")
        if d == "snb.ch":
            feeds.append("https://www.snb.ch/en/rss/\n")
        if d == "reuters.com":
            feeds.append("https://feeds.reuters.com/reuters/businessNews")
        if d == "ft.com":
            feeds.append("https://www.ft.com/?format=rss")
        if d in ("cnbc.com", "www.cnbc.com"):
            feeds.append("https://www.cnbc.com/id/100003114/device/rss/rss.html")
        if d in ("dailyfx.com", "www.dailyfx.com"):
            feeds.append("https://www.dailyfx.com/feeds/latest")
        if d == "forexlive.com":
            feeds.append("https://www.forexlive.com/feed/news")
        if d == "investing.com":
            feeds.append("https://www.investing.com/rss/news.rss")
        if d == "finance.yahoo.com":
            feeds.append("https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US")
    for f in feeds:
        d = feedparser.parse(f)
        for e in d.entries:
            published = e.get("published") or e.get("updated")
            try:
                dt = datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") else cutoff
            except Exception:
                dt = cutoff
            if dt >= cutoff:
                url = e.get("link", "")
                domain = url.split("/")[2] if "//" in url else ""
                if domain in WHITELIST:
                    items.append({"title": e.get("title", ""), "url": url, "domain": domain, "time": dt.isoformat()})
    return pd.DataFrame(items)

def news_summary(pair: str, hours: int):
    domains = PAIR_DOMAINS.get(pair, [])
    rss = _rss_fetch(domains, hours)
    gd = _gdelt_query(hours)
    return {"rss": rss.to_dict(orient="records"), "gdelt": gd}
