#!/usr/bin/env python3
# poll_assignments.py
# Poll Football Zebras Week 5 page until matchups appear, then write CSV.

import re, time, sys, csv, html, urllib.request
from html.parser import HTMLParser
from datetime import datetime

URL   = "https://www.footballzebras.com/2025/09/week-6-referee-assignments-6/"
WEEK  = 6
SEASON= 2025
OUT   = "assignments_week5.csv"
POLL_SECONDS = 120  # every 2 minutes

TEAM = {
    "Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR","Bears":"CHI","Bengals":"CIN",
    "Browns":"CLE","Cowboys":"DAL","Broncos":"DEN","Lions":"DET","Packers":"GB","Texans":"HOU","Colts":"IND",
    "Jaguars":"JAX","Chiefs":"KC","Raiders":"LV","Chargers":"LAC","Rams":"LAR","Dolphins":"MIA","Vikings":"MIN",
    "Patriots":"NE","Saints":"NO","Giants":"NYG","Jets":"NYJ","Eagles":"PHI","Steelers":"PIT","49ers":"SF",
    "Seahawks":"SEA","Buccaneers":"TB","Titans":"TEN","Commanders":"WAS"
}
def last_token(s): return s.strip().split()[-1]

class ArticleText(HTMLParser):
    def __init__(self): super().__init__(); self.in_article=False; self.buf=[]
    def handle_starttag(self, tag, attrs):
        if tag=="article": self.in_article=True
        if self.in_article and tag in ("br","p","li","h2","h3"): self.buf.append("\n")
    def handle_endtag(self, tag):
        if tag=="article": self.in_article=False
        if self.in_article and tag in ("p","li","h2","h3","div"): self.buf.append("\n")
    def handle_data(self, data):
        if self.in_article:
            t = data.replace("\xa0"," ").strip()
            if t: self.buf.append(t+" ")

def fetch_article_lines():
    req = urllib.request.Request(URL, headers={"User-Agent":"Mozilla/5.0", "Accept-Language":"en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode("utf-8","ignore")
    raw = html.unescape(raw)
    p = ArticleText(); p.feed(raw); text = "".join(p.buf)
    # canonicalize lines
    lines = [re.sub(r"\s+"," ",l).strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    return lines

def parse_rows(lines):
    is_date = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,\s+[A-Za-z]+\.?\s+\d{1,2}$", re.I)
    is_match= re.compile(r"^([A-Za-z .]+?)\s+(at|vs\.)\s+([A-Za-z .]+?)$", re.I)
    is_time = re.compile(r"\b\d{1,2}(:\d{2})?\s*[ap]\.m\.\b", re.I)
    is_net  = re.compile(r"(Prime|Peacock|FOX|CBS|NFLN|ESPN(?:\s*ESPN\+)?|ABC|Amazon)", re.I)
    is_name = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")

    rows = []; curr_date = ""
    i = 0
    while i < len(lines):
        t = lines[i]
        if is_date.search(t): curr_date = t; i += 1; continue
        m = is_match.match(t)
        if not m: i += 1; continue

        away_raw, sep, home_raw = m.group(1), m.group(2).lower(), m.group(3)
        away = TEAM.get(last_token(away_raw), last_token(away_raw))
        home = TEAM.get(last_token(home_raw), last_token(home_raw))
        site = "neutral" if sep.startswith("vs") else "home"

        ref, tim, net = "", "", ""
        j = 1
        # greedy lookahead
        if i+j < len(lines) and (("referee" in lines[i+j].lower()) or is_name.match(lines[i+j])): ref = lines[i+j]; j += 1
        if i+j < len(lines) and is_time.search(lines[i+j]): tim = lines[i+j]; j += 1
        # network on same or next line
        if tim:
            after = is_time.sub("", tim).strip()
            if after and is_net.search(after): net = after
        if not net and i+j < len(lines) and is_net.search(lines[i+j]): net = lines[i+j]; j += 1

        rows.append({
            "week": WEEK, "game_date": curr_date, "kickoff_et": tim,
            "home_team": home, "away_team": away, "site_type": site,
            "network": net, "referee": re.sub(r"\s*is the referee.*$", "", ref, flags=re.I).strip()
        })
        i += j
    return rows

def write_csv(rows):
    hdr = ["week","game_date","kickoff_et","home_team","away_team","site_type","network","referee"]
    with open(OUT,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); [w.writerow(r) for r in rows]

def main():
    while True:
        lines = fetch_article_lines()
        rows = parse_rows(lines)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] parsed rows: {len(rows)}")
        # require a minimum to avoid false positives when they drip-feed
        if len(rows) >= 8:
            write_csv(rows)
            print(f"✅ Wrote {OUT} with {len(rows)} rows")
            return
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
