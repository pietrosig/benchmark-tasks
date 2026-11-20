"""Count CPSC NEISS emergency-department cases naming a knife in a dishwasher (context numbers in task.json).
Usage: python neiss_query.py neiss2019.xlsx ... neiss2023.xlsx  (yearly files from https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data;
needs pandas and python-calamine). Prints per year: cases, weighted national estimate, cases saying the knife was up, their weight."""
import sys, pandas as pd
tot = {"cases":0,"w":0,"upc":0,"upw":0}
for f in sys.argv[1:]:
    d = pd.read_excel(f, engine="calamine")
    n = d["Narrative_1"].fillna("").astype(str).str.upper()
    sel = (n.str.contains("DISHWASHER|DISH WASHER")) & n.str.contains("KNIFE|KNIVES")
    up = sel & n.str.contains("STICKING UP|FACING UP|POINTING UP|STRAIGHT UP|UPWARD|BLADE UP|POINT UP|TIP UP|POINTED UP|UPRIGHT")
    c, w, uc, uw = int(sel.sum()), float(d[sel]["Weight"].sum()), int(up.sum()), float(d[up]["Weight"].sum())
    print(f, c, round(w), uc, round(uw))
    for k,v in zip(tot,(c,w,uc,uw)): tot[k]+=v
    for t in n[up]: print("   UP:", t[:160])
print("TOTAL", {k: round(v) for k,v in tot.items()})
