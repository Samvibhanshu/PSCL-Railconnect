"""OPTIONAL: snapshot real train positions/speeds from Fintraffic Digitraffic (Finland).

Not used by main.py and not needed to reproduce any reported number. It is
provided so that users can (i) inspect real speed and position-update
statistics and (ii) adapt the observation-noise model of the simulator.

Data source : https://rata.digitraffic.fi/  (REST/GraphQL; Swagger at /swagger/)
Licence     : Creative Commons Attribution 4.0 (CC BY 4.0), (c) Fintraffic / digitraffic.fi
Etiquette   : send a 'Digitraffic-User' header identifying your application.

Usage:  python ingest_digitraffic.py --out ../results/digitraffic_snapshot.csv
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

URL = "https://rata.digitraffic.fi/api/v1/train-locations/latest/"


def fetch(user: str):
    req = urllib.request.Request(URL, headers={"Digitraffic-User": user, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def to_frame(records):
    rows = []
    for rec in records:
        lon, lat = (rec.get("location") or {}).get("coordinates", [None, None])
        rows.append(dict(train_number=rec.get("trainNumber"), departure_date=rec.get("departureDate"),
                         timestamp=rec.get("timestamp"), speed_kmh=rec.get("speed"),
                         accuracy_m=rec.get("accuracy"), lon=lon, lat=lat))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="digitraffic_snapshot.csv")
    ap.add_argument("--user", default="railconflict-pscl/1.0 (research)")
    a = ap.parse_args()
    df = to_frame(fetch(a.user))
    df.to_csv(a.out, index=False)
    print(f"{len(df)} train locations written to {a.out}")
    print(df[["speed_kmh", "accuracy_m"]].describe())
