"""Recompute the reliance estimates of notebook 04 (report Table 2) from the saved model scores.

No GPU and no X-ray images are needed: the script only reads
  data_derived/triples/triples04_<finding>.csv   (anchor, flip and control images per patient)
  data_derived/scores/scores04_*.json            (model outputs per image)

Usage (from the repository root):
    python scripts/reproduce_table2.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pilot_lib import item_quantities, cluster_bootstrap  # noqa: E402

FINDINGS = ["Effusion", "Atelectasis", "Pneumothorax", "Cardiomegaly"]
VLMS = ["Qwen2.5-VL-7B-Instruct", "medgemma-4b-it"]
ROLES = {"a": "img_a", "f": "img_f", "s1": "img_s1", "s": "img_s"}
MASS_MIN = 0.9


def load(name):
    with open(os.path.join(ROOT, "data_derived", "scores", name)) as fh:
        return json.load(fh)


def estimate(samp, score, mass_min=None):
    """score: image -> (P(finding present), yes/no mass)."""
    s = samp[samp[list(ROLES.values())].apply(lambda r: all(k in score for k in r), axis=1)].copy()
    for c, col in ROLES.items():
        s[f"p_{c}"] = s[col].map(lambda k: score[k][0])
        s[f"m_{c}"] = s[col].map(lambda k: score[k][1])
    if mass_min is not None:
        s = s[s[[f"m_{c}" for c in ROLES]].min(axis=1) >= mass_min]
    r = item_quantities(s, scale="prob")
    b = cluster_bootstrap(r, cols=["R_dir", "R_sign"], B=2000, seed=0)
    return dict(n_patients=r["patient"].nunique(),
                R_dir=b.loc["R_dir", "estimate"], lo=b.loc["R_dir", "ci_low"], hi=b.loc["R_dir", "ci_high"],
                R_sign=b.loc["R_sign", "estimate"])


def main():
    xrv = load("scores04_xrv.json")
    rows = []
    for F in FINDINGS:
        samp = pd.read_csv(os.path.join(ROOT, "data_derived", "triples", f"triples04_{F}.csv"))
        rows.append(dict(finding=F, model="DenseNet (CheXpert)", anchors="all", mass_filter="-",
                         **estimate(samp, {k: (v[F], 1.0) for k, v in xrv.items()})))
        vs = samp[samp["vlm"]]
        for m in VLMS:
            sc = load(f"scores04_{F}_{m}_P0.json")
            for mm in [None, MASS_MIN]:
                rows.append(dict(finding=F, model=m, anchors="vlm", mass_filter=mm or "-", **estimate(vs, sc, mm)))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
