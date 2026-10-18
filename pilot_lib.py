# ---- Pair construction and estimation (no model code) ----
import numpy as np
import pandas as pd

NIH_FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
                "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
                "Pneumonia", "Pneumothorax"]


def add_label(df, finding):
    out = df.copy()
    out["label"] = out["Finding Labels"].str.split("|").apply(lambda L: int(finding in L))
    return out


def build_anchor_triples(df, finding, same_view=True, max_anchors_per_patient=2, seed=0,
                         match_gap=False, gap_tol=1, control="anchor"):
    """For each anchor image A, find in the SAME patient:
       flip partner  F: another study with the opposite label for `finding` (should change the answer)
       same partner  S: another study with the same label (should not change the answer)
    Partners are the nearest follow-ups (smallest |follow-up gap|) and share A's view if same_view.
    Using a same-patient, different-day image as the control means both arms carry the same kind
    of nuisance change (positioning, exposure, time), so R = S - N subtracts it."""
    rng = np.random.default_rng(seed)
    d = add_label(df, finding)
    key = ["Patient ID", "View Position"] if same_view else ["Patient ID"]
    d = d[d.groupby(key)["Image Index"].transform("size") >= 3]
    rows = []
    for k, g in d.groupby(key):
        pid = k[0] if isinstance(k, tuple) else k
        img = g["Image Index"].to_numpy(); lab = g["label"].to_numpy()
        fu = g["Follow-up #"].to_numpy(); view = g["View Position"].to_numpy()
        if lab.min() == lab.max():
            continue
        anchors = []
        for i in range(len(g)):
            other = np.arange(len(g)) != i
            fl = np.where(other & (lab != lab[i]))[0]
            sm = np.where(other & (lab == lab[i]))[0]
            if len(fl) == 0 or len(sm) == 0:
                continue
            f = fl[np.abs(fu[fl] - fu[i]).argmin()]
            if control == "pair":
                # control = a same-label PAIR (u, v) of this patient with EXACTLY the flip pair's signed
                # follow-up gap; prefer the anchor's status, then the pair closest in time to the anchor
                k = fu[f] - fu[i]
                best = None
                for u in range(len(g)):
                    for v in np.where((fu - fu[u]) == k)[0]:
                        if v == u or lab[u] != lab[v] or {u, v} == {i, f}:
                            continue
                        key_ = (lab[u] != lab[i], abs(fu[u] - fu[i]))
                        if best is None or key_ < best[0]:
                            best = (key_, u, v)
                if best is None:
                    continue
                _, u, v = best
                anchors.append(dict(patient=pid, view=view[i], img_a=img[i], lab_a=int(lab[i]),
                                    img_f=img[f], lab_f=int(lab[f]), img_s1=img[u], img_s=img[v],
                                    lab_s=int(lab[v]), gap_f=int(abs(k)), gap_s=int(abs(fu[v] - fu[u])),
                                    sgap_f=int(k), sgap_s=int(fu[v] - fu[u])))
                continue
            if match_gap:
                # control at the same signed time offset as the flip partner (same direction in time,
                # similar distance), so time/positioning drift is equal in both arms
                off = (fu[sm] - fu[i]) - (fu[f] - fu[i])
                s = sm[np.abs(off).argmin()]
                if abs((fu[s] - fu[i]) - (fu[f] - fu[i])) > gap_tol:
                    continue
            else:
                s = sm[np.abs(fu[sm] - fu[i]).argmin()]
            anchors.append(dict(patient=pid, view=view[i], img_a=img[i], lab_a=int(lab[i]),
                                img_f=img[f], lab_f=int(lab[f]), img_s=img[s], lab_s=int(lab[s]),
                                gap_f=int(abs(fu[f] - fu[i])), gap_s=int(abs(fu[s] - fu[i])),
                                sgap_f=int(fu[f] - fu[i]), sgap_s=int(fu[s] - fu[i])))
        if anchors:
            idx = rng.permutation(len(anchors))[:max_anchors_per_patient]
            rows += [anchors[i] for i in idx]
    if same_view and rows:   # a patient can appear in two view groups: keep the per-patient cap
        t = pd.DataFrame(rows).sample(frac=1, random_state=seed)
        rows = t.groupby("patient").head(max_anchors_per_patient).to_dict("records")
    t = pd.DataFrame(rows)
    if len(t):
        t["direction"] = np.where(t["lab_a"] == 0, "absent->present", "present->absent")
    return t


def flip_counts(df, findings=NIH_FINDINGS, same_view=True):
    """Feasibility table: per finding, patients with a within-patient flip (each direction),
    and anchors with BOTH a flip and a same-label partner."""
    out = []
    key = ["Patient ID", "View Position"] if same_view else ["Patient ID"]
    for f in findings:
        d = add_label(df, f).sort_values(key + ["Follow-up #"])
        diff = d.groupby(key)["label"].diff()
        up = d.loc[diff == 1, "Patient ID"].nunique()      # consecutive studies, same view
        down = d.loc[diff == -1, "Patient ID"].nunique()
        trip = build_anchor_triples(df, f, same_view=same_view, max_anchors_per_patient=1)
        out.append(dict(finding=f, patients_absent_to_present=up, patients_present_to_absent=down,
                        patients_with_flip_and_same_partner=len(trip),
                        of_which_anchor_absent=int((trip["lab_a"] == 0).sum()) if len(trip) else 0,
                        of_which_anchor_present=int((trip["lab_a"] == 1).sum()) if len(trip) else 0))
    return pd.DataFrame(out)


def sample_balanced(trip, n, seed=0):
    """Balance the two flip directions so a yes/no bias cannot masquerade as reliance."""
    parts = [g.sample(min(len(g), n // 2), random_state=seed) for _, g in trip.groupby("direction")]
    return pd.concat(parts).reset_index(drop=True)


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def item_quantities(res, scale="prob"):
    """res needs p_a, p_f, p_s (P(yes) under anchor, flip partner, same-label partner).
    scale='prob' uses probability differences; scale='logit' uses log-odds differences
    (less compressed near 0/1, makes models with different calibration more comparable)."""
    r = res.copy()
    if scale == "logit":
        for c in ["p_a", "p_f", "p_s", "p_s1"]:
            if c in r:
                r[c] = _logit(r[c].to_numpy())
    r["d_change"] = (r["p_f"] - r["p_a"]).abs()          # sensitivity (should be large)
    ref = r["p_s1"] if "p_s1" in r else r["p_a"]          # pair control: |p(C2) - p(C1)|
    r["d_preserve"] = (r["p_s"] - ref).abs()             # nuisance sensitivity (should be small)
    r["R"] = r["d_change"] - r["d_preserve"]             # evidence-specific reliance
    dy = r["lab_f"] - r["lab_a"]                          # +1 absent->present, -1 present->absent
    r["signed_dir"] = (r["p_f"] - r["p_a"]) * dy          # >0 = moved the right way
    # Directional, time-matched estimand (primary): change toward the new label on the flip pair,
    # minus the same signed change on a same-label pair of the same patient at the same time offset
    r["D_flip"] = r["signed_dir"]
    r["D_ctrl"] = (r["p_s"] - ref) * dy
    r["R_dir"] = r["D_flip"] - r["D_ctrl"]
    # Sign version: only the DIRECTION of movement counts, so any monotone recalibration of the
    # model's scores (saturation, temperature) leaves it unchanged. Ties count 0.
    r["S_flip"] = np.sign(r["p_f"] - r["p_a"]) * dy
    r["S_ctrl"] = np.sign(r["p_s"] - ref) * dy
    r["R_sign"] = r["S_flip"] - r["S_ctrl"]
    return r


DEFAULT_COLS = ("D_flip", "D_ctrl", "R_dir", "S_flip", "S_ctrl", "R_sign", "d_change", "d_preserve", "R")


def cluster_bootstrap(r, cols=DEFAULT_COLS, B=2000, seed=0, by="patient"):
    """Point estimates (mean of patient means) and percentile CIs resampling PATIENTS."""
    rng = np.random.default_rng(seed)
    cols = [c for c in cols if c in r]
    pm = r.groupby(by)[cols].mean()
    est = pm.mean()
    boots = np.empty((B, len(cols)))
    n = len(pm)
    arr = pm.to_numpy()
    for b in range(B):
        boots[b] = arr[rng.integers(0, n, n)].mean(axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    p_le0 = (boots <= 0).mean(axis=0)               # one-sided bootstrap p for H0: quantity <= 0
    return pd.DataFrame({"estimate": est.values, "ci_low": lo, "ci_high": hi, "p_one_sided": p_le0,
                         "n_patients": n}, index=cols)


def summarize(r, B=2000, seed=0):
    from scipy.stats import binomtest
    overall = cluster_bootstrap(r, B=B, seed=seed)
    by_dir = {k: cluster_bootstrap(g, B=B, seed=seed) for k, g in r.groupby("direction")}
    pm = r.groupby("patient")["R_dir" if "R_dir" in r else "R"].mean()
    nz = pm[pm != 0]
    sign_p = binomtest(int((nz > 0).sum()), len(nz), 0.5, alternative="greater").pvalue if len(nz) else np.nan
    acc = ((r["p_a"] > 0.5).astype(int) == r["lab_a"]).mean()
    return overall, by_dir, sign_p, acc


def paired_difference(r1, r2, col="R", B=4000, seed=0):
    """Difference in a per-item quantity between two models scored on the SAME anchors,
    with a patient-level paired bootstrap. Returns estimate, CI and two-sided bootstrap p."""
    rng = np.random.default_rng(seed)
    a = r1.groupby("patient")[col].mean(); b = r2.groupby("patient")[col].mean()
    a, b = a.align(b, join="inner")
    d = (a - b).to_numpy(); n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(estimate=d.mean(), ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                p_boot=min(p, 1.0), n_patients=n)


def gap_balance(t):
    """Check that the control is time-matched to the flip partner."""
    return pd.Series({"mean_gap_flip": t["gap_f"].mean(), "mean_gap_same": t["gap_s"].mean(),
                      "share_same_time_direction": (np.sign(t["sgap_f"]) == np.sign(t["sgap_s"])).mean()
                      if "sgap_f" in t else np.nan})
