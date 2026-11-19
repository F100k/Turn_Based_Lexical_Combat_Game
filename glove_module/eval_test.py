import re, difflib, gzip, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

OUT = "/mnt/user-data/outputs"
plt.style.use("seaborn-v0_8-whitegrid")

# ---------- load GloVe 6B 100d ----------
# เปลี่ยน path เป็นชื่อไฟล์ของคุณ และเอา skiprows=1 ออก (เพราะไฟล์ GloVe มาตรฐานไม่มี header)
df_g = pd.read_csv("glove.6B.100d.txt", sep=" ", header=None, quoting=3, keep_default_na=False, dtype={0: str})
words = df_g[0].tolist()
E = df_g.iloc[:, 1:101].to_numpy(dtype=np.float32)
w2i = {w: i for i, w in enumerate(words)}
print("GloVe:", E.shape)

STOP = set("a an the of and or to in on for with is are was be by at as from that this it its".split())
SMALL_VOCAB = words[:50000]  # difflib candidate pool (OOV recovery) - speed

def recover(w):
    m = difflib.get_close_matches(w, SMALL_VOCAB, n=1, cutoff=0.6)
    return m[0] if m else None

def tokens(s):
    return re.findall(r"[a-z]+", str(s).lower())

def phrase_vec(s, recover_oov=True):
    vs, used, oov = [], [], []
    for t in tokens(s):
        if t in STOP: continue
        if t in w2i:
            vs.append(E[w2i[t]]); used.append(t)
        elif recover_oov:
            r = recover(t)
            if r: vs.append(E[w2i[r]]); used.append(f"{t}->{r}")
            else: oov.append(t)
        else: oov.append(t)
    if not vs: return None, used, oov
    v = np.mean(vs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9), used, oov

def cos(a, b):
    return float(a @ b) if a is not None and b is not None else np.nan

# ---------- stand-in DOMAIN_ANCHORS (generic, NOT taken from eval answers) ----------
ANCHORS = {
    "Business":   ["business company", "management finance", "market economy", "corporate profit", "customer sales"],
    "Technology": ["computer software", "technology engineering", "internet network", "data programming", "digital system"],
    "Science":    ["science research", "chemistry physics", "biology laboratory", "experiment theory", "molecule energy"],
    "Arts":       ["art painting", "artist gallery", "music film", "culture creative", "museum design"],
}
CENT = {}
for d, ph in ANCHORS.items():
    vs = [phrase_vec(p, False)[0] for p in ph]
    c = np.mean(vs, axis=0); CENT[d] = c / np.linalg.norm(c)

GATE_T = 0.30
CUTS = (0.05, 0.15, 0.30, 0.50)

def tier(sim):
    if np.isnan(sim) or sim < CUTS[0]: return 0
    if sim < CUTS[1]: return 1
    if sim < CUTS[2]: return 2
    if sim < CUTS[3]: return 3
    return 4

# ---------- run ----------
df = pd.read_csv("eval_cases.csv")
rows = []
for _, r in df.iterrows():
    ans = "" if pd.isna(r.answer) else str(r.answer)
    av, used, oov = phrase_vec(ans)
    tv, _, _ = phrase_vec(r.target, False)
    dv = CENT[r.domain]
    rows.append(dict(case_id=r.case_id, sim_target=cos(av, tv), sim_domain=cos(av, dv),
                     oov=";".join(oov), used_tokens=" ".join(used), has_vec=av is not None))
res = pd.concat([df, pd.DataFrame(rows).drop(columns="case_id")], axis=1)
res["gate_pred"] = np.where(res.sim_domain >= GATE_T, "in_domain", "out_of_domain")
res["dmg_tier"] = res.sim_target.apply(tier)
res.to_csv(f"{OUT}/eval_results_glove100.csv", index=False)

# ---------- metrics ----------
G = res[res.expected_domain.isin(["in_domain", "out_of_domain"]) & res.has_vec].copy()
def prf(g, pos):
    tp = ((g.expected_domain == pos) & (g.gate_pred == pos)).sum()
    fp = ((g.expected_domain != pos) & (g.gate_pred == pos)).sum()
    fn = ((g.expected_domain == pos) & (g.gate_pred != pos)).sum()
    p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    return p, rc, 2 * p * rc / (p + rc) if p + rc else 0
acc = (G.expected_domain == G.gate_pred).mean()
P, R, F = prf(G, "in_domain")
print(f"GATE n={len(G)} acc={acc:.4f} P={P:.3f} R={R:.3f} F1={F:.3f}")
cm = pd.crosstab(G.expected_domain, G.gate_pred).reindex(index=["in_domain", "out_of_domain"], columns=["in_domain", "out_of_domain"], fill_value=0)
print(cm)

# threshold sweep + AUC
from sklearn.metrics import roc_auc_score
y = (G.expected_domain == "in_domain").astype(int)
auc = roc_auc_score(y, G.sim_domain)
ths = np.round(np.arange(0.0, 0.81, 0.02), 2)
sweep = []
for t in ths:
    pred = np.where(G.sim_domain >= t, "in_domain", "out_of_domain")
    g2 = G.assign(gate_pred=pred)
    p, rc, f = prf(g2, "in_domain")
    sweep.append(dict(t=t, acc=(g2.expected_domain == pred).mean(), p=p, r=rc, f1=f))
sweep = pd.DataFrame(sweep); best = sweep.loc[sweep.f1.idxmax()]
print("AUC", round(auc, 4), "best F1 threshold", best.t, round(best.f1, 3), "acc", round(best.acc, 3))

# graded relevance
GR = res[(res.category == "graded_relevance") | (res.category.isin(["cross_domain_negative", "everyday_noise"]))].dropna(subset=["expected_relevance", "sim_target"])
rho, pval = spearmanr(GR.expected_relevance, GR.sim_target)
print(f"Spearman(expected_relevance, sim_target)={rho:.3f} p={pval:.2e} n={len(GR)}")
GRpos = GR[GR.expected_relevance > 0]
rho2, p2 = spearmanr(GRpos.expected_relevance, GRpos.sim_target)
print(f"Spearman within in-domain grades 1-3: {rho2:.3f} p={p2:.2e} n={len(GRpos)}")

# per-target ordering accuracy (pairwise)
def pair_acc(g):
    ok = tot = 0
    a = g[["expected_relevance", "sim_target"]].to_numpy()
    for i in range(len(a)):
        for j in range(len(a)):
            if a[i, 0] > a[j, 0]:
                tot += 1; ok += a[i, 1] > a[j, 1]
    return ok, tot
ok = tot = 0
for _, g in GR.groupby("target"):
    o, t_ = pair_acc(g); ok += o; tot += t_
print(f"Pairwise ordering acc = {ok}/{tot} = {ok/tot:.3f}")

# polysemy: did the phrase disambiguate?
POLY = res[res.category == "context_and_polysemy"].copy()
POLY_dec = POLY[POLY.expected_domain.isin(["in_domain", "out_of_domain"])]
pacc = (POLY_dec.expected_domain == POLY_dec.gate_pred).mean()
print(f"Polysemy decisive cases n={len(POLY_dec)} gate acc={pacc:.3f}")
# NOTE: gate is per-domain so ambiguous words: report the max-domain assignment
dom_names = list(CENT)
def best_domain(ans):
    v, _, _ = phrase_vec(ans, False)
    if v is None: return None
    s = {d: float(v @ CENT[d]) for d in dom_names}
    return max(s, key=s.get)
# input rules
def validity(r):
    a = "" if pd.isna(r.answer) else str(r.answer)
    if not a.strip(): return False
    if not re.search(r"[A-Za-z]", a): return False
    toks = re.findall(r"[A-Za-z]+", a.lower())
    if len(toks) > 3: return False
    tgt = r.target.lower()
    if tgt in toks or a.lower().strip() == tgt: return False
    if tgt + "s" in toks or tgt + "es" in toks: return False
    try: used = json.loads(r.used_guesses) if isinstance(r.used_guesses, str) else []
    except Exception: used = []
    norm = " ".join(sorted(toks)); sing = " ".join(sorted(t.rstrip("s") for t in toks))
    for u in used:
        ut = sorted(re.findall(r"[a-z]+", u.lower()))
        if " ".join(ut) == norm or " ".join(t.rstrip("s") for t in ut) == sing: return False
    return True
IR = res.copy(); IR["valid_pred"] = IR.apply(validity, axis=1)
val_acc = (IR.valid_pred == IR.expected_valid).mean()
print(f"Input validation rules acc = {val_acc:.3f} ({(IR.valid_pred==IR.expected_valid).sum()}/{len(IR)})")
IR_bad = IR[IR.valid_pred != IR.expected_valid][["case_id", "answer", "target", "expected_valid", "valid_pred"]]
print(IR_bad.to_string(index=False))

# errors
errs = G[G.expected_domain != G.gate_pred][["case_id", "domain", "target", "answer", "expected_domain", "sim_domain", "label_confidence"]]
errs.to_csv(f"{OUT}/gate_errors_glove100.csv", index=False)
print("gate errors:", len(errs)); print(errs.sort_values("sim_domain").to_string(index=False))

# per-domain / category
per_dom = G.groupby("domain").apply(lambda g: pd.Series(dict(n=len(g), acc=(g.expected_domain == g.gate_pred).mean())), include_groups=False)
per_cat = G.groupby("category").apply(lambda g: pd.Series(dict(n=len(g), acc=(g.expected_domain == g.gate_pred).mean())), include_groups=False)
print(per_dom); print(per_cat)
conf = G.groupby("label_confidence").apply(lambda g: pd.Series(dict(n=len(g), acc=(g.expected_domain == g.gate_pred).mean())), include_groups=False)
print(conf)

# OOV
oov_rows = res[res.oov.fillna("") != ""][["case_id", "answer", "oov"]]
print("rows with OOV tokens:", len(oov_rows)); print(oov_rows.to_string(index=False))
print("no vector at all:", int((~res.has_vec).sum()))

# ---------- FIGURES ----------
C = dict(blue="#2b5c8f", red="#e05a47", green="#2a9d8f", gold="#e9c46a", grey="#8d99ae")

# Fig 1: gate score distributions + confusion matrix + sweep
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
sns.histplot(data=G, x="sim_domain", hue="expected_domain", bins=30, palette={"in_domain": C["blue"], "out_of_domain": C["red"]}, alpha=.65, ax=ax[0])
ax[0].axvline(GATE_T, color="k", ls="--", lw=1.8, label=f"Gate threshold {GATE_T}")
ax[0].set_title(f"Gate score = cos(answer, domain centroid)\nAUC = {auc:.3f}", fontweight="bold"); ax[0].set_xlabel("cosine similarity")
ax[0].legend()
cm_arr = cm.to_numpy()
sns.heatmap(cm_arr, annot=[[f"TP={cm_arr[0,0]}", f"FN={cm_arr[0,1]}"], [f"FP={cm_arr[1,0]}", f"TN={cm_arr[1,1]}"]], fmt="", cmap="Blues", cbar=False,
            xticklabels=["Pred in", "Pred out"], yticklabels=["Actual in", "Actual out"], ax=ax[1], linewidths=2, linecolor="white", annot_kws=dict(fontsize=14, fontweight="bold"))
ax[1].set_title(f"Confusion matrix @ T={GATE_T} (n={len(G)})\nAcc={acc:.1%}  P={P:.2f}  R={R:.2f}  F1={F:.2f}", fontweight="bold")
ax[2].plot(sweep.t, sweep.acc, label="Accuracy", color=C["blue"], lw=2.2)
ax[2].plot(sweep.t, sweep.p, label="Precision(in)", color=C["green"], lw=1.8, ls="--")
ax[2].plot(sweep.t, sweep.r, label="Recall(in)", color=C["gold"], lw=1.8, ls="--")
ax[2].plot(sweep.t, sweep.f1, label="F1(in)", color=C["red"], lw=2.2)
ax[2].axvline(GATE_T, color="k", ls=":", label="0.30 (notebook)")
ax[2].axvline(best.t, color=C["red"], ls=":", label=f"best F1 @ {best.t:.2f}")
ax[2].set_title("Threshold sweep on 280-case set", fontweight="bold"); ax[2].set_xlabel("Gate threshold"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_gate_evaluation.png", dpi=140); plt.close()

# Fig 2: graded relevance
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
order = [0, 1, 2, 3]
bp = ax[0].boxplot([GR[GR.expected_relevance==g].sim_target.values for g in order], positions=range(4), widths=.55, patch_artist=True, showfliers=False)
for patch, col in zip(bp["boxes"], [C["red"], C["gold"], C["green"], C["blue"]]):
    patch.set_facecolor(col); patch.set_alpha(.6)
for med in bp["medians"]: med.set_color("k")
ax[0].scatter(GR.expected_relevance.astype(int)+np.random.RandomState(0).uniform(-.18,.18,len(GR)), GR.sim_target, color="k", alpha=.35, s=10)
ax[0].set_xticks(range(4))
for c in CUTS: ax[0].axhline(c, color=C["grey"], ls=":", lw=1)
ax[0].set_title(f"cos(answer, target) by human-labelled grade\nSpearman ρ = {rho:.2f} (n={len(GR)}, p={pval:.1e})", fontweight="bold")
ax[0].set_xlabel("expected relevance (0=out-of-domain … 3=close)"); ax[0].set_ylabel("cosine to target")
tab = pd.crosstab(GR.expected_relevance, GR.dmg_tier).reindex(index=order, columns=range(5), fill_value=0)
sns.heatmap(tab, annot=True, fmt="d", cmap="YlGnBu", ax=ax[1], cbar=False)
ax[1].set_title("Human grade vs damage tier from cutoffs\n(0.05 / 0.15 / 0.30 / 0.50)", fontweight="bold"); ax[1].set_xlabel("damage tier (0 = no damage … 4 = max)"); ax[1].set_ylabel("expected relevance")
pt = GR.groupby("target").apply(lambda g: spearmanr(g.expected_relevance, g.sim_target)[0] if g.expected_relevance.nunique() > 1 else np.nan, include_groups=False).dropna().sort_values()
ax[2].barh(pt.index, pt.values, color=[C["red"] if v < .5 else C["blue"] for v in pt.values])
ax[2].set_title("Per-target Spearman ρ (ranking quality)", fontweight="bold"); ax[2].set_xlabel("ρ"); ax[2].axvline(0, color="k", lw=1)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_graded_relevance.png", dpi=140); plt.close()

# Fig 3: polysemy heatmap (answer x domain centroid)
PA = ["logic", "business logic", "logic gate", "director", "film director", "company director", "virus", "computer virus", "biological virus", "model", "machine learning", "art auction"]
M = np.array([[ (lambda v: float(v @ CENT[d]) if v is not None else np.nan)(phrase_vec(a, False)[0]) for d in dom_names] for a in PA])
fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.1, 1]))
sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlBu_r", xticklabels=dom_names, yticklabels=PA, ax=ax[0], center=GATE_T, linewidths=1, linecolor="white", cbar_kws=dict(label="cos to domain centroid"))
ax[0].set_title("Polysemy probe: where does each phrase land?\n(centre of colour scale = gate threshold 0.30)", fontweight="bold")
cats = per_cat.sort_values("acc")
ax[1].barh(cats.index, cats.acc * 100, color=C["blue"])
for i, (a, n) in enumerate(zip(cats.acc, cats.n)): ax[1].text(a * 100 + 1, i, f"{a:.0%}  (n={int(n)})", va="center", fontsize=9)
ax[1].set_xlim(0, 120); ax[1].set_xlabel("Gate accuracy (%)"); ax[1].set_title("Gate accuracy by test category", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_polysemy_and_categories.png", dpi=140); plt.close()

# summary json
summary = dict(gate_n=len(G), gate_acc=acc, P=P, R=R, F1=F, auc=auc, best_threshold=float(best.t), best_f1=float(best.f1), best_acc=float(best.acc),
               spearman_all=rho, spearman_pos=rho2, pair_acc=ok / tot, polysemy_gate_acc=pacc, input_rules_acc=val_acc)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
