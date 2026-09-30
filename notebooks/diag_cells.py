# ==== DIAG CELL 1: hubness. Which candidates show up in the top-10 of many *different* queries? ====
from collections import Counter
hub = Counter()
for qid, q, pat, pair in QUERIES:
    top, _, _ = clm_rank(q, "clm-latest", k=10)
    hub.update(cat[i]["text"] for i, _ in top)
print("Candidates in the top-10 of many different queries (hubs):")
for t, n in hub.most_common(8):
    print(f"  {n:2d}/20  {t[:90]}")
HUBS = [(t[:70], n) for t, n in hub.most_common(5)]

# ==== DIAG CELL 2: small candidate sets. Bug or hubness? And does retrieve-then-rerank work? ====
import random
rng = random.Random(0)
K = 50

def rank_subset(query, idxs, model):
    r = S.post("http://127.0.0.1:8700/v1/rank", json={"context": query, "answers": [CANDS[i] for i in idxs], "model": model}, timeout=600)
    r.raise_for_status()
    return [IDX[x["candidate"]] for x in r.json()["ranked"]], float(r.headers["X-CLM-Latency-Ms"])

rows2, lat2 = [], []
for qid, q, pat, pair in QUERIES:
    gold = [i for i, c in enumerate(cat) if is_gold(c["cmd"], pat)]
    gold_set = set(gold)
    bm_scores = bm25.get_scores(tok(q))
    sets = {
        # realistic pipeline: keyword search retrieves 50, CLM re-ranks them
        "bm25_top50": [int(i) for i in np.argsort(-bm_scores)[:K]],
        # isolates CLM's judgment: every correct answer + random distractors (no retrieval step)
        "oracle": gold + rng.sample([i for i in range(len(cat)) if i not in gold_set], K),
    }
    row = {"id": qid, "pair": pair or "", "query": q}
    for sname, idxs in sets.items():
        orders = {"bm25": sorted(idxs, key=lambda i: -bm_scores[i])}
        for m in ["clm-raw", "clm-latest"]:
            orders[m], ms = rank_subset(q, idxs, m)
            lat2.append(ms)
        for meth, order in orders.items():
            row[f"{sname}|{meth}@1"] = order[0] in gold_set
            row[f"{sname}|{meth}@5"] = any(i in gold_set for i in order[:5])
        row[f"{sname}|clm-latest_top1"] = cat[orders["clm-latest"][0]]["cmd"][:55]
    rows2.append(row)
diag = pd.DataFrame(rows2)

DIAG = {}
for sname in ["oracle", "bm25_top50"]:
    DIAG[sname] = {meth: f"{int(diag[f'{sname}|{meth}@1'].sum())}/20 @1, {int(diag[f'{sname}|{meth}@5'].sum())}/20 @5"
                   for meth in ["bm25", "clm-raw", "clm-latest"]}
DIAG["bm25_recall@50"] = f"{int(diag['bm25_top50|bm25@5'].sum())}/20 in top-5, " + \
                         f"{sum(any(is_gold(cat[i]['cmd'], pat) for i in [int(j) for j in np.argsort(-bm25.get_scores(tok(q)))[:K]]) for _, q, pat, _ in QUERIES)}/20 in top-50"
DIAG["rerank_server_p50_ms"] = round(float(np.median(lat2)), 1)
h1 = int(diag["bm25_top50|clm-latest@1"].sum())
DIAG["hybrid_gate"] = {"clm_rerank_top1_ge_11": h1 >= 11,
                       "beats_raw_rerank": h1 > int(diag["bm25_top50|clm-raw@1"].sum())}

display(pd.DataFrame({k: v for k, v in DIAG.items() if k in ("oracle", "bm25_top50")}))
print("\nPolarity pairs, BM25 top-50 -> CLM re-rank, top-1:")
for _, r in diag[diag.pair != ""].iterrows():
    print(f"  [{r['pair']}] {'OK ' if r['bm25_top50|clm-latest@1'] else 'MISS'} {r['query'][:55]:55} -> {r['bm25_top50|clm-latest_top1']}")

print("=" * 20, "PASTE THIS BACK", "=" * 20)
print(json.dumps({"hubs": HUBS, "diag": DIAG,
                  "rerank_misses": [{"q": r["query"], "top1": r["bm25_top50|clm-latest_top1"]}
                                    for _, r in diag.iterrows() if not r["bm25_top50|clm-latest@1"]]}, indent=1))
