from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from qdrant_client import QdrantClient

from common import ROOT, SOURCE_COLLECTION, TARGET_COLLECTION, experiment_config, open_local_client, load_queries
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator


PHASE2 = ROOT / "experiments" / "dim_384_vs_768" / "phase2"
RAW = PHASE2 / "results" / "raw"
AUDIT = PHASE2 / "audit"
ANNOTATION = PHASE2 / "annotation"
SEED = 20261105
BOOTSTRAPS = 10_000


def points(client: QdrantClient, collection: str) -> list[Any]:
    result: list[Any] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=True,
        )
        result.extend(batch)
        if offset is None:
            return result


def dense_vector(point: Any) -> np.ndarray:
    vector = point.vector["dense"] if isinstance(point.vector, dict) else point.vector
    return np.asarray(vector, dtype=np.float64)


def query_relevant(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return bool(terms) and sum(term.lower() in lower for term in terms) >= max(1, math.ceil(len(terms) * 0.5))


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p))


def paired_stats(deltas: list[float]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    sample = np.asarray(deltas, dtype=float)
    means = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        means[index] = float(np.mean(rng.choice(sample, size=len(sample), replace=True)))
    nonzero = sample[sample != 0]
    if len(nonzero):
        try:
            from scipy.stats import wilcoxon
            wilcoxon_result = wilcoxon(nonzero, alternative="two-sided", method="auto")
            test = {"name": "Wilcoxon signed-rank", "p_value": float(wilcoxon_result.pvalue)}
        except ImportError:
            positives = int(np.sum(nonzero > 0))
            negatives = int(np.sum(nonzero < 0))
            n = positives + negatives
            tail = sum(math.comb(n, k) for k in range(min(positives, negatives) + 1)) / (2 ** n)
            test = {"name": "Exact sign test", "p_value": float(min(1.0, 2 * tail))}
    else:
        test = {"name": "No non-zero differences", "p_value": 1.0}
    return {
        "mean_delta": float(np.mean(sample)),
        "ci95_low": percentile(means.tolist(), 2.5),
        "ci95_high": percentile(means.tolist(), 97.5),
        "improved": int(np.sum(sample > 0)),
        "worsened": int(np.sum(sample < 0)),
        "unchanged": int(np.sum(sample == 0)),
        "test": test,
    }


def load_phase1_raw(collection: str) -> dict[str, Any]:
    return json.loads((ROOT / "experiments" / "dim_384_vs_768" / "results" / "raw" / f"{collection}.json").read_text(encoding="utf-8"))


def exact_dense(client: QdrantClient, collection: str, model_name: str, queries: list[dict[str, Any]]) -> dict[str, Any]:
    stored = points(client, collection)
    ids = [str(point.payload.get("chunk_id", point.id)) for point in stored]
    matrix = np.vstack([dense_vector(point) for point in stored])
    texts = {str(point.payload.get("chunk_id", point.id)): point.payload.get("text", "") for point in stored}
    metadata = {
        str(point.payload.get("chunk_id", point.id)): {
            "ticker": point.payload.get("ticker", ""),
            "company": point.payload.get("company_name", point.payload.get("company", "")),
            "form": point.payload.get("document_type", ""),
            "period": point.payload.get("fiscal_period", ""),
            "section": point.payload.get("section", ""),
        }
        for point in stored
    }
    encoder = DenseEmbeddingGenerator(model_name=model_name)
    output: dict[str, Any] = {}
    for query in queries:
        q = np.asarray(encoder.embed_query(query["query"]), dtype=np.float64)
        scores = matrix @ q
        order = np.argsort(-scores, kind="stable")
        ranked = [ids[index] for index in order]
        relevant = [chunk_id for chunk_id in ranked if query_relevant(texts[chunk_id], query["relevant_terms"])]
        first_rank = ranked.index(relevant[0]) + 1 if relevant else None
        reciprocal = 1.0 / first_rank if first_rank else 0.0
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        z_values = {chunk_id: float((scores[ids.index(chunk_id)] - mean) / std) for chunk_id in relevant} if std else {}
        output[query["id"]] = {
            "query": query["query"],
            "collection": collection,
            "total_points": len(ids),
            "relevant_count": len(relevant),
            "first_relevant_rank": first_rank,
            "mrr_full_corpus": reciprocal,
            "median_relevant_rank": float(statistics.median([ranked.index(chunk_id) + 1 for chunk_id in relevant])) if relevant else None,
            "relevant_ids": relevant,
            "relevant_z": z_values,
            "top10": [
                {"chunk_id": chunk_id, "rank": rank + 1, "score": float(scores[ids.index(chunk_id)])}
                for rank, chunk_id in enumerate(ranked[:10])
            ],
        }
    return output


def overlap(left: list[str], right: list[str]) -> float:
    return len(set(left) & set(right)) / 10.0


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_annotation(
    queries: list[dict[str, Any]],
    exact: dict[str, dict[str, Any]],
    raw384: dict[str, Any],
    raw768: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    systems = [("384_dense", exact["384"]), ("768_dense", exact["768"])]
    for label, raw in (("384_bm25_dense", raw384), ("768_bm25_dense", raw768)):
        systems.append((label, {q["id"]: q["methods"]["bm25_dense"]["results"][:10] for q in raw["queries"]}))
    items: dict[str, dict[str, Any]] = {}
    key: dict[str, Any] = {}
    difference_items: set[str] = set()
    for query in queries:
        query_id = query["id"]
        system_ids: dict[str, set[str]] = {}
        for label, result_map in systems:
            results = result_map[query_id]["top10"] if label.endswith("dense") and label in ("384_dense", "768_dense") else result_map[query_id]
            system_ids[label] = {str(item["chunk_id"]) for item in results}
            for item in results:
                chunk_id = str(item["chunk_id"])
                item_id = hashlib.sha256(f"{SEED}|{query_id}|{chunk_id}".encode()).hexdigest()[:16]
                if item_id not in items:
                    payload = metadata[chunk_id]
                    text = metadata[chunk_id]["text"]
                    cut = len(text) > 800
                    items[item_id] = {
                        "item_id": item_id,
                        "query": query["query"],
                        "company": payload["company"],
                        "form": payload["form"],
                        "period": payload["period"],
                        "section": payload["section"],
                        "chunk_text": text[:800] + (" [TRUNCATED]" if cut else ""),
                        "label": "",
                        "wrong_entity_or_period": "",
                    }
                key.setdefault(item_id, []).append({
                    "system": label,
                    "chunk_id": chunk_id,
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                })
        if system_ids["384_dense"] != system_ids["768_dense"] or system_ids["384_bm25_dense"] != system_ids["768_bm25_dense"]:
            difference_items.update(
                item_id for item_id, entries in key.items()
                if any(entry["system"].startswith("384") and entry["chunk_id"] in system_ids["384_dense"] | system_ids["384_bm25_dense"] for entry in entries)
                and any(entry["system"].startswith("768") and entry["chunk_id"] in system_ids["768_dense"] | system_ids["768_bm25_dense"] for entry in entries)
            )
    rng = random.Random(SEED)
    ordered = list(items.values())
    rng.shuffle(ordered)
    short = [item for item in ordered if item["item_id"] in difference_items]
    fields = ["item_id", "query", "company", "form", "period", "section", "chunk_text", "label", "wrong_entity_or_period"]
    write_csv(ANNOTATION / "annotation_sheet.csv", ordered, fields)
    write_csv(ANNOTATION / "annotation_sheet_short.csv", short, fields)
    (ANNOTATION / "KEY_do_not_open.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    (ANNOTATION / "RUBRIC.md").write_text(
        "# Rubric chấm mù\n\n"
        "- `0`: Không liên quan.\n"
        "- `1`: Cùng chủ đề nhưng không trả lời câu hỏi.\n"
        "- `2`: Trả lời một phần.\n"
        "- `3`: Trả lời trực tiếp, đúng công ty, đúng kỳ, đúng chỉ tiêu.\n\n"
        "`wrong_entity_or_period` ghi `1` nếu nội dung đúng chủ đề nhưng sai công ty hoặc sai kỳ báo cáo; nếu không, để `0` hoặc trống theo quy ước người chấm.\n",
        encoding="utf-8",
    )
    return {"all_items": len(ordered), "short_items": len(short), "difference_queries": sum(1 for q in queries if q["id"] in exact["384"] and [x["chunk_id"] for x in exact["384"][q["id"]]["top10"]] != [x["chunk_id"] for x in exact["768"][q["id"]]["top10"]])}


def main() -> None:
    for directory in (RAW, AUDIT, ANNOTATION):
        directory.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    client = open_local_client()
    exact = {
        "384": exact_dense(client, SOURCE_COLLECTION, "BAAI/bge-small-en-v1.5", queries),
        "768": exact_dense(client, TARGET_COLLECTION, "BAAI/bge-base-en-v1.5", queries),
    }
    raw384 = load_phase1_raw("384")
    raw768 = load_phase1_raw("768")
    (RAW / "exact_dense.json").write_text(json.dumps(exact, indent=2), encoding="utf-8")

    rows = []
    for query in queries:
        qid = query["id"]
        for metric in ("first_relevant_rank", "mrr_full_corpus", "median_relevant_rank"):
            left = exact["384"][qid][metric]
            right = exact["768"][qid][metric]
            if metric.endswith("rank") and left is not None and right is not None:
                delta = float(left - right)
            else:
                delta = float((right or 0) - (left or 0))
            rows.append({"query_id": qid, "metric": metric, "value_384": left, "value_768": right, "delta_768_minus_384": delta})
    write_csv(RAW / "exact_rank_metrics.csv", rows, ["query_id", "metric", "value_384", "value_768", "delta_768_minus_384"])

    overlap_rows = []
    paired_rows = []
    for query in queries:
        qid = query["id"]
        d384 = [item["chunk_id"] for item in exact["384"][qid]["top10"]]
        d768 = [item["chunk_id"] for item in exact["768"][qid]["top10"]]
        h384 = [item["chunk_id"] for item in raw384["queries"][[q["id"] for q in raw384["queries"]].index(qid)]["methods"]["bm25_dense"]["results"][:10]]
        h768 = [item["chunk_id"] for item in raw768["queries"][[q["id"] for q in raw768["queries"]].index(qid)]["methods"]["bm25_dense"]["results"][:10]]
        overlap_rows.extend([
            {"query_id": qid, "method": "dense", "overlap_at_10": overlap(d384, d768)},
            {"query_id": qid, "method": "bm25_dense", "overlap_at_10": overlap(h384, h768)},
        ])
        for metric in ("mrr_full_corpus",):
            paired_rows.append({"query_id": qid, "metric": metric, "delta": exact["768"][qid][metric] - exact["384"][qid][metric]})
        for metric in ("first_relevant_rank", "median_relevant_rank"):
            left = exact["384"][qid][metric]
            right = exact["768"][qid][metric]
            if left is not None and right is not None:
                paired_rows.append({"query_id": qid, "metric": metric, "delta": float(left - right)})
        z384 = [value for value in exact["384"][qid]["relevant_z"].values()]
        z768 = [value for value in exact["768"][qid]["relevant_z"].values()]
        if z384 and z768:
            paired_rows.append({"query_id": qid, "metric": "mean_relevant_z", "delta": float(np.mean(z768) - np.mean(z384))})
    write_csv(RAW / "overlap_at10.csv", overlap_rows, ["query_id", "method", "overlap_at_10"])
    write_csv(RAW / "paired_deltas.csv", paired_rows, ["query_id", "metric", "delta"])

    stats = {}
    for metric in sorted({row["metric"] for row in paired_rows}):
        stats[metric] = paired_stats([float(row["delta"]) for row in paired_rows if row["metric"] == metric])
    (RAW / "paired_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    raw_points = points(client, SOURCE_COLLECTION)
    metadata = {
        str(point.payload.get("chunk_id", point.id)): {
            "text": point.payload.get("text", ""),
            "company": point.payload.get("company_name", point.payload.get("company", "")),
            "form": point.payload.get("document_type", ""),
            "period": point.payload.get("fiscal_period", ""),
            "section": point.payload.get("section", ""),
        }
        for point in raw_points
    }
    annotation = build_annotation(queries, exact, raw384, raw768, metadata)
    discrepancy = {
        "readme_expected_bm25_dense_ndcg_at_10": 0.8631,
        "phase1_recreated_bm25_dense_ndcg_at_10": 0.812707307419045,
        "evidence": [
            "benchmark.py computes relevant_ids only from the returned result list using relevant_terms; it does not have independent ground-truth chunk IDs.",
            "metrics.py computes recall denominator from that result-derived relevant_ids and nDCG ideal gains from the same result-derived relevance_map.",
            "Phase 1 integrity confirms 8670 points; README historical figures refer to a different dataset/state (README describes 8609 chunks).",
        ],
        "conclusion": "Có bằng chứng về khác evaluator/điều kiện dữ liệu; không thể quy toàn bộ chênh lệch cho một nguyên nhân duy nhất.",
    }
    (AUDIT / "readme_discrepancy.json").write_text(json.dumps(discrepancy, indent=2, ensure_ascii=False), encoding="utf-8")
    (AUDIT / "annotation_counts.json").write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    client.close()
    print(json.dumps({"exact_queries": len(queries), "annotation": annotation, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
