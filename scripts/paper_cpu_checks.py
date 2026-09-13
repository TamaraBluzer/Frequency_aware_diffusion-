from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from bootstrap_samples import overlap_matrix, upper_triangle_matrix
from downstream_classifier import generate_labeled_sbm_dataset

RESULTS = REPO / "results"
SNAPSHOT = RESULTS / "paper_sample_snapshot.json"
REPORT = RESULTS / "paper_cpu_checks.json"
BANDS = ("none", "low", "high", "random")
METRICS = ("degree", "clustering", "spectral", "wavelet", "orbit")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite observations are required for sample SD")
    return {"n": len(values), "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)), "values": values.tolist()}


def pack_graphs(graphs: list[nx.Graph]) -> dict:
    edges = upper_triangle_matrix(graphs)
    n = graphs[0].number_of_nodes()
    return {"n_nodes": n, "encoding": "upper triangle, row-major, numpy.packbits little-endian, hex",
            "edges": [np.packbits(row, bitorder="little").tobytes().hex() for row in edges]}


def unpack_graphs(payload: dict) -> list[nx.Graph]:
    n = payload["n_nodes"]
    pairs = n * (n - 1) // 2
    iu = np.triu_indices(n, 1)
    graphs = []
    for encoded in payload["edges"]:
        data = np.frombuffer(bytes.fromhex(encoded), dtype=np.uint8)
        if len(data) != (pairs + 7) // 8:
            raise ValueError("Invalid packed graph length")
        adjacency = np.zeros((n, n), dtype=np.uint8)
        adjacency[iu] = np.unpackbits(data, bitorder="little")[:pairs]
        graphs.append(nx.from_numpy_array(adjacency + adjacency.T))
    return graphs


def capture_samples() -> dict:
    import torch
    from fald.data.spectre import load_splits, raw_file

    reference = load_splits("planar")["val"]
    snapshot = {"dataset": "planar", "split": "val", "k": 8,
                "reference_source_sha256": digest(raw_file("planar")),
                "reference": pack_graphs(reference), "arms": {}}
    for band in BANDS:
        for seed in range(3):
            suffix = f"_seed{seed}" if seed else ""
            name = f"adjacency_diffusion_planar_{band}_k8{suffix}_samples.pt"
            path = RESULTS / name
            arrays = torch.load(path, map_location="cpu", weights_only=False)
            graphs = [nx.from_numpy_array((np.asarray(a) > 0).astype(np.uint8)) for a in arrays]
            if len(graphs) != len(reference):
                raise ValueError(f"{name}: sample/reference count mismatch")
            snapshot["arms"][f"{band}_seed{seed}"] = {
                "band": band, "seed": seed, "source_file": name,
                "source_sha256": digest(path), "graphs": pack_graphs(graphs)}
    return snapshot


def copying_audit(snapshot: dict) -> dict:
    reference = unpack_graphs(snapshot["reference"])
    ref_edges = upper_triangle_matrix(reference)
    arms = {}
    for name, source in snapshot["arms"].items():
        graphs = unpack_graphs(source["graphs"])
        scores = overlap_matrix(upper_triangle_matrix(graphs), ref_edges)["f1"]
        n = len(graphs)
        if scores.shape != (n, n) or n < 2:
            raise ValueError("Copying audit requires equally sized paired sample/reference sets")
        matched = np.diag(scores)
        mismatched = (scores.sum(axis=1) - matched) / (n - 1)
        tied = np.isclose(scores, scores.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
        unique_top1 = tied[np.arange(n), np.arange(n)] & (tied.sum(axis=1) == 1)
        connected = np.array([nx.is_connected(g) for g in graphs])
        planar = np.array([nx.check_planarity(g)[0] for g in graphs])
        arms[name] = {
            "band": source["band"], "seed": source["seed"], "n_samples": n,
            "matched_f1": float(matched.mean()), "mismatched_f1": float(mismatched.mean()),
            "gap": float((matched - mismatched).mean()),
            "unique_top1_count": int(unique_top1.sum()),
            "tied_top1_rows": int((tied.sum(axis=1) > 1).sum()),
            "connected_fraction": float(connected.mean()), "planar_fraction": float(planar.mean()),
            "valid_fraction": float((connected & planar).mean()),
            "mean_components": float(np.mean([nx.number_connected_components(g) for g in graphs])),
            "above_planar_edge_bound": float(np.mean([g.number_of_edges() > 3 * g.number_of_nodes() - 6 for g in graphs])),
            "matched_f1_per_graph": matched.tolist(), "mismatched_f1_per_graph": mismatched.tolist()}
    aggregate = {}
    for band in BANDS:
        rows = [arms[f"{band}_seed{s}"] for s in range(3)]
        aggregate[band] = {key: stats([r[key] for r in rows]) for key in
                           ("matched_f1", "mismatched_f1", "gap", "connected_fraction", "planar_fraction")}
        aggregate[band]["unique_top1_count"] = sum(r["unique_top1_count"] for r in rows)
        aggregate[band]["n_samples"] = sum(r["n_samples"] for r in rows)
    return {"arms": arms, "aggregates": aggregate, "chance_top1": 1 / len(reference)}


def numeric_summary() -> dict:
    pilot = {}
    for band in BANDS:
        reports = [load_result(f"adjacency_diffusion_planar_{band}_k8{f'_seed{s}' if s else ''}.json")
                   for s in range(3)]
        pilot[band] = {"ratio": stats([r["evaluation"]["ratio"] for r in reports]),
                       "validation_loss": stats([r["best"]["validation_loss"] for r in reports]),
                       "components": {m: stats([r["evaluation"][f"ratio/{m}"] for r in reports]) for m in METRICS}}
    seed_rows = load_result("k_sweep_planar_seedcheck.json")["rows"]
    seedcheck = {}
    for band, k in (("none", 2), ("low", 8), ("high", 32)):
        rows = [r for r in seed_rows if r["band"] == band and r["k"] == k]
        seedcheck[f"{band}_k{k}"] = {"ratio": stats([r["ratio"] for r in rows]),
                                     "validity": stats([r["validity"] for r in rows])}
    downstream = load_result("stage9_downstream.json")
    downstream_stats = {size: {arm: stats(row["accuracies"]) for arm, row in values.items()}
                        for size, values in downstream["results"].items()}
    return {"pilot": pilot, "seedcheck": seedcheck, "downstream": downstream_stats}


def node_count_baseline() -> dict:
    def features(graphs):
        return np.array([g.number_of_nodes() for g in graphs]), np.array([g.graph["num_communities"] for g in graphs])

    test_x, test_y = features(generate_labeled_sbm_dataset(25, seed=9999))
    val_x, val_y = features(generate_labeled_sbm_dataset(15, seed=8888))
    results = {}
    for size in (20, 40, 80):
        test_acc, val_acc = [], []
        for seed in range(5):
            x, y = features(generate_labeled_sbm_dataset(size // 4, seed=seed * 1000))
            classes = np.array([2, 3, 4, 5])
            centers = np.array([x[y == label].mean() for label in classes])
            predict = lambda values: classes[np.abs(values[:, None] - centers).argmin(axis=1)]
            test_acc.append(float((predict(test_x) == test_y).mean()))
            val_acc.append(float((predict(val_x) == val_y).mean()))
        results[str(size)] = {"test_accuracy": stats(test_acc), "validation_accuracy": stats(val_acc)}
    return {"method": "Nearest class mean in node count; means fitted to real training graphs only; no tuning",
            "source": "scripts/downstream_classifier.py:generate_labeled_sbm_dataset",
            "train_seeds": [0, 1000, 2000, 3000, 4000], "test_seed": 9999, "validation_seed": 8888,
            "test_n": len(test_x), "validation_n": len(val_x), "results": results,
            "test_node_label_sha256": hashlib.sha256(np.column_stack((test_x, test_y)).astype("<i8").tobytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-samples", action="store_true")
    args = parser.parse_args()
    if args.capture_samples:
        snapshot = capture_samples()
        SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    report = {"standard_deviation": "sample, ddof=1; across seeds unless stated otherwise",
              "sample_snapshot_sha256": digest(SNAPSHOT),
              "scope": "Stored pilot samples and original run summaries are distinct evidence sources; no new diffusion training or resampling",
              "numeric": numeric_summary(), "copying": copying_audit(snapshot),
              "node_count_baseline": node_count_baseline()}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, row in report["numeric"]["seedcheck"].items():
        print(f"{name}: Ratio {row['ratio']['mean']:.3f} +/- {row['ratio']['sample_sd']:.3f}; valid {100 * row['validity']['mean']:.2f}%")
    for band, row in report["copying"]["aggregates"].items():
        print(f"{band}: matched F1 {row['matched_f1']['mean']:.4f}; mismatched {row['mismatched_f1']['mean']:.4f}; unique top-1 {row['unique_top1_count']}/{row['n_samples']}")
    for size, row in report["node_count_baseline"]["results"].items():
        print(f"Node count only, n={size}: {100 * row['test_accuracy']['mean']:.1f}% +/- {100 * row['test_accuracy']['sample_sd']:.1f}%")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
