from __future__ import annotations

"""Stage-7 Aim 3: auditable patient-flow network construction (descriptive).

The primary graph is the observed treatment flow from residence municipality to
treating CNES.  A municipality-to-performing-municipality representation is a
prespecified sensitivity.  It is deliberately not a patient-level contact graph,
not a map of formal care pathways, and it supports no effect claim.
"""

import argparse
import hashlib
import heapq
import json
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MIN_DISPLAY_DEFAULT = 5
SEEDS_DEFAULT = (17, 29, 43, 71, 101)
RESOLUTIONS_DEFAULT = (0.8, 1.0, 1.2)
BETWEENNESS_BACKENDS_USED: set[str] = set()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def z6(value: object) -> str:
    return str(value).strip().zfill(6)


def clean_cnes(value: object) -> str:
    return str(value).strip()


def gini(values: Iterable[float]) -> float:
    x = np.sort(np.asarray(list(values), dtype=float))
    if len(x) == 0 or x.sum() <= 0:
        return float("nan")
    return float(((2 * np.arange(1, len(x) + 1) - len(x) - 1) * x).sum() / (len(x) * x.sum()))


def _weighted_brandes_python(edges: pd.DataFrame, source: str, target: str, weight: str) -> dict[str, float]:
    """Weighted undirected Brandes centrality, with distance=1/flow weight.

    The explicit inverse transform is central to interpretation: more observed
    flow means a stronger/shorter structural connection, never a longer one.
    """
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for row in edges[[source, target, weight]].itertuples(index=False):
        left, right, flow = str(row[0]), str(row[1]), float(row[2])
        if not math.isfinite(flow) or flow <= 0 or left == right:
            continue
        cost = 1.0 / flow
        adjacency.setdefault(left, []).append((right, cost))
        adjacency.setdefault(right, []).append((left, cost))
    centrality = {node: 0.0 for node in adjacency}
    for origin in adjacency:
        stack: list[str] = []
        predecessors = {node: [] for node in adjacency}
        sigma = dict.fromkeys(adjacency, 0.0); sigma[origin] = 1.0
        distance = dict.fromkeys(adjacency, math.inf); distance[origin] = 0.0
        queue: list[tuple[float, str]] = [(0.0, origin)]
        while queue:
            dist_v, vertex = heapq.heappop(queue)
            if dist_v > distance[vertex] + 1e-12:
                continue
            stack.append(vertex)
            for neighbour, cost in adjacency[vertex]:
                candidate = dist_v + cost
                if candidate < distance[neighbour] - 1e-12:
                    distance[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
                    sigma[neighbour] = sigma[vertex]
                    predecessors[neighbour] = [vertex]
                elif abs(candidate - distance[neighbour]) <= 1e-12:
                    sigma[neighbour] += sigma[vertex]
                    predecessors[neighbour].append(vertex)
        dependency = dict.fromkeys(adjacency, 0.0)
        while stack:
            vertex = stack.pop()
            for predecessor in predecessors[vertex]:
                dependency[predecessor] += (sigma[predecessor] / sigma[vertex]) * (1.0 + dependency[vertex])
            if vertex != origin:
                centrality[vertex] += dependency[vertex]
    return {node: value / 2.0 for node, value in centrality.items()}


def weighted_brandes_undirected(edges: pd.DataFrame, source: str, target: str, weight: str) -> dict[str, float]:
    """Exact weighted betweenness with distance=1/flow.

    Formal server runs use python-igraph's compiled exact Brandes
    implementation.  The dependency-free implementation is retained as a
    numerically auditable fallback for unit tests and restricted environments.
    """
    clean: list[tuple[str, str, float]] = []
    for row in edges[[source, target, weight]].itertuples(index=False):
        left, right, flow = str(row[0]), str(row[1]), float(row[2])
        if math.isfinite(flow) and flow > 0 and left != right:
            clean.append((left, right, 1.0 / flow))
    if not clean:
        return {}
    try:
        import igraph  # type: ignore
    except ImportError:
        BETWEENNESS_BACKENDS_USED.add("dependency-free exact weighted Brandes")
        return _weighted_brandes_python(edges, source, target, weight)
    nodes = sorted({left for left, _, _ in clean} | {right for _, right, _ in clean})
    index = {node: position for position, node in enumerate(nodes)}
    graph = igraph.Graph(
        n=len(nodes),
        edges=[(index[left], index[right]) for left, right, _ in clean],
        directed=False,
    )
    values = graph.betweenness(
        directed=False,
        weights=[cost for _, _, cost in clean],
    )
    BETWEENNESS_BACKENDS_USED.add("python-igraph exact weighted Brandes")
    return {node: float(values[index[node]]) for node in nodes}


def adjusted_rand_index(first: list[int], second: list[int]) -> float:
    """Dependency-free ARI used solely to audit repeated Leiden partitions."""
    if len(first) != len(second) or len(first) < 2:
        return float("nan")
    labels_a = {x: i for i, x in enumerate(sorted(set(first)))}
    labels_b = {x: i for i, x in enumerate(sorted(set(second)))}
    table: dict[tuple[int, int], int] = {}
    counts_a: dict[int, int] = {}; counts_b: dict[int, int] = {}
    for a, b in zip(first, second):
        ai, bi = labels_a[a], labels_b[b]
        table[(ai, bi)] = table.get((ai, bi), 0) + 1
        counts_a[ai] = counts_a.get(ai, 0) + 1; counts_b[bi] = counts_b.get(bi, 0) + 1
    choose2 = lambda n: n * (n - 1) / 2.0
    index = sum(choose2(n) for n in table.values())
    expected = sum(choose2(n) for n in counts_a.values()) * sum(choose2(n) for n in counts_b.values()) / choose2(len(first))
    maximum = 0.5 * (sum(choose2(n) for n in counts_a.values()) + sum(choose2(n) for n in counts_b.values()))
    return 1.0 if maximum == expected else float((index - expected) / (maximum - expected))


def variation_of_information(first: list[int], second: list[int]) -> float:
    if len(first) != len(second) or not first:
        return float("nan")
    n = float(len(first)); joint: dict[tuple[int, int], int] = {}; a: dict[int, int] = {}; b: dict[int, int] = {}
    for left, right in zip(first, second):
        joint[(left, right)] = joint.get((left, right), 0) + 1; a[left] = a.get(left, 0) + 1; b[right] = b.get(right, 0) + 1
    entropy_a = -sum((count / n) * math.log(count / n) for count in a.values())
    entropy_b = -sum((count / n) * math.log(count / n) for count in b.values())
    mutual = sum((count / n) * math.log((count * n) / (a[left] * b[right])) for (left, right), count in joint.items())
    return float(entropy_a + entropy_b - 2.0 * mutual)


def make_residence_projection(muni_edges: pd.DataFrame) -> pd.DataFrame:
    """One-mode residence projection; shared destination contribution is normalized.

    This is the only graph used for Leiden.  It is explicitly a same-type
    municipality projection, never a fictitious community analysis of a bipartite
    graph.  Pair contribution is (flow_i * flow_j) / provider_total.
    """
    rows: list[dict[str, object]] = []
    for destination, group in muni_edges.groupby("treat_municipio", sort=False):
        group = group.groupby("res_municipio", as_index=False)["n_aih"].sum()
        total = float(group["n_aih"].sum())
        if len(group) < 2 or total <= 0:
            continue
        for left, right in combinations(group.itertuples(index=False), 2):
            rows.append({"residence_a": left.res_municipio, "residence_b": right.res_municipio,
                         "projection_weight": float(left.n_aih * right.n_aih / total),
                         "via_performing_municipio": destination})
    if not rows:
        return pd.DataFrame(columns=["residence_a", "residence_b", "projection_weight"])
    return pd.DataFrame(rows).groupby(["residence_a", "residence_b"], as_index=False)["projection_weight"].sum()


def leiden_partitions(projection: pd.DataFrame, seeds: tuple[int, ...], resolutions: tuple[float, ...], mode: str) -> tuple[pd.DataFrame, dict]:
    try:
        import igraph  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:
        if mode == "required":
            raise RuntimeError("Leiden requires python-igraph and leidenalg; install them in the project environment") from exc
        return pd.DataFrame(columns=["node", "seed", "resolution", "community"]), {"community_status": "UNAVAILABLE", "reason": str(exc)}
    nodes = sorted(set(projection["residence_a"]) | set(projection["residence_b"]))
    if len(nodes) < 2 or projection.empty:
        return pd.DataFrame(columns=["node", "seed", "resolution", "community"]), {"community_status": "NOT_ESTIMABLE", "reason": "projection has fewer than two connected nodes"}
    index = {node: i for i, node in enumerate(nodes)}
    graph = igraph.Graph(n=len(nodes), edges=[(index[row.residence_a], index[row.residence_b]) for row in projection.itertuples(index=False)], directed=False)
    graph.es["weight"] = projection["projection_weight"].astype(float).tolist()
    records: list[dict[str, object]] = []
    for resolution in resolutions:
        for seed in seeds:
            partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=int(seed))
            records.extend({"node": node, "seed": seed, "resolution": resolution, "community": int(partition.membership[i])} for i, node in enumerate(nodes))
    frame = pd.DataFrame(records)
    comparisons = []
    for resolution in resolutions:
        subsets = [frame[(frame["resolution"] == resolution) & (frame["seed"] == seed)].set_index("node")["community"] for seed in seeds]
        for i, left in enumerate(subsets):
            for right in subsets[i + 1:]:
                common = left.index.intersection(right.index)
                comparisons.append({"resolution": resolution, "kind": "seed", "ari": adjusted_rand_index(left.loc[common].tolist(), right.loc[common].tolist()), "vi": variation_of_information(left.loc[common].tolist(), right.loc[common].tolist())})
    reference_seed = seeds[0]
    for i, resolution in enumerate(resolutions):
        for other in resolutions[i + 1:]:
            left = frame[(frame["resolution"] == resolution) & (frame["seed"] == reference_seed)].set_index("node")["community"]
            right = frame[(frame["resolution"] == other) & (frame["seed"] == reference_seed)].set_index("node")["community"]
            common = left.index.intersection(right.index)
            comparisons.append({"resolution": f"{resolution}_vs_{other}", "kind": "resolution", "ari": adjusted_rand_index(left.loc[common].tolist(), right.loc[common].tolist()), "vi": variation_of_information(left.loc[common].tolist(), right.loc[common].tolist())})
    return frame, {"community_status": "PASS", "n_projection_nodes": len(nodes), "n_projection_edges": len(projection), "stability": comparisons}


def build_flow_layers(cohorts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"cohort", "competence_month", "MUNIC_RES", "MUNIC_MOV", "SP_CNES"}
    missing = required - set(cohorts.columns)
    if missing:
        raise ValueError(f"cohorts missing required columns: {sorted(missing)}")
    frame = cohorts.loc[cohorts["cohort"].astype(str).isin(["A", "B"])].copy()
    frame["cohort"] = frame["cohort"].astype(str); frame["year"] = frame["competence_month"].astype(str).str[:4].astype(int)
    frame["res_municipio"] = frame["MUNIC_RES"].map(z6); frame["treat_municipio"] = frame["MUNIC_MOV"].map(z6); frame["SP_CNES"] = frame["SP_CNES"].map(clean_cnes)
    if frame[["res_municipio", "treat_municipio", "SP_CNES"]].eq("").any().any():
        raise ValueError("blank network identifier after normalization")
    # CNES is the primary destination node.  Municipality is intentionally not
    # part of this edge key: a location-record anomaly must not split one CNES
    # into artificial graph nodes.  The municipality layer below is the separate
    # prespecified sensitivity representation.
    cnes = frame.groupby(["cohort", "year", "res_municipio", "SP_CNES"], as_index=False).size().rename(columns={"size": "n_aih"})
    muni = frame.groupby(["cohort", "year", "res_municipio", "treat_municipio"], as_index=False).size().rename(columns={"size": "n_aih"})
    return cnes, muni


def display_suppress(frame: pd.DataFrame, count_columns: list[str], threshold: int) -> pd.DataFrame:
    result = frame.copy()
    for column in count_columns:
        if column in result:
            result[f"{column}_display"] = np.where(result[column] < threshold, f"<{threshold}", result[column].astype(int).astype(str))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-display", type=int, default=MIN_DISPLAY_DEFAULT)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS_DEFAULT)))
    parser.add_argument("--resolutions", default=",".join(map(str, RESOLUTIONS_DEFAULT)))
    parser.add_argument("--community-mode", choices=("required", "record-unavailable"), default="required")
    args = parser.parse_args()
    if args.min_display != MIN_DISPLAY_DEFAULT:
        raise ValueError("public suppression threshold is frozen at n<5")
    seeds = tuple(int(value) for value in args.seeds.split(",")); resolutions = tuple(float(value) for value in args.resolutions.split(","))
    if not seeds or not resolutions:
        raise ValueError("at least one seed and resolution are required")
    args.output_dir.mkdir(parents=True, exist_ok=True); args.audit.parent.mkdir(parents=True, exist_ok=True)
    cohorts = pq.read_table(args.cohorts).to_pandas()
    cnes_edges, muni_edges = build_flow_layers(cohorts)
    expected = cohorts[cohorts["cohort"].astype(str).isin(["A", "B"])].assign(year=lambda x: x["competence_month"].astype(str).str[:4].astype(int)).groupby(["cohort", "year"]).size().rename("expected").reset_index()
    layer_checks = []
    outputs = []
    origin_records = []
    node_records = []
    community_records = []
    community_audits = []
    for layer_name, edges, target in (("cnes", cnes_edges, "SP_CNES"), ("performing_municipio", muni_edges, "treat_municipio")):
        pooled = edges.groupby(["cohort", "res_municipio", target], as_index=False)["n_aih"].sum().assign(year="pooled")
        all_windows = pd.concat([edges, pooled], ignore_index=True)
        for (cohort, year), edge in all_windows.groupby(["cohort", "year"], sort=True):
            total = int(edge["n_aih"].sum())
            exp = expected[(expected["cohort"] == cohort) & (expected["year"].astype(str) == str(year))]["expected"]
            expected_total = total if str(year) == "pooled" else int(exp.iloc[0])
            layer_checks.append({"layer": layer_name, "cohort": cohort, "year": str(year), "observed": total, "expected": expected_total, "flow_conserved": total == expected_total, "unique_edge_key": not edge.duplicated(["res_municipio", target]).any()})
            detail = edge.copy(); detail["year"] = str(year); detail["layer"] = layer_name; detail["target_node"] = detail[target].astype(str); outputs.append(detail)
            by_origin = edge.groupby("res_municipio", as_index=False)["n_aih"].sum().rename(columns={"n_aih": "out_strength"})
            dest = edge.groupby(["res_municipio", target], as_index=False)["n_aih"].sum()
            origin_rows = []
            for residence, group in dest.groupby("res_municipio"):
                weights = group["n_aih"].to_numpy(float); shares = weights / weights.sum()
                origin_rows.append({"res_municipio": residence, "layer": layer_name, "cohort": cohort, "year": str(year), "out_strength": int(weights.sum()), "n_destinations": int(len(group)), "destination_hhi": float(np.square(shares).sum()), "destination_gini": gini(shares), "top_destination_share": float(shares.max())})
            origin_records.extend(origin_rows)
            bip = edge.rename(columns={"res_municipio": "source", target: "target"})[["source", "target", "n_aih"]].copy()
            bip["source"] = "R:" + bip["source"].astype(str); bip["target"] = "T:" + bip["target"].astype(str)
            centrality = weighted_brandes_undirected(bip, "source", "target", "n_aih")
            in_strength = edge.groupby(target, as_index=False)["n_aih"].sum().rename(columns={target: "node", "n_aih": "in_strength"})
            in_strength["node"] = "T:" + in_strength["node"].astype(str)
            in_strength["weighted_betweenness_inverse_flow_cost"] = in_strength["node"].map(centrality).fillna(0.0)
            in_strength["layer"] = layer_name; in_strength["cohort"] = cohort; in_strength["year"] = str(year)
            node_records.append(in_strength)
            if layer_name == "performing_municipio" and cohort == "B":
                projection = make_residence_projection(edge)
                partitions, audit = leiden_partitions(projection, seeds, resolutions, args.community_mode)
                if not partitions.empty:
                    partitions["cohort"] = cohort; partitions["year"] = str(year); community_records.append(partitions)
                audit.update({"cohort": cohort, "year": str(year), "projection_graph": "residence-municipality one-mode projection", "not_bipartite": True})
                community_audits.append(audit)
    edges_out = pd.concat(outputs, ignore_index=True); origins_out = pd.DataFrame(origin_records); nodes_out = pd.concat(node_records, ignore_index=True)
    edges_out.to_parquet(args.output_dir / "aim3_patient_flow_edges_v2.parquet", index=False)
    origins_out.to_parquet(args.output_dir / "aim3_origin_metrics_v2.parquet", index=False)
    nodes_out.to_parquet(args.output_dir / "aim3_target_metrics_v2.parquet", index=False)
    display_suppress(edges_out, ["n_aih"], args.min_display).to_csv(args.output_dir / "aim3_patient_flow_edges_display_v2.csv", index=False)
    if community_records:
        pd.concat(community_records, ignore_index=True).to_parquet(args.output_dir / "aim3_residence_projection_leiden_v2.parquet", index=False)
    else:
        pd.DataFrame(columns=["node", "seed", "resolution", "community", "cohort", "year"]).to_parquet(args.output_dir / "aim3_residence_projection_leiden_v2.parquet", index=False)
    checks = pd.DataFrame(layer_checks)
    checks.to_parquet(args.output_dir / "aim3_network_checks_v2.parquet", index=False)
    all_finite = bool(np.isfinite(origins_out.select_dtypes(include=[np.number]).to_numpy()).all() and np.isfinite(nodes_out.select_dtypes(include=[np.number]).to_numpy()).all())
    qc_checks = {"unique_edge_keys": bool(checks["unique_edge_key"].all()), "flow_conservation": bool(checks["flow_conserved"].all()), "edge_year_serialized_as_string": bool(edges_out["year"].map(type).eq(str).all()), "finite_metrics": all_finite, "privacy_threshold_n_lt_5": args.min_display == 5, "inverse_flow_cost_documented": True, "exact_compiled_betweenness_backend": BETWEENNESS_BACKENDS_USED == {"python-igraph exact weighted Brandes"}, "communities_on_same_type_projection_only": True, "community_stability_recorded": bool(community_audits)}
    community_required_fail = any(audit.get("community_status") == "UNAVAILABLE" for audit in community_audits) and args.community_mode == "required"
    qc_checks["community_required_available"] = not community_required_fail
    status = "PASS" if all(qc_checks.values()) else "DOWNGRADE"
    audit = {"schema_version": "2.0", "generated_at": utc_now(), "status": status, "evidence_level": "descriptive observed treatment flow; scenario-based only where applicable; no effect estimate", "node_definitions": {"primary": "residence municipality -> treating CNES", "sensitivity": "residence municipality -> performing municipality"}, "betweenness": "undirected bipartite structural topology with distance=1/flow; high observed flow is shorter, not longer", "betweenness_backends": sorted(BETWEENNESS_BACKENDS_USED), "community_method": "Leiden only on a residence-municipality same-type projection", "community_audits": community_audits, "checks": qc_checks, "input_hashes": {"cohorts_sha256": sha256_file(args.cohorts)}, "suppression": "Public displays suppress n<5; analysis outputs retain full aggregate counts; no patient-level release."}
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(f"Aim 3 network QC not PASS: {qc_checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
