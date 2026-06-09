#!/usr/bin/env python3
"""Task-diversity entropy for a data config (or the whole SQL dataset).

Computes, over the per-task **hours** distribution of a training mix:

  * effective number of tasks   D_q = ( Σ p_i^q ) ^ (1/(1-q))   (Hill numbers)
        - q=1 :  D_1 = exp(H)            H = -Σ p_i ln p_i   (Shannon)
        - q=2 :  D_2 = 1 / Σ p_i^2       (inverse Simpson)
  * evenness                    E_q = D_q / N    (N = richness = # tasks)
        E_q ∈ (0, 1];  1 ⇔ hours perfectly uniform across tasks.
        q=2 is more sensitive to a few tasks dominating than q=1.

`p_i` is task i's share of total hours. Hours for a task in the config are
`(# config episodes of that task) × (seconds/episode)`, where seconds/episode
is task-specific (`frames/episodes ÷ fps` from task_distribution_sql_v2.json)
or a flat constant fallback. Because episode lengths are similar across tasks,
the hours distribution is close to the episode-count distribution — pass
`--sql` to use real task-specific lengths.

------------------------------------------------------------------------------
Usage
-----
Per data config (the common case) — needs a hash→task map:

    # one-time: pull the curation scores (hash→task) and, optionally, the
    # per-task length table from the robotics Modal env:
    modal volume get egoverse-training-outputs \\
        data_curation/per_task_v8_sharded_2026-05-20_01-33-48/scores_by_task.json scratch/sbt.json

    python egomimic/scripts/mecka_process/task_entropy.py \\
        --config egomimic/hydra_configs/data/mecka_zip_h200_500h_25task.yaml \\
        --scores scratch/sbt.json \\
        --sql logs/task_distribution_sql_v2.json

Whole-dataset distribution (no config, no scores — fully local):

    python egomimic/scripts/mecka_process/task_entropy.py \\
        --global --sql logs/task_distribution_sql_v2.json

`--split {train,valid,train_viz,all}` picks which `*_datasets` block of the
config to measure (default: train — the mix the optimizer actually sees).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

FPS = 30.0
# Fallback mean episode length (s) — matches build_task_subset_configs.py.
DEFAULT_SECONDS_PER_EPISODE = 96.35


# --------------------------------------------------------------------------- #
# Entropy / diversity math
# --------------------------------------------------------------------------- #
def hill_number(p: list[float], q: float) -> float:
    """Hill number of order q: the effective number of equally-common tasks."""
    p = [x for x in p if x > 0.0]
    if not p:
        return 0.0
    if abs(q - 1.0) < 1e-9:  # limit q→1 : exp(Shannon)
        return math.exp(-sum(x * math.log(x) for x in p))
    return sum(x**q for x in p) ** (1.0 / (1.0 - q))


def diversity_report(hours_by_task: dict[str, float]) -> dict:
    tasks = {t: h for t, h in hours_by_task.items() if h > 0.0}
    total = sum(tasks.values())
    p = sorted((h / total for h in tasks.values()), reverse=True)
    N = len(p)  # richness = D_0
    H = -sum(x * math.log(x) for x in p) if p else 0.0
    D1, D2 = hill_number(p, 1.0), hill_number(p, 2.0)
    return {
        "n_tasks": N,
        "total_hours": total,
        "shannon_H": H,
        "shannon_J": (H / math.log(N)) if N > 1 else 1.0,  # Pielou (q=1 evenness)
        "D1_effective_tasks": D1,
        "D2_effective_tasks": D2,
        "E1_evenness": (D1 / N) if N else 0.0,
        "E2_evenness": (D2 / N) if N else 0.0,
        "top_tasks": sorted(tasks.items(), key=lambda kv: -kv[1])[:10],
    }


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def seconds_per_episode_by_task(sql_path: Path | None) -> dict[str, float]:
    """task -> mean seconds/episode from the SQL distribution (frames/eps/fps)."""
    if sql_path is None:
        return {}
    sql = json.loads(sql_path.read_text())
    out: dict[str, float] = {}
    for row in sql.get("per_task", []):
        eps = row.get("episodes", 0)
        if eps:
            out[row["task"]] = row["frames"] / eps / FPS
    return out


def invert_scores(scores_path: Path) -> dict[str, str]:
    """scores_by_task.json (task -> {hash: score}) -> {hash: task}."""
    scores = json.loads(scores_path.read_text())
    hash_to_task: dict[str, str] = {}
    for task, scored in scores.items():
        for h in scored:
            hash_to_task[h] = task
    return hash_to_task


def config_episode_hashes(config_path: Path, split: str) -> list[str]:
    """Flatten every eps_to_use list under the chosen *_datasets block(s)."""
    cfg = yaml.safe_load(config_path.read_text())
    blocks = (
        ["train_datasets", "valid_datasets", "train_viz_datasets"]
        if split == "all"
        else [f"{split}_datasets"]
    )
    hashes: list[str] = []
    seen_files: set[str] = set()
    for block in blocks:
        for ds in (cfg.get(block) or {}).values():
            rel = (ds.get("resolver") or {}).get("eps_to_use")
            if not rel or rel in seen_files:
                continue
            seen_files.add(rel)
            p = Path(rel)
            if not p.is_absolute():
                # eps_to_use is repo-root-relative; config lives at
                # <root>/egomimic/hydra_configs/data/<cfg>.yaml → parents[3] = root.
                p = config_path.resolve().parents[3] / rel
            data = json.loads(p.read_text())
            hashes.extend(data if isinstance(data, list) else list(data))
    return hashes


def hours_from_config(
    config_path: Path,
    split: str,
    hash_to_task: dict[str, str],
    spe_by_task: dict[str, float],
) -> tuple[dict[str, float], int]:
    """Per-task hours for the config's episode set. Returns (hours, n_unmapped)."""
    hashes = config_episode_hashes(config_path, split)
    hours: dict[str, float] = {}
    unmapped = 0
    for h in set(hashes):  # dedupe in case train/valid share a list
        task = hash_to_task.get(h)
        if task is None:
            unmapped += 1
            continue
        spe = spe_by_task.get(task, DEFAULT_SECONDS_PER_EPISODE)
        hours[task] = hours.get(task, 0.0) + spe / 3600.0
    return hours, unmapped


def hours_global(sql_path: Path) -> dict[str, float]:
    """Whole-dataset per-task hours straight from the SQL frame counts."""
    sql = json.loads(sql_path.read_text())
    return {r["task"]: r["frames"] / FPS / 3600.0 for r in sql.get("per_task", [])}


# --------------------------------------------------------------------------- #
def _print_report(label: str, rep: dict, n_unmapped: int = 0) -> None:
    print(f"\n=== {label} ===")
    print(f"  tasks (richness D0)   : {rep['n_tasks']}")
    print(f"  total hours           : {rep['total_hours']:.1f}")
    print(f"  Shannon H (nats)      : {rep['shannon_H']:.4f}")
    print("  --- q=1 (Shannon / perplexity) ---")
    print(f"    effective # tasks D1: {rep['D1_effective_tasks']:.3f}")
    print(
        f"    evenness E1 = D1/N  : {rep['E1_evenness']:.4f}   (Pielou J = {rep['shannon_J']:.4f})"
    )
    print("  --- q=2 (Simpson / inverse-Simpson) ---")
    print(f"    effective # tasks D2: {rep['D2_effective_tasks']:.3f}")
    print(f"    evenness E2 = D2/N  : {rep['E2_evenness']:.4f}")
    if n_unmapped:
        print(f"  ! {n_unmapped} config episode(s) had no task in --scores (excluded)")
    print("  top tasks by hours:")
    for t, h in rep["top_tasks"]:
        print(f"    {h:8.2f} h  {t}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", help="path to a data/*.yaml config")
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "train_viz", "all"],
        help="which *_datasets block to measure (default: train)",
    )
    ap.add_argument(
        "--scores", help="scores_by_task.json (hash->task); required for --config"
    )
    ap.add_argument(
        "--sql", help="task_distribution_sql_v2.json (task-specific episode lengths)"
    )
    ap.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="ignore --config; score the whole-dataset distribution from --sql",
    )
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()

    spe_by_task = seconds_per_episode_by_task(Path(args.sql)) if args.sql else {}

    if args.use_global:
        if not args.sql:
            ap.error("--global requires --sql (task_distribution_sql_v2.json)")
        hours = hours_global(Path(args.sql))
        label, n_unmapped = f"GLOBAL  {Path(args.sql).name}", 0
    else:
        if not args.config or not args.scores:
            ap.error("--config and --scores are both required (or use --global --sql)")
        hash_to_task = invert_scores(Path(args.scores))
        hours, n_unmapped = hours_from_config(
            Path(args.config), args.split, hash_to_task, spe_by_task
        )
        label = f"{Path(args.config).name}  [{args.split}]"
        if not spe_by_task:
            label += f"  (flat {DEFAULT_SECONDS_PER_EPISODE}s/ep)"

    rep = diversity_report(hours)
    if args.json:
        rep["top_tasks"] = [{"task": t, "hours": h} for t, h in rep["top_tasks"]]
        print(json.dumps({"label": label, "n_unmapped": n_unmapped, **rep}, indent=2))
    else:
        _print_report(label, rep, n_unmapped)


if __name__ == "__main__":
    main()
