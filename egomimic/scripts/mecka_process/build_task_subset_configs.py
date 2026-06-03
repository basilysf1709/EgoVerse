#!/usr/bin/env python3
"""Build task-balanced episode subsets for the staged (zip) training configs.

Produces the ``eps_to_use`` hash lists consumed by:

  - data/mecka_zip_h200_5h_1task.yaml    (1 task,  ~5h,  random sample)
  - data/mecka_zip_h200_50h_5task.yaml   (5 tasks, ~50h, balanced 10h/task)
  - data/mecka_zip_h200_500h_25task.yaml (25 tasks, all episodes ~528h)

Task -> episode mapping comes from a DemInf curation ``scores_by_task.json``
(task_name -> {episode_hash: score}); we keep only hashes that exist in the
zip-volume ``catalog.json``. Episodes are ~96 s each (mean of the 1000 catalog
entries that carry a real ``n_frames``; the rest are -1), so hours are
estimated as ``n_episodes * 96.35 / 3600``.

Selection per the agreed plan:
  * 1-task / 5-task: random sample (fixed seed) to the per-task episode target.
  * 25-task:         take ALL episodes of the top-25 tasks by episode count
                     (balanced 500h is infeasible — only 11 tasks have >=20h).

Outputs (written to egomimic/hydra_configs/data/extra/):
  mecka_5h_1task.json           list[str]  episode hashes (train/valid set)
  mecka_50h_5task.json          list[str]
  mecka_500h_25task.json        list[str]
  mecka_5h_1task_viz.json       list[str]  one episode per task (train_viz set)
  mecka_50h_5task_viz.json      list[str]
  mecka_500h_25task_viz.json    list[str]

The ``*_viz.json`` lists are built by grouping the chosen episodes by task and
taking one per task (a ``groupby('task').head(1)``-style lambda). A train_viz
dataset points its ``eps_to_use`` at one of these and sets
``max_frames_per_episode=1``, so each task contributes exactly one sample.

Usage:
  python egomimic/scripts/mecka_process/build_task_subset_configs.py \
      --scores scratch/sbt.json --catalog scratch/catalog.json

The scores/catalog files can be pulled from the robotics Modal env first:
  modal volume get egoverse-training-outputs \
      data_curation/per_task_v8_sharded_2026-05-20_01-33-48/scores_by_task.json scratch/sbt.json
  modal volume get mecka_data_zip catalog.json scratch/catalog.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRA_DIR = REPO_ROOT / "egomimic" / "hydra_configs" / "data" / "extra"

# Mean episode length (seconds) from the 1000 catalog entries with real n_frames.
SECONDS_PER_EPISODE = 96.35
SEED = 42


def hours_to_episodes(hours: float) -> int:
    return round(hours * 3600 / SECONDS_PER_EPISODE)


def task_episode_counts(
    scores_by_task: dict, catalog_hashes: set[str]
) -> list[tuple[str, list[str]]]:
    """[(task_name, [hashes present in catalog])] sorted by descending count."""
    rows: list[tuple[str, list[str]]] = []
    for task, scored in scores_by_task.items():
        hashes = [h for h in scored if h in catalog_hashes]
        if hashes:
            rows.append((task, hashes))
    rows.sort(key=lambda r: -len(r[1]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scores", default="scratch/sbt.json", help="scores_by_task.json path"
    )
    ap.add_argument(
        "--catalog", default="scratch/catalog.json", help="catalog.json path"
    )
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    scores_by_task = json.loads(Path(args.scores).read_text())
    catalog = json.loads(Path(args.catalog).read_text())
    catalog_hashes = {e["episode_hash"] for e in catalog}

    rows = task_episode_counts(scores_by_task, catalog_hashes)
    rng = random.Random(args.seed)

    def sample(hashes: list[str], n: int) -> list[str]:
        h = sorted(hashes)  # deterministic base order before seeded shuffle
        rng.shuffle(h)
        return sorted(h[:n])

    # Each config is built as an ordered {task: [selected hashes]} mapping so we
    # can derive both the full train/valid list (flatten) and the one-per-task
    # viz list (a groupby('task').head(1)-style lambda).
    per_task_select = lambda by_task: sorted(  # noqa: E731 — one per task
        hs[0] for hs in by_task.values() if hs
    )

    # --- 1 task / ~5h, random sample -------------------------------------
    n_5h = hours_to_episodes(5)
    task_1, hashes_1 = rows[0]
    cfg_1 = {task_1: sample(hashes_1, n_5h)}

    # --- 5 tasks / ~50h balanced (10h/task), random sample ---------------
    n_10h = hours_to_episodes(10)
    cfg_5 = {task: sample(hashes, n_10h) for task, hashes in rows[:5]}

    # --- 25 tasks / all episodes (~528h, imbalanced) ---------------------
    cfg_25 = {task: sorted(hashes) for task, hashes in rows[:25]}

    def flatten(by_task: dict) -> list[str]:
        return sorted({h for hs in by_task.values() for h in hs})

    train_1, train_5, train_25 = flatten(cfg_1), flatten(cfg_5), flatten(cfg_25)
    viz_1, viz_5, viz_25 = (
        per_task_select(cfg_1),
        per_task_select(cfg_5),
        per_task_select(cfg_25),
    )

    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mecka_5h_1task.json": train_1,
        "mecka_50h_5task.json": train_5,
        "mecka_500h_25task.json": train_25,
        "mecka_5h_1task_viz.json": viz_1,
        "mecka_50h_5task_viz.json": viz_5,
        "mecka_500h_25task_viz.json": viz_25,
    }
    for name, payload in outputs.items():
        (EXTRA_DIR / name).write_text(json.dumps(payload, indent=0))

    def hrs(n: int) -> float:
        return n * SECONDS_PER_EPISODE / 3600

    print("=== 1-task / 5h ===")
    print(
        f"  task: {task_1}   train={len(train_1)} eps (~{hrs(len(train_1)):.1f}h)  viz={len(viz_1)}"
    )
    print("=== 5-task / 50h balanced ===")
    print(f"  tasks: {list(cfg_5)}")
    print(
        f"  train={len(train_5)} eps (~{hrs(len(train_5)):.1f}h, ~{hrs(n_10h):.1f}h/task)  viz={len(viz_5)}"
    )
    print("=== 25-task / all ===")
    print(f"  tasks: {list(cfg_25)}")
    print(
        f"  train={len(train_25)} eps (~{hrs(len(train_25)):.1f}h)  viz={len(viz_25)}"
    )
    print(f"written to {EXTRA_DIR}")


if __name__ == "__main__":
    main()
