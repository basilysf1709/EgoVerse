#!/usr/bin/env python3
"""Build task-balanced episode subsets for the staged (zip) training configs.

Produces the ``eps_to_use`` hash lists consumed by:

  - data/mecka_zip_h200_5h_1task.yaml    (1 task,  ~5h,  random sample)
  - data/mecka_zip_h200_50h_5task.yaml   (5 tasks, ~50h, balanced 10h/task)
  - data/mecka_zip_h200_500h_25task.yaml (25 tasks, all episodes ~528h)
  - data/mecka_zip_h200_20h_1task.yaml   (1 task,  ~20h, random sample)
  - data/mecka_zip_h200_20h_5task.yaml   (5 tasks, ~20h, balanced 4h/task)
  - data/mecka_zip_h200_20h_10task.yaml  (10 tasks,~20h, balanced 2h/task)
  - data/mecka_zip_h200_1task_1h.yaml    (1 task,  ~1h)
  - data/mecka_zip_h200_1task_5h.yaml    (1 task,  ~5h)
  - data/mecka_zip_h200_1task_10h.yaml   (1 task,  ~10h)
  - data/mecka_zip_h200_1task_20h.yaml   (1 task,  ~20h)
  - data/mecka_zip_h200_1task_50h.yaml   (1 task,  ~50h = all available)
  - data/mecka_zip_h200_ironing_30h.yaml (ironing_clothes, ~30h, LR-sweep base)

The three ``20h`` configs are a TASK-SCALING sweep: total data is held fixed at
~20h while task diversity grows (1 → 5 → 10). Task sets are nested top-N by
episode count (the 1-task set ⊂ the 5-task set ⊂ the 10-task set), so the only
varied factor is how many tasks the 20h is spread across. All 10 tasks have far
more than 2h available (the top-25 range from ~49h down to ~9.6h), so unlike
the 500h case every 20h split is exactly balanceable.

The five ``1task`` configs are the inverse — a DATA-SCALING (hours) sweep: a
SINGLE task (the top task by episode count, ``potting_plants`` ≈ 49h available)
is held fixed while the amount of data grows (1 → 5 → 10 → 20 → 50h). The
episode subsets are strictly NESTED (1h ⊂ 5h ⊂ 10h ⊂ 20h ⊂ 50h), built by
shuffling the task's episodes once and taking prefixes, so the only varied
factor is how many hours of that one task the model sees. The 50h point is
capped at all available episodes (~49h, the task does not have a full 50h).
This sweep uses a dedicated RNG so the existing task-sweep JSONs above are left
byte-for-byte unchanged when this script is re-run.

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

    # --- task-scaling sweep: ~20h total, vary task count -----------------
    # Nested top-N task sets; ~20h held fixed by shrinking the per-task budget
    # as the task count grows: 1×20h, 5×4h, 10×2h. (Defined AFTER cfg_1/cfg_5
    # so the shared rng state — and therefore the existing 5h/50h JSONs — are
    # byte-for-byte unchanged when this script is re-run.)
    n_20h, n_4h, n_2h = (
        hours_to_episodes(20),
        hours_to_episodes(4),
        hours_to_episodes(2),
    )
    cfg_20h_1 = {rows[0][0]: sample(rows[0][1], n_20h)}
    cfg_20h_5 = {task: sample(hashes, n_4h) for task, hashes in rows[:5]}
    cfg_20h_10 = {task: sample(hashes, n_2h) for task, hashes in rows[:10]}

    # Warn (don't silently truncate) if any task can't cover its budget.
    for label, cfg, want in (
        ("20h/1task", cfg_20h_1, n_20h),
        ("20h/5task", cfg_20h_5, n_4h),
        ("20h/10task", cfg_20h_10, n_2h),
    ):
        short = {t: len(h) for t, h in cfg.items() if len(h) < want}
        if short:
            print(f"WARNING {label}: tasks under {want} eps target: {short}")

    # --- data-scaling (hours) sweep: ONE task, nested 1/5/10/20/50h ------
    # Hold the top task fixed and grow the data. Shuffle the task's episodes
    # ONCE with a dedicated RNG, then take prefixes so the subsets nest
    # (1h ⊂ 5h ⊂ 10h ⊂ 20h ⊂ 50h). Independent rng_h keeps the task-sweep
    # JSONs above untouched regardless of where this block sits.
    HOURS_SWEEP = [1, 5, 10, 20, 50]
    rng_h = random.Random(args.seed)
    hsweep_task, hsweep_hashes = rows[0]
    hsweep_base = sorted(hsweep_hashes)  # deterministic order before shuffle
    rng_h.shuffle(hsweep_base)
    hsweep_nested = {
        h: sorted(hsweep_base[: min(hours_to_episodes(h), len(hsweep_base))])
        for h in HOURS_SWEEP
    }
    # One viz episode, drawn from the smallest (1h) set so it is present in
    # every nested superset — each hours config shows the SAME single sample.
    hsweep_viz = [min(hsweep_nested[HOURS_SWEEP[0]])]

    # --- ironing_clothes 30h LR-sweep dataset ----------------------------
    # A single ~30h subset of ironing_clothes, the data base for an LR sweep
    # (see the pi0.5_ironing30h_lr{1,3,5}e5 model configs). Independent rng_ic
    # leaves every JSON above byte-for-byte unchanged on re-run.
    rng_ic = random.Random(args.seed)
    ironing_hashes = dict(rows)["ironing_clothes"]
    ironing_base = sorted(ironing_hashes)  # deterministic order before shuffle
    rng_ic.shuffle(ironing_base)
    n_ironing_30h = min(hours_to_episodes(30), len(ironing_base))
    ironing_30h = sorted(ironing_base[:n_ironing_30h])
    ironing_30h_viz = [min(ironing_30h)]

    # --- ironing_clothes 10h subset (nested inside the 30h set) ----------
    # Built from the 30h hash list (NOT from the catalog directly) so we are
    # guaranteed that every selected episode is also present in the modal zip
    # volume that the 30h sweep already trained on. The re-shuffle uses a
    # fresh RNG seeded the same way so the 10h pick is reproducible without
    # touching the 30h ordering above. SQL has ~136.8h of ironing_clothes,
    # but only ~30h is staged in the zip volume — so we sample from the
    # confirmed-present 30h, not from SQL.
    rng_ic10 = random.Random(args.seed)
    ironing_10h_base = sorted(ironing_30h)
    rng_ic10.shuffle(ironing_10h_base)
    n_ironing_10h = min(hours_to_episodes(10), len(ironing_10h_base))
    ironing_10h = sorted(ironing_10h_base[:n_ironing_10h])
    ironing_10h_viz = [min(ironing_10h)]

    def flatten(by_task: dict) -> list[str]:
        return sorted({h for hs in by_task.values() for h in hs})

    train_1, train_5, train_25 = flatten(cfg_1), flatten(cfg_5), flatten(cfg_25)
    train_20h_1, train_20h_5, train_20h_10 = (
        flatten(cfg_20h_1),
        flatten(cfg_20h_5),
        flatten(cfg_20h_10),
    )
    viz_1, viz_5, viz_25 = (
        per_task_select(cfg_1),
        per_task_select(cfg_5),
        per_task_select(cfg_25),
    )
    viz_20h_1, viz_20h_5, viz_20h_10 = (
        per_task_select(cfg_20h_1),
        per_task_select(cfg_20h_5),
        per_task_select(cfg_20h_10),
    )

    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mecka_5h_1task.json": train_1,
        "mecka_50h_5task.json": train_5,
        "mecka_500h_25task.json": train_25,
        "mecka_5h_1task_viz.json": viz_1,
        "mecka_50h_5task_viz.json": viz_5,
        "mecka_500h_25task_viz.json": viz_25,
        "mecka_20h_1task.json": train_20h_1,
        "mecka_20h_5task.json": train_20h_5,
        "mecka_20h_10task.json": train_20h_10,
        "mecka_20h_1task_viz.json": viz_20h_1,
        "mecka_20h_5task_viz.json": viz_20h_5,
        "mecka_20h_10task_viz.json": viz_20h_10,
    }
    for h in HOURS_SWEEP:
        outputs[f"mecka_1task_{h}h.json"] = hsweep_nested[h]
        outputs[f"mecka_1task_{h}h_viz.json"] = hsweep_viz
    outputs["mecka_ironing_30h.json"] = ironing_30h
    outputs["mecka_ironing_30h_viz.json"] = ironing_30h_viz
    outputs["mecka_ironing_10h.json"] = ironing_10h
    outputs["mecka_ironing_10h_viz.json"] = ironing_10h_viz
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
    print("=== task-scaling sweep (~20h held fixed) ===")
    print(
        f"  20h/1task : {list(cfg_20h_1)}\n"
        f"    train={len(train_20h_1)} eps (~{hrs(len(train_20h_1)):.1f}h, ~{hrs(n_20h):.1f}h/task)  viz={len(viz_20h_1)}"
    )
    print(
        f"  20h/5task : {list(cfg_20h_5)}\n"
        f"    train={len(train_20h_5)} eps (~{hrs(len(train_20h_5)):.1f}h, ~{hrs(n_4h):.1f}h/task)  viz={len(viz_20h_5)}"
    )
    print(
        f"  20h/10task: {list(cfg_20h_10)}\n"
        f"    train={len(train_20h_10)} eps (~{hrs(len(train_20h_10)):.1f}h, ~{hrs(n_2h):.1f}h/task)  viz={len(viz_20h_10)}"
    )
    print(f"=== data-scaling (hours) sweep: 1 task = {hsweep_task} ===")
    for h in HOURS_SWEEP:
        n = len(hsweep_nested[h])
        print(f"  {h:>2}h: {n} eps (~{hrs(n):.1f}h)  viz={len(hsweep_viz)}")
    print("=== ironing_clothes 30h LR-sweep dataset ===")
    print(
        f"  {len(ironing_30h)} eps (~{hrs(len(ironing_30h)):.1f}h)  viz={len(ironing_30h_viz)}"
    )
    print("=== ironing_clothes 10h subset (nested in 30h) ===")
    print(
        f"  {len(ironing_10h)} eps (~{hrs(len(ironing_10h)):.1f}h)  viz={len(ironing_10h_viz)}"
    )
    print(f"written to {EXTRA_DIR}")


if __name__ == "__main__":
    main()
