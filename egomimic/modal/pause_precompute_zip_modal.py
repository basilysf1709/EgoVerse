"""Modal pause-filter precompute fan-out for the per-episode ZIP dataset.

The zip training data is a directory of per-episode tar archives on the
``mecka_data_zip`` volume mounted at ``/mnt/zarr-zip``, indexed by
``catalog.json`` (``{episode_hash, tar_filename, n_frames, embodiment}``).
This is the zip-pipeline analogue of ``pause_precompute_modal.py`` (which
targets the ~20-episode tar *shards* on ``mecka_data_wds_v2``).

Pipeline:
  1. Orchestrator (CPU): clone the repo, hydra-compose the training config,
     walk train/valid ``ZipEpisodeResolver`` blocks to union their
     ``eps_to_use`` allowlists (honoring the exact subset the trainer reads),
     read ``catalog.json``, and resolve (episode_hash, tar_filename) pairs.
  2. Partition episodes into batches and fan out
     ``_pause_precompute_zip_batch.starmap(...)`` across up to 500 CPU
     containers — every worker extracts its per-episode tars onto local NVMe,
     opens the zarr stores, and emits per-episode keep-indices for every
     epsilon in a single pass.
  3. Aggregate the returned dicts and write one cache per epsilon to
     ``/mnt/zarr-zip/pause_cache/<base>_eps<eps>/cache.json``.

Output JSON shape (matches ``_apply_pause_precompute_cache`` /
``ZarrDataset.precompute_pause_filter`` consumer)::

    {"<episode_hash>": {"raw_total": int, "keep_indices": [int, ...]}, ...}

The keep-mask logic is kept in sync with ``_build_pause_keep_mask`` in
``egomimic.rldb.zarr.zarr_dataset_multi``.

One-shot:
    MODAL_ENVIRONMENT=robotics python egomimic/modal/pause_precompute_zip_modal.py \
        name=pause_500h description=scaling \
        pause_config_name=train_zarr_cartesian_pi \
        pause_epsilon=0.005,0.0075,0.01 \
        data=mecka_zip_h200_500h_25task

``pause_epsilon=<float[,float,...]>`` and ``pause_config_name=<name>`` are
required. After completion the cache paths are printed; wire one into a run
via ``data.pause_precompute_cache=/mnt/zarr-zip/pause_cache/<base>_eps<eps>/cache.json``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _local_hf_token,
    _prepare_repo,
    _resolve_git_state,
    app,
    app_name_from_hydra_args,
    launch_detached,
    pop_init_submodules,
    training_outputs_volume,
    zip_volume,
)

ZIP_MOUNT_PATH = "/mnt/zarr-zip"

# Orchestrator is CPU-only (config walk + catalog filter + result aggregation).
PAUSE_ORCHESTRATOR = ModalCompute(gpu=None, cpu=4.0, memory_mb=16384)

# Worker streams a batch of small per-episode tars onto local NVMe, one at a
# time, and reads each zarr store locally. Light on RAM (one episode resident).
PAUSE_WORKER = ModalCompute(gpu=None, cpu=2.0, memory_mb=8192)

PAUSE_MAX_CONTAINERS = int(os.environ.get("EGOMIMIC_PAUSE_MAX_CONTAINERS", "500"))
# Episodes processed per container. ~50 small tars/worker keeps the container
# count well under the cap for the full ~20k-episode 500h set.
EPISODES_PER_WORKER = int(os.environ.get("EGOMIMIC_PAUSE_EPISODES_PER_WORKER", "50"))

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]


def _format_eps(eps: float) -> str:
    """Canonical epsilon → string form used in cache paths and dict keys."""
    s = f"{eps:.10f}".rstrip("0").rstrip(".")
    return s or "0"


# ---------------------------------------------------------------------------
# Batch worker
# ---------------------------------------------------------------------------
#
# Inherits the main app image (imports modal_setup at module load). Tar
# extraction + numpy delta math; no heavy deps. The keep-mask logic must stay
# in sync with ``_build_pause_keep_mask`` in zarr_dataset_multi.


@app.function(
    cpu=PAUSE_WORKER.cpu,
    memory=PAUSE_WORKER.memory_mb,
    timeout=1800,
    volumes={ZIP_MOUNT_PATH: zip_volume},
    max_containers=PAUSE_MAX_CONTAINERS,
)
def _pause_precompute_zip_batch(
    batch_id: int,
    epsilons: tuple[float, ...],
    episodes: tuple[tuple[str, str], ...],
) -> tuple[int, dict[str, dict[str, dict]]]:
    """Process a batch of per-episode tars.

    ``episodes`` is a tuple of ``(episode_hash, tar_filename)`` relative to
    ``/mnt/zarr-zip``. Returns ``(batch_id, {eps_str: {episode_hash: entry}})``
    where ``entry = {"raw_total": int, "keep_indices": [int, ...]}``. Failures
    collapse to ``raw_total == 0`` so the consumer surfaces them as cache
    misses (and falls back to in-process precompute) at training time.
    """
    import shutil
    import tarfile
    import time as _time

    import numpy as np
    import zarr

    LEFT_EE = "left.obs_ee_pose"
    RIGHT_EE = "right.obs_ee_pose"
    LEFT_KP = "left.obs_keypoints"
    RIGHT_KP = "right.obs_keypoints"

    eps_strs = [_format_eps(e) for e in epsilons]

    # See fresh tar writes since the container last warmed.
    zip_volume.reload()

    def _keypoint_max_delta(kp: np.ndarray) -> np.ndarray:
        T = kp.shape[0]
        if T < 2 or kp.ndim != 2 or kp.shape[1] % 3 != 0 or kp.shape[1] == 0:
            return np.zeros(max(T - 1, 0))
        n_landmarks = kp.shape[1] // 3
        diff = np.diff(kp.reshape(T, n_landmarks, 3), axis=0)
        per_landmark_norm = np.linalg.norm(diff, axis=-1)
        return per_landmark_norm.max(axis=-1)

    def _keep_indices_from_deltas(
        T: int,
        left_d: np.ndarray,
        right_d: np.ndarray,
        left_kp_d: np.ndarray | None,
        right_kp_d: np.ndarray | None,
        epsilon: float,
    ) -> list[int]:
        if T < 2:
            return list(range(T))
        is_paused = (left_d < epsilon) & (right_d < epsilon)
        if left_kp_d is not None:
            is_paused = is_paused & (left_kp_d < epsilon)
        if right_kp_d is not None:
            is_paused = is_paused & (right_kp_d < epsilon)
        keep = np.ones(T, dtype=bool)
        in_pause = False
        for t in range(1, T):
            if is_paused[t - 1]:
                if in_pause:
                    keep[t] = False
                else:
                    in_pause = True
            else:
                in_pause = False
        return np.flatnonzero(keep).astype(np.int64).tolist()

    def _find_zarr_dir(scratch: Path, episode_hash: str) -> Path | None:
        """Locate the zarr store dir inside an extracted per-episode tar."""
        # Common layouts: scratch/<hash>/zarr.json, or scratch/zarr.json.
        cand = scratch / episode_hash
        if (cand / "zarr.json").exists():
            return cand
        if (scratch / "zarr.json").exists():
            return scratch
        for p in scratch.rglob("zarr.json"):
            return p.parent
        return None

    def _process_episode(
        ep_path: Path, episode_hash: str
    ) -> tuple[str, int, dict[str, list[int]]]:
        try:
            store = zarr.open_group(str(ep_path), mode="r")
        except Exception:
            return (episode_hash, 0, {s: [] for s in eps_strs})
        try:
            left = np.asarray(store[LEFT_EE][:])
            right = np.asarray(store[RIGHT_EE][:])
        except KeyError:
            # No EE pose → keep every frame (consumer-compatible).
            try:
                sample = next(iter(store.array_keys()), None)
                total = int(store[sample].shape[0]) if sample else 0
            except Exception:
                total = 0
            return (episode_hash, total, {s: list(range(total)) for s in eps_strs})
        except Exception:
            return (episode_hash, 0, {s: [] for s in eps_strs})

        T = int(left.shape[0])
        left_kp = right_kp = None
        try:
            left_kp = np.asarray(store[LEFT_KP][:])
        except Exception:
            pass
        try:
            right_kp = np.asarray(store[RIGHT_KP][:])
        except Exception:
            pass
        try:
            if T < 2:
                return (episode_hash, T, {s: list(range(T)) for s in eps_strs})
            left_d = np.linalg.norm(np.diff(left, axis=0), axis=-1)
            right_d = np.linalg.norm(np.diff(right, axis=0), axis=-1)
            left_kp_d = (
                _keypoint_max_delta(left_kp)
                if left_kp is not None and len(left_kp) == T
                else None
            )
            right_kp_d = (
                _keypoint_max_delta(right_kp)
                if right_kp is not None and len(right_kp) == T
                else None
            )
            per_eps = {
                s: _keep_indices_from_deltas(
                    T, left_d, right_d, left_kp_d, right_kp_d, e
                )
                for s, e in zip(eps_strs, epsilons)
            }
        except Exception:
            return (episode_hash, 0, {s: [] for s in eps_strs})
        return (episode_hash, T, per_eps)

    t0 = _time.monotonic()
    out: dict[str, dict[str, dict]] = {s: {} for s in eps_strs}
    n_err = 0

    for episode_hash, tar_filename in episodes:
        tar_p = Path(ZIP_MOUNT_PATH) / tar_filename
        scratch = Path("/tmp") / f"pause_{batch_id}_{episode_hash}"
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        try:
            try:
                with tarfile.open(str(tar_p), "r") as tar:
                    tar.extractall(path=str(scratch))
            except Exception as exc:
                print(
                    f"[pause-zip {batch_id}] extract FAILED {tar_filename}: {exc!r}",
                    file=sys.stderr,
                )
                for s in eps_strs:
                    out[s][episode_hash] = {"raw_total": 0, "keep_indices": []}
                n_err += 1
                continue

            ep_dir = _find_zarr_dir(scratch, episode_hash)
            if ep_dir is None:
                print(
                    f"[pause-zip {batch_id}] no zarr.json in {tar_filename}",
                    file=sys.stderr,
                )
                for s in eps_strs:
                    out[s][episode_hash] = {"raw_total": 0, "keep_indices": []}
                n_err += 1
                continue

            _, raw_total, per_eps = _process_episode(ep_dir, episode_hash)
            if raw_total == 0:
                n_err += 1
            for s in eps_strs:
                out[s][episode_hash] = {
                    "raw_total": raw_total,
                    "keep_indices": per_eps[s],
                }
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    n_episodes = len(episodes)
    summary_parts = []
    for s in eps_strs:
        ep_dict = out[s]
        n_kept = sum(len(v["keep_indices"]) for v in ep_dict.values())
        n_total = sum(v["raw_total"] for v in ep_dict.values())
        pct = (100.0 * n_kept / n_total) if n_total else 100.0
        summary_parts.append(f"eps={s}:{n_kept}/{n_total}({pct:.1f}%)")
    print(
        f"[pause-zip {batch_id}] {n_episodes} eps × {len(eps_strs)} ε | "
        f"{' '.join(summary_parts)} | errors={n_err} | "
        f"{_time.monotonic() - t0:.1f}s"
    )
    return batch_id, out


# ---------------------------------------------------------------------------
# Orchestrator (main training image — needs hydra + egomimic for discovery)
# ---------------------------------------------------------------------------


@app.function(
    cpu=PAUSE_ORCHESTRATOR.cpu,
    memory=PAUSE_ORCHESTRATOR.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        ZIP_MOUNT_PATH: zip_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_pause_precompute_zip(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    config_name: str,
    out_subdir_base: str,
    epsilons: tuple[float, ...],
    init_submodules: bool = True,
    hf_token: str = "",
) -> list[str]:
    """Orchestrator: hydra-compose → resolve zip episodes → fan-out → aggregate.

    Writes one ``cache.json`` per epsilon to
    ``/mnt/zarr-zip/pause_cache/<out_subdir_base>_eps<eps>/cache.json``.
    """
    import json
    import sys as _sys
    import time as _time

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(
        git_remote=git_remote, git_commit=git_commit, init_submodules=init_submodules
    )
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    zip_volume.reload()

    with initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base=None,
    ):
        cfg = compose(config_name=config_name, overrides=list(hydra_args))

    data_cfg = cfg.get("data") if cfg is not None else None
    if data_cfg is None:
        raise RuntimeError(
            f"pause-precompute(zip): composed config '{config_name}' has no `data` group"
        )

    # ── Walk ZipEpisodeResolver blocks → union eps_to_use allowlists + zip_dir ──
    allow_hashes: set[str] | None = set()
    zip_dirs: set[str] = set()
    seen_blocks = 0
    for block_name in ("train_datasets", "valid_datasets"):
        block = OmegaConf.select(data_cfg, block_name)
        if block is None:
            continue
        for ds_name, ds_cfg in block.items():
            resolver = OmegaConf.select(ds_cfg, "resolver") if ds_cfg else None
            target = OmegaConf.select(resolver, "_target_") if resolver else None
            if target is None or "ZipEpisodeResolver" not in str(target):
                continue
            seen_blocks += 1
            zip_dirs.add(str(OmegaConf.select(resolver, "zip_dir")))
            eps_to_use = OmegaConf.select(resolver, "eps_to_use", default=None)
            if not eps_to_use:
                # No allowlist on this block → all catalog episodes are in play.
                allow_hashes = None
                print(
                    f"[pause-precompute(zip)] {block_name}.{ds_name}: no eps_to_use "
                    "— will cover the entire catalog"
                )
                continue
            if allow_hashes is None:
                continue
            path = Path(eps_to_use)
            if not path.is_absolute():
                path = Path(CFG.remote_repo_dir) / eps_to_use
            with open(path) as f:
                hs = set(json.load(f))
            allow_hashes |= hs
            print(
                f"[pause-precompute(zip)] {block_name}.{ds_name}: "
                f"{len(hs)} hashes from {eps_to_use}"
            )

    if seen_blocks == 0:
        raise RuntimeError(
            f"pause-precompute(zip): no ZipEpisodeResolver blocks in '{config_name}'."
        )
    if len(zip_dirs) != 1:
        raise RuntimeError(
            f"pause-precompute(zip): expected one zip_dir, got {sorted(zip_dirs)}"
        )
    zip_dir = Path(next(iter(zip_dirs)))

    # ── Read catalog.json, resolve (hash, tar_filename) for the allowed set ──
    catalog_path = zip_dir / "catalog.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    with open(catalog_path) as f:
        raw_catalog: list[dict] = json.load(f)

    episodes: list[tuple[str, str]] = []
    n_missing_tar = 0
    for e in raw_catalog:
        h = e["episode_hash"]
        if allow_hashes is not None and h not in allow_hashes:
            continue
        tar_filename = e["tar_filename"]
        if not (zip_dir / tar_filename).exists():
            n_missing_tar += 1
            continue
        episodes.append((h, tar_filename))

    # Stable order so batch assignment is deterministic across reruns.
    episodes.sort(key=lambda t: t[0])
    total_eps = len(episodes)
    eps_strs = [_format_eps(e) for e in epsilons]
    if total_eps == 0:
        raise RuntimeError(
            "pause-precompute(zip): 0 episodes resolved "
            f"(allow={None if allow_hashes is None else len(allow_hashes)}, "
            f"missing_tars={n_missing_tar})"
        )
    print(
        f"[pause-precompute(zip)] {total_eps} episodes "
        f"(missing_tars={n_missing_tar}, epsilons={eps_strs}, "
        f"batch={EPISODES_PER_WORKER}, max_containers={PAUSE_MAX_CONTAINERS})"
    )

    # ── Partition into batches; fan out one container per batch ──────────────
    eps_tuple = tuple(float(e) for e in epsilons)
    batches: list[tuple[int, tuple[float, ...], tuple[tuple[str, str], ...]]] = []
    for i in range(0, total_eps, EPISODES_PER_WORKER):
        chunk = tuple(episodes[i : i + EPISODES_PER_WORKER])
        batches.append((len(batches), eps_tuple, chunk))
    total_batches = len(batches)

    caches: dict[str, dict[str, dict]] = {s: {} for s in eps_strs}
    completed = 0
    n_failures = 0
    log_every = max(1, total_batches // 20)
    t0 = _time.time()
    for result in _pause_precompute_zip_batch.starmap(batches, return_exceptions=True):
        completed += 1
        if isinstance(result, Exception):
            n_failures += 1
            print(f"[pause-precompute(zip)] batch FAILED: {result!r}", file=sys.stderr)
        else:
            _, per_eps = result
            for s in eps_strs:
                caches[s].update(per_eps.get(s, {}))
        if completed % log_every == 0 or completed == total_batches:
            elapsed = _time.time() - t0
            n = len(caches[eps_strs[0]]) if eps_strs else 0
            print(
                f"[pause-precompute(zip)] {completed}/{total_batches} batches | "
                f"failures={n_failures} | episodes={n} | elapsed {elapsed:.0f}s"
            )

    elapsed = _time.time() - t0
    summary_lines = []
    for s in eps_strs:
        cache = caches[s]
        n_kept = sum(len(v["keep_indices"]) for v in cache.values())
        n_total = sum(v["raw_total"] for v in cache.values())
        n_miss = sum(1 for v in cache.values() if v["raw_total"] == 0)
        pct = (100.0 * n_kept / n_total) if n_total else 100.0
        summary_lines.append(
            f"  eps={s}: {len(cache)} episodes, kept {n_kept}/{n_total} "
            f"({pct:.1f}%), misses={n_miss}"
        )
    print(
        f"[pause-precompute(zip)] complete in {elapsed:.1f}s\n"
        + "\n".join(summary_lines)
    )

    # ── Write one cache.json per epsilon ────────────────────────────────────
    out_paths: list[str] = []
    for s in eps_strs:
        out_dir = Path(ZIP_MOUNT_PATH) / "pause_cache" / f"{out_subdir_base}_eps{s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "cache.json"
        tmp_path = cache_path.with_suffix(".json.tmp")
        with tmp_path.open("w") as f:
            json.dump(caches[s], f)
        tmp_path.replace(cache_path)
        out_paths.append(str(cache_path))
    zip_volume.commit()

    print("\n=== DONE ===")
    for p in out_paths:
        print(f"pause_precompute_cache: {p}")
    print()
    return out_paths


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_pause_precompute_zip(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a zip pause-precompute job from a pushed commit.

    Recognized non-hydra args (stripped before composition):
      - ``pause_config_name=<name>`` — Hydra training config (required).
      - ``pause_epsilon=<float[,float,...]>`` — epsilons (required).
      - ``init_submodules=<bool>`` — clone with --recurse-submodules.
      - ``name=<str>`` / ``description=<str>`` — label the output subdir.
    """
    config_name: str | None = None
    epsilon_str: str | None = None
    name = description = ""
    args: list[str] = []
    for arg in hydra_args:
        bare = arg.lstrip("+")
        k, sep, v = bare.partition("=")
        if sep and k == "pause_config_name":
            config_name = v
        elif sep and k == "pause_epsilon":
            epsilon_str = v
        else:
            if sep and k == "name":
                name = v
            elif sep and k == "description":
                description = v
            args.append(arg)
    args, init_submodules = pop_init_submodules(args)

    if not config_name:
        raise SystemExit(
            "pause-precompute(zip): pause_config_name=<hydra-config-name> is required"
        )
    if epsilon_str is None:
        raise SystemExit(
            "pause-precompute(zip): pause_epsilon=<float[,float,...]> is required"
        )
    raw_parts = [p.strip() for p in epsilon_str.split(",") if p.strip()]
    if not raw_parts:
        raise SystemExit(
            "pause-precompute(zip): pause_epsilon needs at least one value"
        )
    try:
        epsilons = tuple(float(p) for p in raw_parts)
    except ValueError:
        raise SystemExit(
            f"pause-precompute(zip): pause_epsilon entries must be floats, got {epsilon_str!r}"
        )
    seen: set[float] = set()
    epsilons = tuple(e for e in epsilons if not (e in seen or seen.add(e)))

    import time as _time

    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    out_subdir_base = f"{name or 'pause'}_{description or config_name}_{timestamp}"

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(
        f"Submitting zip pause-precompute at commit {git_commit[:12]} from {git_remote}\n"
        f"  config_name={config_name}  epsilons={list(epsilons)}\n"
        f"  out_subdir_base={out_subdir_base}"
    )

    handle = run_pause_precompute_zip.spawn(
        tuple(args),
        git_remote,
        git_commit,
        config_name,
        out_subdir_base,
        epsilons,
        init_submodules=init_submodules,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted Modal zip pause-precompute job: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


if __name__ == "__main__":
    modal_env = os.environ.copy()
    hydra_args: list[str] = []
    for arg in sys.argv[1:]:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key in MODAL_COMPUTE_ARG_MAP:
            modal_env[MODAL_COMPUTE_ARG_MAP[key]] = val
        else:
            hydra_args.append(arg)

    modal_env["MODAL_APP_NAME"] = app_name_from_hydra_args(hydra_args)
    print(f"Modal app:                           {modal_env['MODAL_APP_NAME']}")
    print(f"Modal pause-precompute orchestrator: {PAUSE_ORCHESTRATOR.summary()}")
    print(
        f"Modal pause-precompute zip worker:   {PAUSE_WORKER.summary()} "
        f"(max_containers={PAUSE_MAX_CONTAINERS}, batch={EPISODES_PER_WORKER})"
    )
    launch_detached(
        Path(__file__).resolve(), "submit_pause_precompute_zip", hydra_args, modal_env
    )
