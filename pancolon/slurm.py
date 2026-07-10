"""Submit the pipeline to SLURM as a dependency chain and monitor it.

This is the importable twin of ``scripts/slurm/submit_all.sh``: it submits one
``stage.sbatch`` job per step (each ``afterok`` the previous), captures the job
ids, and pins every step's log to a known per-run path so a caller (the webapp)
can tail it without guessing ``%j``. State is read back with ``sacct`` (falling
back to ``squeue`` for very fresh jobs) so it works during and after the run.

    from pancolon import slurm
    jobs = slurm.submit_chain(cfg, "tile", "attention_map", run_dir)
    slurm.query_states([j.job_id for j in jobs])   # {"tile": "running", ...}
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .steps import STEP_ORDER

# Steps that need a GPU (mirrors scripts/slurm/submit_all.sh).
GPU_STEPS = {"project", "infer_survival", "attention_map"}

MANIFEST_NAME = "run_manifest.json"


@dataclass
class StepJob:
    step: str
    job_id: str
    needs_gpu: bool
    log_path: str


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------

def _sbatch_resources(cfg, needs_gpu):
    s = cfg.get("slurm", {}) or {}
    if needs_gpu:
        return ["--partition", s.get("gpu_partition", "gpu4_short"),
                "--gres", s.get("gpu_gres", "gpu:1"),
                "--mem", s.get("gpu_mem", "100G"),
                "--time", s.get("time_gpu", "12:00:00")]
    return ["--partition", s.get("cpu_partition", "cpu_short"),
            "--mem", s.get("cpu_mem", "60G"),
            "--time", s.get("time_cpu", "12:00:00")]


def _stage_script(cfg) -> str:
    return os.path.join(cfg["repo_root"], "scripts", "slurm", "stage.sbatch")


def build_sbatch_argv(cfg, step, run_dir, dep_job_id=None):
    """Return the sbatch argv for one step (used for real runs and dry-run preview)."""
    needs_gpu = step in GPU_STEPS
    s = cfg.get("slurm", {}) or {}
    log_path = os.path.join(run_dir, "logs", f"{step}.out")
    argv = ["sbatch", "--parsable",
            "--job-name", f"pancolon_{step}",
            "--output", log_path,
            "--error", log_path]
    account = s.get("account", "")
    if account:
        argv += ["--account", account]
    argv += _sbatch_resources(cfg, needs_gpu)
    if dep_job_id:
        argv += ["--dependency", f"afterok:{dep_job_id}"]
    argv += ["--export", f"ALL,PANCOLON_STEP={step}", _stage_script(cfg)]
    return argv, log_path, needs_gpu


def _chain_steps(from_step, to_step):
    i0 = STEP_ORDER.index(from_step)
    i1 = STEP_ORDER.index(to_step)
    if i0 > i1:
        raise ValueError(f"--from ({from_step}) is after --to ({to_step}).")
    return STEP_ORDER[i0:i1 + 1]


def submit_chain(cfg, from_step, to_step, run_dir, *, dry_run=False):
    """Submit (or, with dry_run, only resolve) the SLURM dependency chain.

    Returns a list[StepJob]. With dry_run=True the job ids are empty and nothing
    is submitted; the resolved argv is printed so submission can be sanity-checked
    before consuming cluster time.
    """
    steps = _chain_steps(from_step, to_step)
    Path(run_dir, "logs").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PANCOLON_CONFIG"] = cfg["_config_path"]
    env["PANCOLON_REPO"] = cfg["repo_root"]

    jobs: list[StepJob] = []
    prev_jid = None
    for step in steps:
        argv, log_path, needs_gpu = build_sbatch_argv(cfg, step, run_dir, prev_jid)
        if dry_run:
            print(f"[slurm] would submit {step}: "
                  + " ".join(shlex.quote(a) for a in argv))
            jobs.append(StepJob(step, "", needs_gpu, log_path))
            continue
        result = subprocess.run(argv, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed for step '{step}' (rc={result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}")
        jid = result.stdout.strip().split(";")[0]  # "<jobid>[;cluster]"
        jobs.append(StepJob(step, jid, needs_gpu, log_path))
        prev_jid = jid

    if not dry_run:
        write_manifest(run_dir, cfg, from_step, to_step, jobs)
    return jobs


# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------

# SLURM job state -> our coarse status.
_STATE_MAP = {
    "PENDING": "pending", "CONFIGURING": "pending", "REQUEUED": "pending",
    "RUNNING": "running", "COMPLETING": "running", "SUSPENDED": "running",
    "COMPLETED": "done",
    "FAILED": "failed", "CANCELLED": "failed", "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed", "NODE_FAIL": "failed", "BOOT_FAIL": "failed",
    "DEADLINE": "failed", "PREEMPTED": "failed",
}


def _norm(raw_state: str) -> str:
    # sacct decorates cancellations as "CANCELLED by <uid>"; keep the first word.
    key = (raw_state or "").split()[0].upper() if raw_state else ""
    return _STATE_MAP.get(key, "pending")


def query_states(job_ids):
    """Map each job id -> pending|running|done|failed via sacct (squeue fallback)."""
    states = {jid: "pending" for jid in job_ids if jid}
    if not states:
        return states
    id_arg = ",".join(states)
    try:
        out = subprocess.run(
            ["sacct", "-j", id_arg, "--format=JobID,State",
             "--parsable2", "--noheader"],
            capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            job_field = parts[0].split(".")[0]  # drop ".batch"/".extern" steps
            if job_field in states:
                states[job_field] = _norm(parts[1])
    except (OSError, subprocess.SubprocessError):
        pass
    # squeue catches jobs too fresh for the accounting DB.
    try:
        out = subprocess.run(
            ["squeue", "-j", id_arg, "-h", "-o", "%i %T"],
            capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            bits = line.split()
            if len(bits) >= 2 and bits[0] in states:
                states[bits[0]] = _norm(bits[1])
    except (OSError, subprocess.SubprocessError):
        pass
    return states


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------

def manifest_path(run_dir) -> str:
    return os.path.join(run_dir, MANIFEST_NAME)


def write_manifest(run_dir, cfg, from_step, to_step, jobs):
    data = {
        "config_path": cfg["_config_path"],
        "dataset_name": cfg.get("dataset_name", "cohort"),
        "work_dir": cfg.get("paths", {}).get("work_dir", ""),
        "from_step": from_step,
        "to_step": to_step,
        "jobs": [asdict(j) for j in jobs],
    }
    with open(manifest_path(run_dir), "w") as fh:
        json.dump(data, fh, indent=2)
    return manifest_path(run_dir)


def read_manifest(run_dir):
    p = manifest_path(run_dir)
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)
