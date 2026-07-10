"""Subprocess execution with per-stage conda environment activation.

Each pipeline stage runs a vendored upstream script inside a specific conda
environment. `run_stage` wraps that: it optionally activates the stage env
(sourcing the user's conda.sh), changes into the tool directory, and execs the
command. With dry_run=True it only prints the resolved command, which is how
`--dry-run` and the `--help` self-check work.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def _quote(argv):
    return " ".join(shlex.quote(str(a)) for a in argv)


def build_bash(cfg, env_name, argv, cwd=None, extra_env=None, modules=None):
    """Return a single bash -c payload that activates a stage env and runs argv.

    A stage may be entered via a conda env (env_name) and/or one or more
    environment modules (modules, `module load`ed in order). Modules are loaded
    before the conda env, so a module that only sets up CUDA can coexist with a
    conda env; a module that IS the environment is used with env_name=None.
    """
    lines = ["set -euo pipefail"]
    envs = cfg.get("envs", {})
    modules = modules or []
    if modules:
        init = envs.get("module_init", "")
        if init:
            lines.append(f"source {shlex.quote(init)}")
        else:
            # best-effort: make `module` available in a non-interactive shell
            lines.append("command -v module >/dev/null 2>&1 || "
                         "{ [ -f /etc/profile.d/modules.sh ] && "
                         "source /etc/profile.d/modules.sh; } 2>/dev/null || true")
        for m in modules:
            lines.append(f"module load {shlex.quote(m)}")
    if env_name:
        conda_sh = envs.get("conda_sh", "")
        if conda_sh and conda_sh != "CHANGE_ME":
            lines.append(f"source {shlex.quote(conda_sh)}")
        lines.append(f"conda activate {shlex.quote(env_name)}")
    for key, val in (extra_env or {}).items():
        lines.append(f"export {key}={shlex.quote(str(val))}")
    if cwd:
        lines.append(f"cd {shlex.quote(str(cwd))}")
    lines.append(_quote(argv))
    return "\n".join(lines)


def run_stage(cfg, *, step, env, argv, cwd=None, extra_env=None,
              dry_run=False, no_env_switch=False, modules=None):
    """Execute (or print) one stage command.

    step    : short label for logging (e.g. "project")
    env     : conda env name, or None; ignored when no_env_switch is True
    modules : list of environment modules to `module load`; ignored when
              no_env_switch is True
    argv    : list, the command to run
    cwd     : working directory
    """
    env_name = None if no_env_switch else env
    mods = [] if no_env_switch else (modules or [])
    payload = build_bash(cfg, env_name, argv, cwd=cwd, extra_env=extra_env,
                         modules=mods)

    where = env_name or ("module:" + "+".join(mods) if mods else "(current)")
    banner = f"[{step}]"
    if dry_run:
        print(f"{banner} DRY-RUN — would execute in {where}"
              + (f", cwd={cwd}" if cwd else "") + ":")
        for line in payload.splitlines():
            print(f"    {line}")
        return 0

    print(f"{banner} running ({where})"
          + (f", cwd={cwd}" if cwd else ""))
    print(f"{banner} $ {_quote(argv)}")
    sys.stdout.flush()
    result = subprocess.run(["bash", "-lc", payload])
    if result.returncode != 0:
        raise SystemExit(f"{banner} FAILED with exit code {result.returncode}")
    print(f"{banner} done.")
    return 0


def ensure_dir(path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)
