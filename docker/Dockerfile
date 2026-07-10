# ==========================================================================
# PanColon-CHiPS pipeline image.
#
# Bundles BOTH conda envs (TF1 tiling/HPL + PyTorch SurvCLAM), the vendored
# tools, and openslide, so the whole WSI -> CHiPS + HPC + attention + results
# bundle runs anywhere with a container runtime. GPU is used if present
# (`docker run --gpus all` / `singularity run --nv`); otherwise it runs on CPU.
#
# Weights are NOT baked in (too large, and versioned on Zenodo) — mount them at
# /weights. Mount slides at /data and outputs at /out.  See docker/README.md.
#
#   docker build -t pancolon-chips:latest -f docker/Dockerfile .
# ==========================================================================
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates git procps \
        openslide-tools libopenslide0 && \
    rm -rf /var/lib/apt/lists/*

# ---- Miniconda + libmamba solver (fast solves for the old TF1 stack) -------
# Recent Miniconda gates Anaconda's default channels (pkgs/main, pkgs/r) behind a
# Terms-of-Service acceptance; accept it up front so the base install AND the
# `conda env create` steps below (both env files list the `defaults` channel) can
# solve non-interactively.
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main \
        --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda install -n base -y conda-libmamba-solver && \
    /opt/conda/bin/conda config --set solver libmamba && \
    /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/bin:$PATH

# ---- Build both stage envs (env files first for layer caching) ------------
# Each env brings its own cudatoolkit/pytorch-cuda via conda, so the host only
# needs a compatible NVIDIA driver (injected by the container runtime at run).
COPY envs/ /opt/pancolon/envs/
RUN conda env create -f /opt/pancolon/envs/env_survclam.yml && conda clean -afy
RUN conda env create -f /opt/pancolon/envs/env_tiling.yml   && conda clean -afy

# ---- Repo (code + vendored tools + configs + static viewer) ---------------
COPY . /opt/pancolon/
WORKDIR /opt/pancolon

ENV PANCOLON_CONFIG=/opt/pancolon/config/pipeline.container.yaml
ENTRYPOINT ["/opt/pancolon/docker/entrypoint.sh"]
# No args  -> run the full pipeline then export the results bundle.
# Any args -> passed straight to pancolon_pipeline.py (e.g. `list`, `export`).
CMD []
