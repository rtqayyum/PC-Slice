# Study and reproducibility guide

## What is reproduced

The included per-trial observations regenerate the paper's figures directly. They come from the completed repaired V3 workload: two development layouts, eight impairment profiles, two system seeds (31011/31012), and nine methods, giving 288 system executions. The policy setting is the upper route, delay2, budget 24, seed 31013: twelve requests, three models, four retrieval conditions, two decoders, and twelve lexical executions, giving 300 policy executions.

`src/surgical_v3_gate.py` is the actual workload entry point. Inherited configuration lists for earlier studies are not the executed matrix. The run remains development-only and its allocation gate failed. Regenerating figures is not a new experiment, and rerunning a development matrix does not create a held-out final test.

## 1. Inspect the recorded results

```bash
python3 tools/generate_plot_data.py --records study/observations.json --output figures/data
python3 tools/verify_release.py
```

These commands use the Python standard library only. Read `study/measured_summary.json` and `docs/DATA.md` for definitions. Complete raw physics/event logs are not redistributed here; this release includes the reported trial-level metrics, displayed trajectories, source hashes, and plotting data.

## 2. Prepare an isolated experiment environment

Use Python 3.11 and a dedicated writable environment. Do not replace packages in a shared or inherited base environment.

```bash
python3.11 -m venv envs/pcslice_v3
source envs/pcslice_v3/bin/activate
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX
export PYTHONNOUSERSITE=1 PIP_USER=false
python -m pip install --upgrade pip
python -m pip install -r requirements-surgical-v3.txt
```

For the recorded CUDA 12.6 inference environment, on a compatible NVIDIA node:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-inference.txt
```

The original hardware was one A100 80-GB GPU and 20 allocated CPUs. Inference/planning timings depend on hardware, numerical libraries, OS scheduling, and parallel load; new timing values need not equal the archived values.

## 3. Acquire credited sources and checkpoints

From the repository root:

```bash
python -m tools.prepare_sources
python -m tools.download_models
```

The source tool fetches the exact SurRoL revision and the pinned network-parameter text. It does not redistribute the upstream replay recording. Review upstream asset/model licenses and any access conditions. The model downloader uses the exact revisions in `study/checkpoints.json`; all actual inference is executed from downloaded weights. It does not call a hosted model-inference service. The original weight files were not rehashed in the paper's local audit, and published manifest hashes identify the original installation rather than certify a new download.

## 4. Software checks

After acquiring assets and dependencies:

```bash
python -m unittest discover -s tests -p 'test_surgical*.py' -v
```

Tests may use explicitly labeled software fixtures, including placeholder responses; these are not research observations. The original deployment-specific runtime checks are retained separately under `archive/tests/` because they address historical absolute-path launchers and earlier matrices. Public launchers are separate, portable wrappers. Frozen algorithm modules retain their original hashes except the documented unused CLI path normalization.

## 5. Optional rerun on Slurm

On a new checkout on the cluster, prepare the environment, sources, and checkpoints first. The full command is explicit and never run automatically during figure generation:

```bash
module load slurm
bash hpc/submit.sh
```

The launcher queries `gpuq` MaxTime at submission and requests exactly 20 CPUs and one GPU. It refuses a duplicate active `pcslice_comparison` job. Outputs go to `results/comparison_JOBID`; logs go to `logs/pcslice_comparison_JOBID.out` and `.err`. No other projects' jobs are inspected or canceled.

For an allocated shell with those resources, the equivalent entry point is:

```bash
python -m src.surgical_v3_gate --workers 20 --output-dir results/comparison_manual
```

An output folder must be fresh. A failed development comparison is a scientific outcome, not necessarily a process failure; inspect `DEVELOPMENT_STATUS.json`. Do not relax task thresholds or remove failures to obtain a passing gate.

## 6. Figures

The `figures/` directory contains standalone-export tooling and the exact numerical tables used by TikZ/PGFPlots. Install Tectonic to compile vector figures with `python3 figures/export_figures.py`. Captions identify units, sample sizes, quantiles, and adaptation boundaries. No random samples or interpolated benchmark scores are invented for figures.
