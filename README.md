# BOB

BOB is a Python-based causal EEG phase estimator for evoked and resting-state
rhythms. The local pipeline currently supports the onset/theta and LEMON/alpha
datasets and a four-way optimizer ablation.

## Local setup (VS Code on Windows)

Open this repository as the VS Code workspace, then run the following in its
PowerShell terminal. Python 3.11 or 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

VS Code is configured to use `.venv`. If it does not switch automatically,
run **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`.

Confirm data discovery without starting an experiment:

```powershell
python -m Pipeline.Models.Experiments.bob_ablation --band both --zero-phase both --dry-run
```

## BOB optimizer ablation

The default experiment runs Skopt GP, Optuna GP, Optuna TPE, and Optuna's seeded
`RandomSampler` as the random-search arm. Each optimizer uses 100 total objective
evaluations (the first 20 are startup/exploratory evaluations for the adaptive
optimizers), five seeds (`42 43 44 45 46`), and held-out test checkpoints every
10 evaluations. Random search remains random for its complete evaluation budget.

Run the forward-only and zero-phase conditions separately:

```powershell
python -m Pipeline.Models.Experiments.bob_ablation --band theta --zero-phase false
python -m Pipeline.Models.Experiments.bob_ablation --band theta --zero-phase true

python -m Pipeline.Models.Experiments.bob_ablation --band alpha --zero-phase false
python -m Pipeline.Models.Experiments.bob_ablation --band alpha --zero-phase true
```

Alternatively, open **Run and Debug** in VS Code and select **BOB: full optimizer
ablation**. VS Code will prompt for the band and zero-phase condition. The
one-subject random-search launch configuration is a fast end-to-end smoke test.

The runner resumes completed, configuration-matching subject files by default.
Use `--overwrite` only when intentionally replacing selected results. Subject
parallelism is available with `--workers 2`; begin with one worker because the
Optuna GP sampler can use substantial memory.

To interpret the requested budget as 20 startup evaluations plus 100 additional
evaluations, pass `--n-evaluations 120 --n-startup 20`. The default of 100 total
matches the supplied optimizer notebook.

## Outputs

Results are written under:

```text
Pipeline/Results/Raw/BOB/<band>/zero_phase_<true|false>/
├── bob_<band>_zero_phase_<true|false>.json
├── subjects/<subject>.json
└── raw_errors/<subject>.npz
```

The aggregate and per-subject JSON files contain the complete proposal history,
incumbent parameters, calibration objective, test metrics at every checkpoint,
and seed-variability summaries. The compressed NumPy files contain the raw,
wrapped estimated-minus-ground-truth test errors in radians for the default and
every optimizer/seed/checkpoint/timepoint combination.

Run the dependency-light checks with:

```powershell
python -m unittest discover -s tests -v
```

## Code layout

- `Pipeline/data_loader.py`: shared theta/onset and alpha/LEMON loading.
- `Pipeline/Models/bob.py`: BOB signal processing, Burg prediction, and optional
  AR-padded backward/zero-phase pass.
- `Pipeline/Models/Metrics/`: reusable circular metrics and descriptors.
- `Pipeline/Models/Optimizers/`: four optimizer adapters behind one interface.
- `Pipeline/Models/Experiments/bob_ablation.py`: resumable CLI experiment loop
  and result serialization.

## Data availability

The loader discovers the processed `.npz` datasets below
`Pipeline/Data/Processed Data`. Raw and processed data are excluded from Git.
