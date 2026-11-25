# Wind from wind

These are Jupyter notebooks for a one-hour workshop. You download open ERA5 wind data with one HTTP call, then train forecasts from scratch with numpy and PyTorch. A laptop CPU is enough.

| Notebook | Content | Time |
| --- | --- | --- |
| `01_open_data.ipynb` | one Open-Meteo request; u, v and the speed; persistence, climatology and the best straight line by formula, for the wind speed; forecasts against the truth | 20 min (fast track 15) |
| `02_sgd_to_adamw.ipynb` | a line like notebook 1's (for the eastward wind u) by gradient descent, SGD and AdamW written by hand, compared on new data; PyTorch autodiff; a curve without a formula, the wind's memory time; the same loop trains a small MLP in the delta form; its limits, and a zero-shot test in Košice | 20 min (fast track 15) |
| `03_bonus_cnn.ipynb` | bonus: the same loop trains a small CNN on a 9 × 9 wind field | fast track only: 12 min |

The notebooks hold what is done live: a few sentences, the key formula and the code of every step. The reasons and derivations behind every step are in `notes/`, one file per notebook, to read at home. The syllabus explains the two tracks: the standard one runs notebooks 1 and 2, and the fast track, for a quick group, also fits notebook 3 into the hour.

## Setup

Install before the workshop rather than on the venue Wi-Fi: it takes a few minutes and about 1 GB. A laptop CPU is enough.

### Recommended: uv

[uv](https://docs.astral.sh/uv/) installs the right Python by itself, and the exact package versions from `uv.lock`. Install it once:

- Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Linux, WSL and macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then:

```sh
git clone https://github.com/<org>/noch26-wind.git
cd noch26-wind
uv sync              # Python 3.12 and the locked packages, into .venv
uv run jupyter lab
```

### Or: pip

This needs Python 3.12, 3.13 or 3.14. Python 3.15 is too new: one of Jupyter's packages has no ready-made Windows build for it yet. On Windows, `py install 3.14` adds 3.14, and `py -3.14 -m venv .venv` uses it in the first line below.

```sh
python -m venv .venv             # python3 on Linux and macOS
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Or: Google Colab

Colab already has numpy, PyTorch and matplotlib. Open a notebook from GitHub (File → Open notebook → GitHub), then run these two lines in a new first cell, so that `windlab.py`, `plots.py` and `data/` are there:

```
!git clone -q https://github.com/<org>/noch26-wind.git
%cd noch26-wind
```

Colab's package versions differ from the pinned ones, so some results can differ in the last digits.

### Tested

On 4 November 2026, all three notebooks ran without errors and printed the same numbers in each of these setups:

| System | Setup | Python |
| --- | --- | --- |
| Windows 11 | uv, on a machine without Python 3.12 (uv downloaded it) | 3.12 |
| Windows 11 | pip | 3.14 |
| Ubuntu 24.04 in WSL 2 | uv | 3.12 |
| Ubuntu 24.04 in WSL 2 | pip | 3.12 |
| Ubuntu 24.04 in WSL 2 | Colab's package versions: torch 2.11, numpy 2.4, matplotlib 3.10 | 3.13 |

The install took between 10 seconds (uv with a filled cache) and under 3 minutes (pip on Windows).

## Files

- `SYLLABUS.md`: audience, learning outcomes and the 60-minute schedule.
- `windlab.py`: the helpers, about 170 lines: the data download, the samples, least squares, standardisation, the MLP, the warm-up + cosine schedule and the training loop.
- `plots.py`: the figures of the notebooks, so that the notebooks show the modelling, not the plotting.
- `notes/`: the background and derivations for each notebook, with code snippets that reproduce every number quoted.
- `data/`: ERA5 snapshots, so that notebooks 2 and 3 need no network.
    - 3 × 3 grid points around Bratislava for 2020–2024 and around Košice for 2024;
    - 9 × 9 points around Bratislava for 2023–2024, stored as float16 (rounding at most 0.004 m/s).

## Data and credits

Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It comes from the ERA5 reanalysis of the Copernicus Climate Change Service.

The snapshots in `data/` were downloaded with `windlab.fetch_wind` (`models=era5`). Speed and direction were converted to the u and v components.

The free Open-Meteo API is for non-commercial use, at under 10,000 calls a day and 600 a minute. One run of notebook 1 costs about 10 calls. Downloading the 9 × 9 field again costs about 850 calls.

## Licence

Pending: the company decides before publication.
