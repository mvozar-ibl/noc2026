# Syllabus: Wind from wind

**Event:** Night of Chances (Noc šancí), FEI STU Bratislava, 2026
**Length:** 60 min, hands-on
**Format:** two Jupyter notebooks of 20 min each, and a third one if the group is quick (see Schedule). Participants run them along with the presenter on their own laptop (a CPU is enough), or in Google Colab.

## Who it is for

Students in years 2–3 of a bachelor's or year 1 of a master's degree who program in Python. Typical fields are informatics, robotics and electrical engineering. No machine learning, no meteorology and no GPU are needed: every model is built up from a straight line.

- **Required:** basic Python, and knowing how to run a Jupyter cell.
- **Helpful:** numpy; derivatives and partial derivatives (the chain rule); the inverse and the eigenvalues of a 2 × 2 matrix; what a correlation is.

## Learning outcomes

After the workshop you can:

1. download open weather data with one HTTP request, without an account;
2. build the persistence and climatology baselines, and score a forecast by RMSE and skill;
3. fit a straight-line forecast ŷ = w·x + b twice, by formula (least squares) and by gradient descent, explain why SGD is needed when the data do not fit in memory and what AdamW adds, and check on new data that they learn the same line;
4. fit a curve that has no formula, the wind's memory time τ in w(h) = e^(−h/τ), by gradient descent, and check its slope against autodiff;
5. say what autodiff does, and train a small `torch.nn.Module` with the same loop, in the delta form ŷ = wind now + f(x; θ);
6. read forecasts against the truth to see a model's limits, and test it zero-shot, on a town it never saw in training;
7. on the fast track: train a small CNN on a 9 × 9 wind field with the same loop, and see why the whole field helps.

## Schedule

The whole workshop is 60 minutes. Each notebook is planned for 20 minutes, and the remaining 20 minutes absorb the delays that a hands-on session always has. For a group that turns out to be quicker than expected, a **fast track** fits notebook 3 into the same hour. The presenter decides on the spot.

| Part | Standard track | Fast track |
| --- | --- | --- |
| Notebook 1: open data, and a first forecast | 0–20 min | 0–15 min |
| Notebook 2: how a model learns | 20–40 min | 15–30 min |
| Notebook 3: a small CNN on a wind field | after the workshop | 30–42 min |
| Reserve for delays | 40–60 min | 42–60 min |

### Choosing the track

Both tracks start the same way: notebook 1, sections 1–3, in the first 10 minutes. These are the first four code cells: the download, the first plots, and the scores of the two baselines. At the end of section 3 the presenter decides:

- **Fast track** if almost everyone has run every cell so far without help, and the questions are about the method rather than about Python, numpy or Jupyter.
- **Standard track** otherwise: if setup problems are still open, if several people are behind, or if the questions are about the code itself.

On the fast track every cell still runs. The time comes from the explanations, which the presenter summarises instead of walking through, and from the two exercises of notebook 1, which move to after the workshop. The minutes of every section, for both tracks, are listed below.

### Notebook 1: open weather data, and a first forecast (20 min; fast track 15 min)

Minutes: standard / fast track.

1. One request: ERA5 and the open API, speed and direction → u, v (3 / 3 min)
2. A first look: two weeks of data, the diurnal cycle (2 / 2 min)
3. The names used in all notebooks (input x, target y, forecast ŷ); two forecasts of the wind speed that need no learning: persistence and climatology; RMSE and skill; splitting by time (5 / 5 min). **Then the presenter chooses the track.**
4. The best straight line ŷ = w·x + b for the speed, by least squares; the wind's memory, the autocorrelation ρ, which sets all three scores; forecasts against the truth (8 / 4 min)
5. Your turn: two exercises (2 / 1 min; on the fast track the exercises move to after the workshop)

### Notebook 2: how a model learns, from a straight line to a neural network (20 min; fast track 15 min)

1. Gradient descent by hand: a line like notebook 1's, found by walking downhill and checked against the formula; then in one step instead of 200, by Newton's second-order step (exact here, rarely feasible) or a rescaled input (5 / 4 min)
2. SGD and AdamW, written by hand: steps from batches, for data that do not fit in memory, smarter steps, and the same line on a new year and in a new town (4 / 3 min)
3. Autodiff: PyTorch computes the same gradients and takes the same steps (2 / 1 min)
4. A curve without a formula: the wind's memory time, fitted by gradient descent and checked (3 / 2 min)
5. The same loop, a bigger model: a small MLP in the delta form (3 / 2 min)
6. Forecasts against the truth, their limits, and a zero-shot test in Košice (3 / 3 min)

### Notebook 3: a small CNN on a wind field (fast track 12 min; otherwise after the workshop)

`03_bonus_cnn.ipynb` treats the wind on a 9 × 9 ERA5 grid as an image with channels. The same training loop fits a convolutional block that predicts the wind at the centre 6 hours ahead; it is compared with linear models and an MLP. Its data are in the repository, so it needs no network.

1. The wind field as an image (2 min)
2. Samples: a stack of fields (2 min)
3. A convolutional block; training the CNN and an MLP (4 min)
4. Forecasts against the truth (4 min)

On the test laptop each notebook runs in under a minute, so the time goes to explanation and experiments. Each notebook section is a few sentences, the key formula, the code and one thing to look for. The derivations and background (the skill score, the autocorrelation and the daily cycle, least squares, the gradient, the stable learning rate, why SGD, AdamW line by line, autodiff, the memory curve, the network, the normal equations, the convolution) are in `notes/`, for reading at home.

## Setup

- **Laptop (recommended):** follow the [README](README.md); uv is the easiest way. It takes a few minutes and about 1 GB, so install before the workshop, not on the venue Wi-Fi.
- **Google Colab:** if you can't install locally; you need a Google account.

## Data

ERA5 reanalysis through the [Open-Meteo](https://open-meteo.com/) archive API, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
