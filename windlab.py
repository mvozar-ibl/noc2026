"""Small helpers for the "wind from wind" workshop notebooks: data, samples, models and training (figures: plots.py)."""

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ERA5_STEP = 0.25  # ERA5 grid spacing in degrees


# --- data -------------------------------------------------------------------------------------


def wind_to_uv(speed, direction_deg):
    """Speed and meteorological direction (where the wind blows FROM) -> u (east), v (north)."""
    rad = np.deg2rad(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def patch(lat, lon, radius=1):
    """A (2r+1) x (2r+1) square of ERA5 grid points around (lat, lon), row by row; the centre is the middle one."""
    lat0, lon0 = (round(deg / ERA5_STEP) * ERA5_STEP for deg in (lat, lon))
    offsets = ERA5_STEP * np.arange(-radius, radius + 1)
    lats, lons = np.meshgrid(lat0 + offsets, lon0 + offsets, indexing="ij")
    return lats.ravel(), lons.ravel()


def fetch_wind(lats, lons, start, end, cache=None):
    """Hourly 10 m wind from ERA5 via the Open-Meteo archive API, all points in one call.

    Returns time (hours,) and u, v (hours, points) in m/s as float32. With `cache`, a saved file is reused.
    """
    if cache and Path(cache).exists():
        data = np.load(cache)
        return data["time"], data["u"].astype(np.float32), data["v"].astype(np.float32)
    query = urllib.parse.urlencode({
        "latitude": ",".join(f"{deg:.2f}" for deg in np.atleast_1d(lats)),
        "longitude": ",".join(f"{deg:.2f}" for deg in np.atleast_1d(lons)),
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "models": "era5",
    })
    with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=300) as response:
        points = json.load(response)
    points = points if isinstance(points, list) else [points]  # a single point comes as one object
    time = np.array(points[0]["hourly"]["time"], dtype="datetime64[m]").astype("datetime64[h]")
    speed, direction = (
        np.array([p["hourly"][key] for p in points], dtype=np.float32).T
        for key in ("wind_speed_10m", "wind_direction_10m")
    )
    u, v = wind_to_uv(speed, direction)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, time=time, u=u, v=v)
    return time, u, v


def year(time):
    """Calendar year of datetime64 values."""
    return time.astype("datetime64[Y]").astype(int) + 1970


def make_samples(inputs, targets, lags=1, horizon=1):
    """Input x: `inputs` at hours t-lags+1 .. t, flattened. Target y: `targets` at hour t+horizon.

    Returns X (n, lags * input channels), Y (n, target channels) and t (n,), the index of hour t.
    """
    inputs, targets = (arr.reshape(len(arr), -1) for arr in (inputs, targets))
    n = len(inputs) - lags - horizon + 1
    X = np.concatenate([inputs[i : i + n] for i in range(lags)], axis=1)
    t = np.arange(lags - 1, lags - 1 + n)
    return X, targets[t + horizon], t


def field_samples(u, v, lags, horizon):
    """Wind field of a patch (u and v at every point) -> wind at the centre point `horizon` hours later.

    Returns X (inputs), Y (targets), the centre wind at hour t ("now", which is the persistence forecast) and t.
    """
    centre = u.shape[1] // 2
    targets = np.stack([u[:, centre], v[:, centre]], axis=1)
    X, Y, t = make_samples(np.concatenate([u, v], axis=1), targets, lags, horizon)
    return X, Y, targets[t], t


def rmse(forecast, truth):
    """Root-mean-square error of the forecast against the truth, over all samples and components, in m/s."""
    return float(np.sqrt(np.mean((np.asarray(forecast) - np.asarray(truth)) ** 2)))


# --- models and training ----------------------------------------------------------------------


def least_squares(X, Y):
    """Weights W (inputs, outputs) and biases b (outputs,) with the smallest mean squared error of X @ W + b against Y.

    The least-squares formula of notebook 1, for many inputs at once.
    """
    Wb, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)
    return Wb[:-1], Wb[-1]


class Standardize(nn.Module):
    """(x - mean) / std with statistics fixed from the training inputs."""

    def __init__(self, x):
        super().__init__()
        x = torch.as_tensor(x, dtype=torch.float32)
        self.register_buffer("mean", x.mean(0))
        self.register_buffer("std", x.std(0))

    def forward(self, x):
        return (x - self.mean) / self.std


def mlp(x_train, n_out, hidden=64, depth=2):
    """Standardize -> depth x (Linear + GELU) -> Linear."""
    layers, n = [Standardize(x_train)], x_train.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(n, hidden), nn.GELU()]
        n = hidden
    return nn.Sequential(*layers, nn.Linear(n, n_out))


def warmup_cosine(optimizer, total_steps, warmup=0.05):
    """Learning rate: linear warm-up over the first `warmup` fraction of steps, then cosine decay to 0."""
    n_warm = max(1, int(warmup * total_steps))

    def factor(step):
        if step < n_warm:
            return (step + 1) / n_warm
        return 0.5 * (1 + math.cos(math.pi * (step - n_warm) / max(1, total_steps - n_warm)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def fit(model, X, Y, optimizer, epochs=1, batch_size=1024, scheduler=None, callback=None, seed=0):
    """Minibatch training on the MSE loss; returns the loss of every step.

    `callback(model)` runs after every step, e.g. to record the parameters.
    """
    X, Y = (torch.as_tensor(arr, dtype=torch.float32) for arr in (X, Y))
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=generator).split(batch_size):
            loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            if callback is not None:
                callback(model)
    return losses


@torch.no_grad()
def predict(model, X):
    """Model output for numpy inputs, as numpy."""
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy()
