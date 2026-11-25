"""The figures of the "wind from wind" notebooks, kept here so that the notebooks show the modelling, not the plotting."""

import matplotlib.pyplot as plt
import numpy as np

from windlab import rmse


def hardest_week(error, hours=7 * 24):
    """Indices of the 7-day block (counted from the start) with the largest mean absolute error."""
    n = len(error) // hours * hours
    k = np.argmax(np.abs(error[:n]).reshape(-1, hours).mean(axis=1))
    return np.arange(k * hours, (k + 1) * hours)


def plot_forecasts(time, truth, forecasts, ylabel, title=None):
    """The truth (thick black) and every forecast over the same hours."""
    plt.figure(figsize=(11, 3.5))
    plt.plot(time, truth, "k", lw=2.2, label="truth (ERA5)")
    for name, forecast in forecasts.items():
        plt.plot(time, forecast, lw=1.3, label=name)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(axis="x", rotation=30)
    plt.legend(fontsize="small")
    plt.show()


# --- notebook 1 ---------------------------------------------------------------------------------


def plot_first_look(time, u, v):
    """The first two weeks of the speed, u and v, and the mean speed for every hour of the day."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5), width_ratios=[2, 1])
    weeks, speed = slice(0, 24 * 14), np.hypot(u, v)
    ax1.plot(time[weeks], speed[weeks], "k", lw=1.8, label="speed")
    ax1.plot(time[weeks], u[weeks], lw=0.9, label="u (east)")
    ax1.plot(time[weeks], v[weeks], lw=0.9, label="v (north)")
    ax1.set(ylabel="m/s", title="the first two weeks")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize="small")
    hour = time.astype(int) % 24
    ax2.plot(range(24), [speed[hour == h].mean() for h in range(24)], "k", marker="o")
    ax2.set(xlabel="hour (UTC)", ylabel="mean speed (m/s)", title="the diurnal cycle")
    plt.show()


def plot_rmse_by_horizon(horizons, curves, climatology):
    """RMSE against the forecast horizon for every forecast in `curves`, and climatology as a flat line."""
    plt.figure(figsize=(7, 4))
    for (name, curve), colour in zip(curves.items(), ("C0", "C2", "C3")):
        plt.plot(horizons, curve, colour, label=name)
    plt.axhline(climatology, color="C1", label="climatology")
    plt.xlabel("forecast horizon h (hours)")
    plt.ylabel("RMSE (m/s)")
    plt.legend()
    plt.show()


def plot_best_lines(pairs, best_line, label="speed"):
    """For each horizon h in `pairs` (h -> (x, y)): the data, persistence, climatology and the best line."""
    ends = np.array([min(min(x.min(), y.min()) for x, y in pairs.values()), max(max(x.max(), y.max()) for x, y in pairs.values())])
    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 4.3), sharex=True, sharey=True)
    for ax, (h, (x, y)) in zip(axes, pairs.items()):
        w, b = best_line(x, y)
        ax.plot(x, y, ".", ms=2, alpha=0.15, color="grey")
        ax.plot(ends, ends, "C0--", label="persistence: w = 1, b = 0")
        ax.axhline(y.mean(), color="C1", label=f"climatology: w = 0, b = {y.mean():.2f}")
        ax.plot(ends, w * ends + b, "C2", lw=2.5, label=f"best line: w = {w:.2f}, b = {b:.2f}")
        ax.set(title=f"h = {h} h", xlabel=f"x: {label} now (m/s)", xlim=ends, ylim=ends, aspect="equal")
        ax.legend(fontsize="x-small", loc="upper left")
    axes[0].set_ylabel(f"y: {label} in h hours (m/s)")
    plt.show()


# --- notebook 2 ---------------------------------------------------------------------------------


def plot_landscape(ax, loss, best, paths, marker=None):
    """Contours of loss(w, b) above its lowest point, the paths taken, and the best (w, b) as a star."""
    w_grid, b_grid = np.meshgrid(np.linspace(-0.1, 1.15, 120), np.linspace(-0.6, 0.8, 120))
    excess = np.vectorize(loss)(w_grid, b_grid) - loss(*best)
    ax.contour(w_grid, b_grid, excess, levels=np.geomspace(excess.max() / 3e3, excess.max(), 12), colors="lightgrey")
    for name, path in paths.items():
        ax.plot(*np.transpose(path), lw=1.3, marker=marker, ms=3, label=name)
    ax.plot(*best, "k*", ms=14, label="best line (least squares)")
    ax.set(xlabel="w (weight)", ylabel="b (bias)")
    ax.legend(fontsize="small")


def plot_descent(x, y, path, loss, best, steps=(0, 1, 2, 5, 20, 200)):
    """Left: the data and the line after some steps. Right: the same steps on the loss landscape."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    sample = np.random.default_rng(0).choice(len(x), min(len(x), 1500), replace=False)
    ax1.plot(x[sample], y[sample], ".", ms=2, color="grey", alpha=0.4)
    ends = np.array([x.min(), x.max()])
    for j, k in enumerate(steps):
        w, b = path[k]
        ax1.plot(ends, w * ends + b, color=plt.cm.viridis(j / (len(steps) - 1)), lw=2,
                 label=f"step {k}: w = {w:.2f}, b = {b:.2f}")
    ax1.set(xlabel="x: u now (m/s)", ylabel="y: u in 1 h (m/s)", title="the line after k steps")
    ax1.legend(fontsize="small")
    plot_landscape(ax2, loss, best, {"gradient descent": path}, marker="o")
    ax2.set_title("the same steps on the loss landscape")
    plt.show()


def plot_one_step(path, loss, best, newton, x_scaled, y, scaled_step):
    """Left: the original bowl, with gradient descent and one Newton step. Right: the round bowl of the rescaled input."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_landscape(ax1, loss, best, {f"gradient descent, {len(path) - 1} steps": path}, marker="o")
    ax1.annotate("", xy=newton, xytext=path[0], arrowprops=dict(arrowstyle="-|>", color="C3", lw=2.5))
    ax1.plot([], [], "C3", lw=2.5, label="Newton: one step")
    ax1.legend(fontsize="small")
    ax1.set_title("the original bowl: long and thin")

    w_best, b_best = np.mean(x_scaled * (y - y.mean())), y.mean()  # least squares for an input with mean 0, std 1
    w_grid, b_grid = np.meshgrid(np.linspace(-0.3, w_best + 0.6, 160), np.linspace(-0.6, w_best + 0.3, 160))
    excess = np.vectorize(lambda w, b: np.mean((w * x_scaled + b - y) ** 2))(w_grid, b_grid)
    excess -= excess.min()
    ax2.contour(w_grid, b_grid, excess, levels=np.geomspace(excess.max() / 3e3, excess.max(), 12), colors="lightgrey")
    ax2.annotate("", xy=scaled_step, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="C0", lw=2.5))
    ax2.plot([], [], "C0", lw=2.5, label="gradient descent: one step, lr = 1/2")
    ax2.plot(w_best, b_best, "k*", ms=14, label="best line (least squares)")
    ax2.set(xlabel="w (weight of the rescaled input)", ylabel="b (bias)", title="the rescaled input: a round bowl", aspect="equal")
    ax2.legend(fontsize="small")
    plt.show()


def plot_optimizers(paths, hours_per_step, new_data, loss, best):
    """Left: the paths on the loss landscape. Middle and right: the error on new data per step and per hour read."""
    x_new, y_new = new_data
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_landscape(axes[0], loss, best, paths)
    axes[0].set_title("the paths")
    for name, path in paths.items():
        steps = np.arange(1, len(path))
        errors = [rmse(w * x_new + b, y_new) for w, b in path[1:]]
        axes[1].semilogx(steps, errors, label=name)
        axes[2].semilogx(hours_per_step[name] * steps, errors, label=name)
    one_pass = max(hours_per_step.values())
    axes[2].axvline(one_pass, color="grey", lw=0.8, ls=":")
    axes[2].text(one_pass * 1.15, 0.74, "one pass over\nthe data", color="grey", fontsize="small", va="top")
    for ax, xlabel, title in zip(axes[1:], ("step", "hours of data read"), ("per step", "per hour of data read")):
        ax.set(xlabel=xlabel, ylabel="RMSE on the new year (m/s)", ylim=(0.45, 0.75), title=title)
        ax.legend(fontsize="small", loc="lower left")
    plt.show()


def plot_curve_fit(horizons, w_h, path_tau, curve_loss, tau_best, steps=(0, 1, 2, 5, 10, 100)):
    """Left: the measured weights and the curve after some steps. Right: the same steps on the loss curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(horizons, w_h, "ko", ms=4, label="weight of the best line (least squares)")
    for j, k in enumerate(steps):
        ax1.plot(horizons, np.exp(-horizons / path_tau[k]), color=plt.cm.viridis(j / (len(steps) - 1)), lw=2,
                 label=f"step {k}: τ = {path_tau[k]:.1f} h")
    ax1.set(xlabel="horizon h (hours)", ylabel="weight w (memory)", ylim=(0, 1.05), title="the curve after k steps")
    ax1.legend(fontsize="small")
    taus = np.linspace(1, 60, 300)
    ax2.plot(taus, np.vectorize(curve_loss)(taus), color="lightgrey", lw=2, label="loss L(τ)")
    ax2.plot(path_tau, np.vectorize(curve_loss)(path_tau), "o-", ms=3, label="gradient descent")
    ax2.plot(tau_best, curve_loss(tau_best), "k*", ms=14, label="best τ (trying every value)")
    ax2.set(xlabel="τ (hours)", ylabel="loss", title="the same steps on the loss curve")
    ax2.legend(fontsize="small")
    plt.show()


def plot_training(losses, mse_persistence, steps_per_epoch):
    """Training loss of every model in `losses` against the epoch, with persistence as a dashed line."""
    plt.figure(figsize=(8, 3.5))
    for name, curve in losses.items():
        plt.plot(np.arange(1, len(curve) + 1) / steps_per_epoch, curve, label=name)
    plt.axhline(mse_persistence, color="grey", ls="--", label="persistence")
    lowest = min(min(curve) for curve in losses.values())
    plt.ylim(0.95 * lowest, 1.4 * mse_persistence)  # zoom in: a model that starts from scratch starts far above
    plt.xlabel("epoch")
    plt.ylabel("training loss (m²/s²)")
    plt.legend()
    plt.show()


def plot_error_by_change(forecasts, truth, now, edges=(0, 1, 2, 3, 4, 6, np.inf)):
    """RMSE of every forecast, grouped by how much the wind changed between now and the target hour."""
    change = np.linalg.norm(truth - now, axis=1)
    groups = [(change >= lo) & (change < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    labels = [f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"> {lo:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    plt.figure(figsize=(8, 4))
    for name, forecast in forecasts.items():
        error = forecast - truth
        plt.plot([rmse(error[g], 0) for g in groups], "o-", label=name)
    plt.xticks(range(len(groups)), [f"{lab}\n{g.mean():.0%}" for lab, g in zip(labels, groups)])
    plt.xlabel("change of the wind (m/s), and share of hours")
    plt.ylabel("RMSE (m/s)")
    plt.title("error by the size of the change")
    plt.legend(fontsize="small")
    plt.show()


# --- notebook 3 ---------------------------------------------------------------------------------


def plot_convolution(field, stencil, point, label="u (m/s)"):
    """One 3 x 3 convolution, drawn: the field with the window at `point`, the stencil, and the output map.

    The output at (i, j) is the sum of stencil * field[i-1 : i+2, j-1 : j+2], as in torch.nn.Conv2d (bias 0). The
    edge, where a window would stick out of the map, is left grey; in the network, padding=1 fills it with zeros.
    """
    from fractions import Fraction

    from matplotlib.patches import ConnectionPatch, Rectangle

    def as_text(weight):  # 1/3 rather than 0.33 when the weight is a simple fraction
        fraction = Fraction(float(weight)).limit_denominator(12)
        return str(fraction) if abs(fraction - weight) < 1e-9 else f"{weight:.2f}"

    def write(ax, j, i, text, darkness):  # white text on dark cells, black on light ones
        ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="white" if darkness > 0.6 else "black")

    rows, cols = field.shape
    out = np.full(field.shape, np.nan)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            out[i, j] = np.sum(stencil * field[i - 1:i + 2, j - 1:j + 2])
    i0, j0 = point
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.88")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.4), width_ratios=[1, 0.6, 1])
    bar = dict(orientation="horizontal", shrink=0.8, pad=0.1)

    lim = np.abs(field).max()
    fig.colorbar(ax1.imshow(field, origin="lower", cmap=cmap, vmin=-lim, vmax=lim), ax=ax1, label=label, **bar)
    ax1.add_patch(Rectangle((j0 - 1.5, i0 - 1.5), 3, 3, fill=False, ec="k", lw=2.5))
    for i in range(i0 - 1, i0 + 2):
        for j in range(j0 - 1, j0 + 2):
            write(ax1, j, i, f"{field[i, j]:.2f}", abs(field[i, j]) / lim)
    ax1.set(title="1. the input, with one 3 × 3 window", xlabel="west → east", ylabel="south → north", xticks=[], yticks=[])

    slim = np.abs(stencil).max()
    ax2.imshow(stencil, origin="lower", cmap="PiYG", vmin=-2 * slim, vmax=2 * slim)  # light colours, readable text
    for (i, j), weight in np.ndenumerate(stencil):
        ax2.text(j, i, as_text(weight), ha="center", va="center", fontsize=13)
    ax2.set(title="2. × the stencil of weights w,\nthen add up", xticks=[], yticks=[])

    olim = np.nanmax(np.abs(out))
    fig.colorbar(ax3.imshow(out, origin="lower", cmap=cmap, vmin=-olim, vmax=olim), ax=ax3, label="weighted sum", **bar)
    ax3.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec="k", lw=2.5))
    write(ax3, j0, i0, f"{out[i0, j0]:.1f}", abs(out[i0, j0]) / olim)
    ax3.set(title="3. the sum, at the same point;\nthen slide to the next one", xlabel="west → east", xticks=[], yticks=[])

    for start, end, ax_a, ax_b in (((j0 + 1.5, i0), (-0.5, 1), ax1, ax2), ((2.5, 1), (j0 - 0.5, i0), ax2, ax3)):
        fig.add_artist(ConnectionPatch(start, end, "data", "data", axesA=ax_a, axesB=ax_b, arrowstyle="-|>",
                                       lw=1.5, color="k", mutation_scale=15))
    plt.show()


def plot_field(lons, lats, u, v, hour, grid, centre, time):
    """The wind field at one hour as arrows coloured by speed, with the target point marked."""
    plt.figure(figsize=(6, 4.8))
    shape = lambda a: a.reshape(grid)
    plt.quiver(shape(lons), shape(lats), shape(u[hour]), shape(v[hour]), shape(np.hypot(u[hour], v[hour])))
    plt.colorbar(label="speed (m/s)")
    plt.plot(lons[centre], lats[centre], "r+", ms=16, mew=2, label="target point")
    plt.title(f"10 m wind, {time[hour]} UTC")
    plt.xlabel("longitude (°E)")
    plt.ylabel("latitude (°N)")
    plt.legend(loc="lower right")
    plt.show()
