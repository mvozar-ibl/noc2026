# Notes to notebook 1: open weather data, and a first forecast

These notes hold the background and the derivations; the workshop runs the notebook only. The sections follow the notebook. Every code snippet runs in the notebook after its last cell.

## 1. One request

**ERA5** is a *reanalysis*: a weather model run over the past and corrected with millions of observations. It gives the weather for every hour since 1940 on a grid of 0.25° (about 25 km). Open-Meteo serves it through a free web API, so one HTTP request returns years of hourly data. `fetch_wind` in `windlab.py` builds the request and unpacks the answer; `models=era5` pins the ERA5 data, so everyone gets the same numbers.

**Why components, not speed and direction.** A direction is an angle on a circle: 359° and 1° are almost the same wind, but their average, 180°, is the opposite one. Averages, differences and squared errors work for numbers on a line, so we describe the wind as an arrow with an east part $u$ and a north part $v$.

**The signs, on two examples.** Wind from the north ($\varphi = 0^\circ$) at 5 m/s blows towards the south: $\sin 0^\circ = 0$ and $\cos 0^\circ = 1$ give $u = 0$ and $v = -5$ m/s. Wind from the west ($\varphi = 270^\circ$) at 5 m/s: $\sin 270^\circ = -1$ and $\cos 270^\circ = 0$ give $u = +5$ m/s, towards the east, and $v = 0$. The minus signs turn "from" into "towards". Back to the speed: $\text{speed} = \sqrt{u^2 + v^2}$, `np.hypot(u, v)` in the code. Notebook 1 forecasts the speed, which is easy to read; notebook 2 forecasts the components, which a network needs to tell a westerly from an easterly.

## 2. A first look

**Why the early afternoon is windiest.** The peak is at 13 UTC, 14–15 h local time, when the ground is warmest. The sun warms the ground, the air near the ground mixes with the faster air above, and the wind at 10 m picks up speed. At night the ground cools, the lowest air settles, and the 10 m wind slows down.

**Seams in the data.** The small jumps of the diurnal cycle at 9–10 and 21–22 UTC are not weather: ERA5 is computed in 12-hour pieces that start at 9 and 21 UTC, and neighbouring pieces do not join perfectly. Every dataset has such seams.

## 3. Two forecasts that need no learning

**The names**, the same in all notebooks and in `windlab.py`:

| Symbol | In the code | Meaning |
| --- | --- | --- |
| $t$ | `t` | the hour "now" |
| $h$ | `h`, `HORIZON` | how many hours ahead we forecast |
| $x$ | `x`, `X` | the **input**: what we know at hour $t$; in notebook 1 the wind speed now |
| $y$ | `y`, `Y` | the **target**: the true wind at hour $t + h$ |
| $\hat y$ | `forecast` | a forecast of $y$; its **error** is $\hat y - y$ |
| $\bar y$ | `y.mean()` | the average of $y$ over the training hours |
| $N$ | | the number of hours in a sum or an average |
| $w$, $b$ | `w`, `b` | the **weight** and the **bias** of a line $\hat y = w\,x + b$ |

**Why square the errors.** Squaring makes every miss count as positive and punishes big misses more: one miss of 4 m/s counts as much as sixteen misses of 1 m/s. The root at the end of the RMSE brings the unit back to m/s.

**The skill score.** Skill compares a forecast with a reference forecast, in the notebooks persistence: $1 - \mathrm{MSE}_\text{model}/\mathrm{MSE}_\text{reference}$, with $\mathrm{MSE} = \mathrm{RMSE}^2$ the mean squared error. 1 is a perfect forecast, 0 is as good as the reference, and below 0 is worse. It is the standard *mean squared error skill score* of forecast verification: A. H. Murphy, "Skill scores based on the mean square error and their relationships to the correlation coefficient", *Monthly Weather Review* 116, 2417–2424 (1988); D. S. Wilks, *Statistical Methods in the Atmospheric Sciences*, the chapter on forecast verification. Because it compares squares, a skill of 0.15 means an RMSE about 8 % below the reference's: $\sqrt{1 - 0.15} = 0.92$.

**The best constant forecast is the average.** Which single number $c$ is the best forecast, if it may not depend on anything? Its loss, the mean squared error, is $L(c) = \frac1N\sum (c - y)^2$, a parabola in $c$. At its lowest point the slope is zero:

$$\frac{dL}{dc} = \frac2N\sum (c - y) = 2\,(c - \bar y) = 0 \quad\Longrightarrow\quad c = \bar y.$$

So climatology is the best constant forecast on the training hours. Section 4 repeats this argument for a line.

**Why split by time.** Neighbouring hours are almost equal: one hour ahead, persistence misses by only 0.5 m/s. With a random split, almost every test hour would have its neighbours in the training data, and a model could score well just by remembering them. A split by time tests what we want: forecasting a future the model has not seen.

## 4. The best straight line

**The problem, stated exactly.** The model is the line $\hat y = w\,x + b$. Its loss on the $N$ training hours is $L(w, b) = \frac1N\sum(w\,x + b - y)^2$, and the best line is the pair $(w^*, b^*)$ with the smallest loss of all pairs: $L(w^*, b^*) \le L(w, b)$ for every $w$ and $b$. The stars mark the best values; without a star, $w$ and $b$ are any values. Below: how to read the solution, how to derive it, why it really is the lowest point, and what it guarantees.

**How to read the formula.** The bar means "average over the training hours", and $x - \bar x$, $y - \bar y$ are the **deviations** from the average, now and $h$ hours later. The numerator, the **covariance**, is large when the two deviations tend to have the same sign: a wind stronger than usual now stays stronger than usual later. The denominator, the **variance**, is the typical squared size of the deviation now. Their ratio $w$ is the share of the present deviation still there $h$ hours later: $w = 1$ keeps all of it (persistence), $w = 0$ forgets all of it (climatology). Then $b$ makes the line pass through the average point $(\bar x, \bar y)$.

**Deriving least squares.** At the lowest point of the loss its slope is zero in every direction: $\partial L/\partial b = 0$ and $\partial L/\partial w = 0$ ($\partial$ reads "the slope of $L$ when only this one parameter changes").

*Step 1, the bias.* By the chain rule, the slope of a square is twice the thing inside the square, times the slope of that thing: $\frac{\partial}{\partial b}(w x + b - y)^2 = 2\,(w x + b - y)$. Averaging over the hours,

$$\frac{\partial L}{\partial b} = \frac2N\sum\left(w x + b - y\right) = 2\left(w\bar x + b - \bar y\right) = 0 \quad\Longrightarrow\quad b = \bar y - w\,\bar x.$$

*Step 2, the weight.* With this $b$ the line is $\hat y = \bar y + w\,(x - \bar x)$, and its error is $\hat y - y = w\,(x - \bar x) - (y - \bar y)$. The loss now depends on $w$ alone:

$$\frac{dL}{dw} = \frac2N\sum\Big(w\,(x - \bar x) - (y - \bar y)\Big)(x - \bar x) = 2\Big(w\;\overline{(x - \bar x)^2} - \overline{(x - \bar x)(y - \bar y)}\Big) = 0$$

gives the formula for $w$. The function `best_line` is exactly these two results.

*Both at once: the normal equations.* Written out, the two conditions are two linear equations in $w$ and $b$:

$$\begin{pmatrix}\overline{x^2} & \bar x \cr \bar x & 1\end{pmatrix}\begin{pmatrix}w \cr b\end{pmatrix} = \begin{pmatrix}\overline{x\,y} \cr \bar y\end{pmatrix}.$$

They have exactly one solution when the determinant, $\overline{x^2} - \bar x^2 = \overline{(x - \bar x)^2}$, the variance of $x$, is not zero: unless the input never changes. Notebook 2 meets this matrix again, doubled, as the Hessian, the matrix of the second derivatives of the loss, and the notes to notebook 2 (section 6) give the same equations for many inputs.

**Why this is the lowest point.** A zero slope alone also holds at a top or a saddle. Here it is the lowest point, and the only one. Write $e^* = w^* x + b^* - y$ for the errors of the best line; the two conditions say $\overline{e^*} = 0$ and $\overline{e^* x} = 0$. Any other line has the errors $w x + b - y = e^* + d$, with $d = (w - w^*)\,x + (b - b^*)$. Then

$$L(w, b) = \overline{(e^* + d)^2} = \overline{e^{*2}} + 2\,\overline{e^* d} + \overline{d^2} = L(w^*, b^*) + \overline{d^2},$$

because $\overline{e^* d} = (w - w^*)\,\overline{e^* x} + (b - b^*)\,\overline{e^*} = 0$. As $\overline{d^2} \ge 0$, no line has a smaller loss, and only $(w^*, b^*)$ itself has the same loss, as soon as $x$ takes two different values.

**What the best line guarantees.** Four facts follow from the two conditions:

1. it passes through the average point $(\bar x, \bar y)$;
2. its errors average to zero, $\overline{\hat y - y} = 0$: it is too high as often, and by as much, as it is too low;
3. its errors are uncorrelated with the input, $\overline{(\hat y - y)(x - \bar x)} = 0$: no line in $x$ can explain what is left of the error, so a better forecast needs other inputs;
4. with $\rho$ the correlation of $x$ and $y$, and $\sigma_x$, $\sigma_y$ their spreads, $w^* = \rho\,\sigma_y/\sigma_x$ and the smallest loss is $L(w^*, b^*) = \sigma_y^2\,(1 - \rho^2)$.

Fact 4 is the formula for $w^*$ with the covariance written as $\rho\,\sigma_x\sigma_y$; putting it into the error of step 2 gives the loss $w^{*2}\sigma_x^2 - 2w^*\rho\,\sigma_x\sigma_y + \sigma_y^2 = \sigma_y^2(1 - \rho^2)$. A check on the 6-hour line of the notebook, on its 2023 training hours:

```python
error = w * x + b - y  # the errors of the best line on its training hours (2023, 6 h ahead)
rho_xy = np.corrcoef(x, y)[0, 1]
print(f"1. at the average x the line gives {w * x.mean() + b:.4f}, the average y is {y.mean():.4f}")
print(f"2. average error {error.mean():+.1e} m/s")
print(f"3. average of error times (x - mean x): {np.mean(error * (x - x.mean())):+.1e}")
print(f"4. w = {w:.4f}, rho std(y) / std(x) = {rho_xy * y.std() / x.std():.4f}; "
      f"loss {np.mean(error**2):.4f}, var(y) (1 - rho^2) = {y.var() * (1 - rho_xy**2):.4f}")
```

All four hold: at the average speed the line gives 3.2084 m/s, the average of $y$; the two averages of the errors are about $10^{-7}$, zero up to the rounding of the 32-bit numbers in the data; and $w = 0.6151$ and the loss, 1.8036 m²/s², agree with fact 4.

**Why the line flattens.** Over a year, $x$ and $y$ are the same series shifted by $h$ hours, so they have nearly the same spread, and $w$ is then the **correlation** between the wind now and $h$ hours later. The further ahead, the weaker the link, so the forecast is pulled towards the average: *regression towards the mean*.

**Why it cannot lose to the baselines on the training data.** Persistence and climatology are lines too, $(w, b) = (1, 0)$ and $(0, \bar y)$. The best line has the smallest loss of all lines (shown above), so its loss is at most theirs. On new data, 2024, this is not guaranteed, but it holds when the years are alike, as the plot shows.

**Three forecasts, one number: the autocorrelation.** The correlation between the speed now and the speed $h$ hours later is the **autocorrelation** $\rho$: the correlation of the wind with itself, $h$ hours apart. As $x$ and $y$ have about the same average $\bar y$ and the same spread $\sigma$ (the standard deviation), the best line's weight is $w = \rho$ and its forecast is $\hat y = \bar y + \rho\,(x - \bar y)$. Write each error with the deviations $x - \bar y$ and $y - \bar y$ and average its square: the average of a squared deviation is $\sigma^2$, and that of the product of the two deviations is $\rho\,\sigma^2$. The mean squared errors of the three forecasts are then

- persistence, error $(x - \bar y) - (y - \bar y)$: $\;\sigma^2 + \sigma^2 - 2\rho\,\sigma^2 = 2\sigma^2(1 - \rho)$;
- climatology, error $\bar y - y$: $\;\sigma^2$;
- the best line, error $\rho\,(x - \bar y) - (y - \bar y)$: $\;\rho^2\sigma^2 + \sigma^2 - 2\rho^2\sigma^2 = \sigma^2(1 - \rho^2)$.

Three things follow.

- *Climatology beats persistence* once $2(1 - \rho) > 1$, that is once $\rho < \tfrac12$: the crossing in the plot of section 3 is the horizon at which the autocorrelation drops below ½.
- *The line never loses to either*: $1 - \rho^2 \le 1$, and $2(1 - \rho) - (1 - \rho^2) = (1 - \rho)^2 \ge 0$. It ties with persistence only at $\rho = 1$ and with climatology only at $\rho = 0$; in between it wins by pulling the forecast towards the average by the factor $\rho$.
- *Its skill* against persistence is $1 - \frac{1 - \rho^2}{2(1 - \rho)} = \frac{1 - \rho}{2}$, and against climatology $\rho^2$ (Murphy 1988, above).

A check on 2024:

```python
rho = np.array([np.corrcoef(speed[test_hours[:-h]], speed[test_hours[:-h] + h])[0, 1] for h in horizons])
sigma = speed[test_hours].std()
print(f"rho < 1/2 from {horizons[np.argmax(rho < 0.5)]} h; climatology beats persistence from "
      f"{horizons[np.argmax(rmse_persistence > rmse_climatology)]} h")
for h in (1, 6, 12, 24):
    r = rho[h - 1]
    print(f"{h:2d} h, rho = {r:.3f}. RMSE measured (from rho): persistence {rmse_persistence[h - 1]:.3f} "
          f"({sigma * np.sqrt(2 * (1 - r)):.3f}), best line {rmse_line[h - 1]:.3f} ({sigma * np.sqrt(1 - r**2):.3f}), "
          f"climatology {rmse_climatology:.3f} ({sigma:.3f})")
```

The autocorrelation drops below ½ at 10 hours, the horizon from which climatology beats persistence. The RMSEs computed from $\rho$ match the measured ones within 0.002 m/s for persistence and 0.015 m/s for the line, whose measured error is a little larger because it was fitted on 2023; for climatology the gap of 0.007 m/s comes from the 2023 average being a little off for 2024.

**Why the line is exactly as late as persistence.** As a forecast, the line is $\hat y = w \cdot (\text{wind now}) + b$: the persistence curve, multiplied by $w$ and shifted by $b$. Multiplying and shifting change the height of a curve, never its timing, so every turn of the line comes at the same moment as the turn of persistence, $h$ hours after the truth. The line wins only by shrinking the swings. A check:

```python
line, persistence = w * x_test + b, x_test
print(f"correlation of the line with persistence: {np.corrcoef(line, persistence)[0, 1]:.4f}")
for name, f in [("persistence", persistence), ("best line", line)]:
    match = [np.corrcoef(f[k:], y_test[:len(y_test) - k])[0, 1] for k in range(13)]
    print(f"{name}: best match with the truth {np.argmax(match)} hours earlier")
```

It prints a correlation of 1.0000, and a best match 6 hours earlier for both. In the week of the notebook even the daily rise and fall of the speed, which the clock alone could foresee, arrives 6 hours late: the hour of the day is not among the line's inputs.

**The best line is not the best linear model.** It is the best line with *one input*, the speed now. Linear models with more inputs, fitted by the same least squares, can do better. Fitted on 2023 and scored on 2024, 6 hours ahead, for the wind speed at the grid point of notebooks 2 and 3 (notebook 1's numbers agree to rounding). The hour of the day enters in two ways: as a sine and a cosine of the hour, for each of the 6 hours, or as 24 categories, 23 inputs of 0 or 1 for the hour now (the 24th is the bias).

```python
from windlab import least_squares, make_samples, patch

time_p, u_p, v_p = fetch_wind(*patch(48.15, 17.11), "2020-01-01", "2024-12-31", cache="data/bratislava_patch_2020-2024.npz")
u_p, v_p = u_p.astype(float), v_p.astype(float)
speed_p, centre = np.hypot(u_p, v_p), 4  # the 3 x 3 patch; its centre is the grid point of notebooks 2 and 3
target = speed_p[:, centre]
hour_p = time_p.astype(int) % 24
sine_cosine = np.c_[np.sin(2 * np.pi * hour_p / 24), np.cos(2 * np.pi * hour_p / 24)]
field9 = np.load("data/bratislava_field9_2023-2024.npz")  # the 9 x 9 field, 2023-2024
u9, v9, start9 = field9["u"].astype(float), field9["v"].astype(float), len(time_p) - len(field9["time"])

def linear_model(inputs, lags, hours=time_p, truth=target, categories=False, fit_year=2023):
    # Least squares 6 h ahead, fitted on fit_year and scored on 2024
    X, Y, t = make_samples(inputs, truth, lags, 6)
    if categories:  # the hour of the day at hour t, as 23 inputs of 0 or 1
        X = np.c_[X, (hours[t].astype(int) % 24)[:, None] == np.arange(1, 24)]
    fit_rows, score_rows = year(hours[t]) == fit_year, year(hours[t]) == 2024
    W, b_fit = least_squares(X[fit_rows], Y[fit_rows])
    forecast, truth_2024 = (X[score_rows] @ W + b_fit)[:, 0], Y[score_rows][:, 0]
    error, error_persistence = rmse(forecast, truth_2024), rmse(truth[t[score_rows]], truth_2024)
    match = [np.corrcoef(forecast[k:], truth_2024[:len(truth_2024) - k])[0, 1] for k in range(13)]
    return (f"{X.shape[1]:3d} inputs, RMSE {error:.3f} (persistence {error_persistence:.3f}), "
            f"skill {1 - (error / error_persistence) ** 2:+.3f}, best match {np.argmax(match)} h earlier")

one_point = np.c_[target, u_p[:, centre], v_p[:, centre]]
print("best line:                    ", linear_model(target, 1))
print("speed, last 6 h:              ", linear_model(target, 6))
print("speed, u, v, last 6 h:        ", linear_model(one_point, 6))
print("  + hour as sine and cosine:  ", linear_model(np.c_[one_point, sine_cosine], 6))
print("  + hour as 24 categories:    ", linear_model(one_point, 6, categories=True))
print("3 x 3 field, last 6 h:        ", linear_model(np.c_[speed_p, u_p, v_p], 6))
print("9 x 9 field, last 3 h:        ", linear_model(np.c_[np.hypot(u9, v9), u9, v9], 3, hours=field9["time"], truth=target[start9:]))
print("best line fitted on 2024 (cheating):", linear_model(target, 1, fit_year=2024))
```

| Linear model | Inputs | RMSE (m/s) | Skill | Best match with the truth |
| --- | --- | --- | --- | --- |
| persistence | – | 1.417 | 0 | 6 h earlier |
| best line: the speed now | 1 | 1.303 | +0.154 | 6 h earlier |
| the speed, the last 6 hours | 6 | 1.293 | +0.167 | 6 h earlier |
| the speed, u and v, the last 6 hours | 18 | 1.303 | +0.154 | 6 h earlier |
| the same, and the hour of the day as a sine and a cosine | 30 | 1.260 | +0.209 | 6 h earlier |
| the same, and the hour of the day as 24 categories | 41 | 1.249 | +0.223 | 6 h earlier |
| the 3 × 3 field (speed, u, v), the last 6 hours | 162 | 1.237 | +0.238 | 5 h earlier |
| the 9 × 9 field (speed, u, v), the last 3 hours | 729 | 1.200 | +0.282 | 5 h earlier |

A line fitted on 2024 itself, which is cheating, gains only 0.006 m/s (1.297 instead of 1.303): within its own family, the best line is as good as it gets. More history at one point barely helps. The hour of the day does, because the speed has a daily cycle, and more as 24 categories, which can follow any shape of the cycle, than as one sine and cosine. And only the wind around the point, which arrives later, makes the forecast earlier.

**The daily cycle in the climatology.** The right panel of section 2 is a climatology of its own: the average speed for each hour of the day. The daily cycle also shows in the autocorrelation, which does not simply fall: around 48 hours it rises again, because the target is once more at the same hour of the day as now. The cycle can be moved into the climatology. Forecast the 2023 average $c$ of the target's hour of the day, and let a line work only on the **anomaly**, the deviation of the speed from the average of its hour:

$$\hat y = c_{t+h} + w\,(x - c_t) + b,$$

with $c_t$ the average speed at the hour of the day of hour $t$, and $w$, $b$ the best line for the anomaly, by the formula of section 4.

```python
hour_of_day = time.astype(int) % 24  # UTC
c = np.array([speed[train_hours][hour_of_day[train_hours] == k].mean() for k in range(24)])  # 2023, each hour of the day
anomaly = speed - c[hour_of_day]
print(f"climatology: one average {rmse_climatology:.3f} m/s, one for each hour of the day "
      f"{rmse(c[hour_of_day[test_hours]], speed[test_hours]):.3f} m/s")
for name, series in [("speed", speed), ("anomaly", anomaly)]:
    print(f"autocorrelation of the {name} at 36, 42, 48 h:",
          ", ".join(f"{np.corrcoef(series[test_hours[:-h]], series[test_hours[:-h] + h])[0, 1]:.3f}" for h in (36, 42, 48)))
h = 6
w_anomaly, b_anomaly = best_line(anomaly[train_hours[:-h]], anomaly[train_hours[:-h] + h])
forecast = c[hour_of_day[test_hours[:-h] + h]] + w_anomaly * anomaly[test_hours[:-h]] + b_anomaly
print(f"6 h: best line {rmse(w * x_test + b, y_test):.3f} m/s, the hour's average + a line on the anomaly "
      f"{rmse(forecast, speed[test_hours[:-h] + h]):.3f} m/s")
print(f"persistence: largest RMSE {rmse_persistence.max():.3f} m/s at {horizons[np.argmax(rmse_persistence)]} h, "
      f"{rmse_persistence[-1]:.3f} m/s at 48 h")
```

At this grid point the daily cycle is weak, so one average for each hour of the day gains little over one average for all hours: 1.743 instead of 1.768 m/s. But the anomaly's autocorrelation keeps falling where the speed's rises again: 0.158, 0.133 and 0.128 at 36, 42 and 48 hours, against 0.132, 0.128 and 0.153. The same ripple shows in persistence, whose error is $2\sigma^2(1 - \rho)$: it peaks at 39 hours (2.340 m/s) and is a little smaller at 48 hours (2.295 m/s), when the hour of the day matches again. And 6 hours ahead, the hour's average plus a line on the anomaly scores 1.249 m/s: better than the best line (1.303), and as good as the 41-input linear model of the table with the hour as 24 categories, with only 2 parameters and 24 averages. A climatology for what the clock foresees and a fading memory for the rest: this split is a common baseline in forecasting, known as damped persistence of the anomaly.

## More exercises

1. In the week plot, set `h = 1` or `h = 24`. How do the three forecasts look then?
2. Run the timing check above for $h = 1$ and $h = 24$ (rerun the week cell first). Is the line always exactly as late as persistence?
