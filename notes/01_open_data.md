# Notes to notebook 1: open weather data, and a first forecast

These notes hold the background and the derivations; the workshop runs the notebook only. The sections follow the notebook. Every code snippet runs in the notebook after its last cell.

## 1. One request

**ERA5** is a *reanalysis*: a weather model run over the past and corrected with millions of observations. It gives the weather for every hour since 1940 on a grid of 0.25° (about 25 km). Open-Meteo serves it through a free web API, so one HTTP request returns years of hourly data. `fetch_wind` in `windlab.py` builds the request and unpacks the answer; `models=era5` pins the ERA5 data, so everyone gets the same numbers.

**Why components, not speed and direction.** A direction is an angle on a circle: 359° and 1° are almost the same wind, but their average, 180°, is the opposite one. Averages, differences and squared errors work for numbers on a line, so we describe the wind as an arrow with an east part $u$ and a north part $v$.

**The signs, on two examples.** Wind from the north ($\varphi = 0^\circ$) at 5 m/s blows towards the south: $\sin 0^\circ = 0$ and $\cos 0^\circ = 1$ give $u = 0$ and $v = -5$ m/s. Wind from the west ($\varphi = 270^\circ$) at 5 m/s: $\sin 270^\circ = -1$ and $\cos 270^\circ = 0$ give $u = +5$ m/s, towards the east, and $v = 0$. The minus signs turn "from" into "towards". Back to the speed: $\text{speed} = \sqrt{u^2 + v^2}$, `np.hypot(u, v)` in the code. Notebook 1 forecasts the speed, which is easy to read; notebook 2 forecasts the components, which a network needs to tell a westerly from an easterly.

## 2. A first look

**Why midday is windiest.** The sun warms the ground, the air near the ground mixes with the faster air above, and the wind at 10 m picks up speed. At night the ground cools, the lowest air settles, and the 10 m wind slows down.

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

**How to read the formula.** The bar means "average over the training hours", and $x - \bar x$, $y - \bar y$ are the **deviations** from the average, now and $h$ hours later. The numerator, the **covariance**, is large when the two deviations tend to have the same sign: a wind stronger than usual now stays stronger than usual later. The denominator, the **variance**, is the typical squared size of the deviation now. Their ratio $w$ is the share of the present deviation still there $h$ hours later: $w = 1$ keeps all of it (persistence), $w = 0$ forgets all of it (climatology). Then $b$ makes the line pass through the average point $(\bar x, \bar y)$.

**Deriving least squares.** At the lowest point of the loss its slope is zero in every direction: $\partial L/\partial b = 0$ and $\partial L/\partial w = 0$ ($\partial$ reads "the slope of $L$ when only this one parameter changes").

*Step 1, the bias.* By the chain rule, the slope of a square is twice the thing inside the square, times the slope of that thing: $\frac{\partial}{\partial b}(w x + b - y)^2 = 2\,(w x + b - y)$. Averaging over the hours,

$$\frac{\partial L}{\partial b} = \frac2N\sum\left(w x + b - y\right) = 2\left(w\bar x + b - \bar y\right) = 0 \quad\Longrightarrow\quad b = \bar y - w\,\bar x.$$

*Step 2, the weight.* With this $b$ the line is $\hat y = \bar y + w\,(x - \bar x)$, and its error is $\hat y - y = w\,(x - \bar x) - (y - \bar y)$. The loss now depends on $w$ alone:

$$\frac{dL}{dw} = \frac2N\sum\Big(w\,(x - \bar x) - (y - \bar y)\Big)(x - \bar x) = 2\Big(w\;\overline{(x - \bar x)^2} - \overline{(x - \bar x)(y - \bar y)}\Big) = 0$$

gives the formula for $w$. The function `best_line` is exactly these two results.

**Why the line flattens.** Over a year, $x$ and $y$ are the same series shifted by $h$ hours, so they have nearly the same spread, and $w$ is then the **correlation** between the wind now and $h$ hours later. The further ahead, the weaker the link, so the forecast is pulled towards the average: *regression towards the mean*.

**Why it cannot lose to the baselines on the training data.** Persistence and climatology are lines too, $(w, b) = (1, 0)$ and $(0, \bar y)$. The best line has the smallest loss of all lines, so its loss is at most theirs. On new data, 2024, this is not guaranteed, but it holds when the years are alike, as the plot shows.

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

**The best line is not the best linear model.** It is the best line with *one input*, the speed now. Linear models with more inputs, fitted by the same least squares, can do better. Fitted on 2023 and scored on 2024, 6 hours ahead, for the wind speed at the grid point of notebooks 2 and 3 (notebook 1's numbers agree to rounding):

| Linear model | Inputs | RMSE (m/s) | Skill | Best match with the truth |
| --- | --- | --- | --- | --- |
| persistence | – | 1.417 | 0 | 6 h earlier |
| best line: the speed now | 1 | 1.303 | +0.154 | 6 h earlier |
| the speed, the last 6 hours | 6 | 1.293 | +0.167 | 6 h earlier |
| the speed, u and v, the last 6 hours | 18 | 1.303 | +0.154 | 6 h earlier |
| the same, and the hour of the day | 30 | 1.260 | +0.209 | 6 h earlier |
| the 3 × 3 field (speed, u, v), the last 6 hours | 162 | 1.237 | +0.238 | 5 h earlier |
| the 9 × 9 field (speed, u, v), the last 3 hours | 729 | 1.200 | +0.282 | 5 h earlier |

A line fitted on 2024 itself, which is cheating, gains only 0.006 m/s (1.297 instead of 1.303): within its own family, the best line is as good as it gets. More history at one point barely helps. The hour of the day does, because the speed has a daily cycle. And only the wind around the point, which arrives later, makes the forecast earlier.

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
```

At this grid point the daily cycle is weak, so one average for each hour of the day gains little over one average for all hours: 1.743 instead of 1.768 m/s. But the anomaly's autocorrelation keeps falling where the speed's rises again: 0.158, 0.133 and 0.128 at 36, 42 and 48 hours, against 0.132, 0.128 and 0.153. And 6 hours ahead, the hour's average plus a line on the anomaly scores 1.249 m/s: better than the best line (1.303), and better than every single-point model in the table above, with 2 parameters and 24 averages. A climatology for what the clock foresees and a fading memory for the rest: this split is a common baseline in forecasting, known as damped persistence of the anomaly.

## More exercises

1. In the week plot, set `h = 1` or `h = 24`. How do the three forecasts look then?
2. Run the timing check above for $h = 1$ and $h = 24$ (rerun the week cell first). Is the line always exactly as late as persistence?
