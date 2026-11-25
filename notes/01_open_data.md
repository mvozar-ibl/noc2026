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

**The best constant forecast is the average.** Which single number $c$ is the best forecast, if it may not depend on anything? Its loss, the mean squared error, is $L(c) = \frac1N\sum (c - y)^2$, a parabola in $c$. At its lowest point the slope is zero:

$$\frac{dL}{dc} = \frac2N\sum (c - y) = 2\,(c - \bar y) = 0 \quad\Longrightarrow\quad c = \bar y.$$

So climatology is the best constant forecast on the training hours. Section 4 repeats this argument for a line.

**Why split by time.** Neighbouring hours are almost equal: one hour ahead, persistence misses by only 0.5 m/s. With a random split, almost every test hour would have its neighbours in the training data, and a model could score well just by remembering them. A split by time tests what we want: forecasting a future the model has not seen.

## 4. The best straight line

**How to read the formula.** The bar means "average over the training hours", and $x - \bar x$, $y - \bar y$ are the **deviations** from the average, now and $h$ hours later. The numerator, the **covariance**, is large when the two deviations tend to have the same sign: a wind stronger than usual now stays stronger than usual later. The denominator, the **variance**, is the typical squared size of the deviation now. Their ratio $w$ is the share of the present deviation still there $h$ hours later: $w = 1$ keeps all of it (persistence), $w = 0$ forgets all of it (climatology). Then $b$ makes the line pass through the average point $(\bar x, \bar y)$.

**Deriving least squares.** At the lowest point of the loss its slope is zero in every direction: $\partial L/\partial b = 0$ and $\partial L/\partial w = 0$ ($\partial$ reads "the slope of $L$ when only this one parameter changes").

*Step 1, the bias.* By the chain rule, the slope of a square is twice the thing squared times the slope of that thing: $\frac{\partial}{\partial b}(w x + b - y)^2 = 2\,(w x + b - y)$. Averaging over the hours,

$$\frac{\partial L}{\partial b} = \frac2N\sum\left(w x + b - y\right) = 2\left(w\bar x + b - \bar y\right) = 0 \quad\Longrightarrow\quad b = \bar y - w\,\bar x.$$

*Step 2, the weight.* With this $b$ the line is $\hat y = \bar y + w\,(x - \bar x)$, and its error is $\hat y - y = w\,(x - \bar x) - (y - \bar y)$. The loss now depends on $w$ alone:

$$\frac{dL}{dw} = \frac2N\sum\Big(w\,(x - \bar x) - (y - \bar y)\Big)(x - \bar x) = 2\Big(w\;\overline{(x - \bar x)^2} - \overline{(x - \bar x)(y - \bar y)}\Big) = 0$$

gives the formula for $w$. The function `best_line` is exactly these two results.

**Why the line flattens.** The wind varies about as much at every hour of the year, so $x$ and $y$ have nearly the same spread, and $w$ is then the **correlation** between the wind now and $h$ hours later. The further ahead, the weaker the link, so the forecast is pulled towards the average: *regression towards the mean*.

**Why it cannot lose to the baselines on the training data.** Persistence and climatology are lines too, $(w, b) = (1, 0)$ and $(0, \bar y)$. The best line has the smallest loss of all lines, so its loss is at most theirs. On new data, 2024, this is not guaranteed, but it holds when the years are alike, as the plot shows.

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
| best line: the speed now | 1 | 1.303 | +0.080 | 6 h earlier |
| the speed, the last 6 hours | 6 | 1.293 | +0.087 | 6 h earlier |
| the speed, u and v, the last 6 hours | 18 | 1.303 | +0.080 | 6 h earlier |
| the same, and the hour of the day | 30 | 1.260 | +0.111 | 6 h earlier |
| the 3 × 3 field (speed, u, v), the last 6 hours | 162 | 1.237 | +0.127 | 5 h earlier |
| the 9 × 9 field (speed, u, v), the last 3 hours | 729 | 1.200 | +0.153 | 5 h earlier |

A line fitted on 2024 itself, which is cheating, gains only 0.006 m/s (1.297 instead of 1.303): within its own family, the best line is as good as it gets. More history at one point barely helps. The hour of the day does, because the speed has a daily cycle. And only the wind around the point, which arrives later, makes the forecast earlier.

## More exercises

1. In the week plot, set `h = 1` or `h = 24`. How do the three forecasts look then?
2. Run the timing check above for $h = 1$ and $h = 24$ (rerun the week cell first). Is the line always exactly as late as persistence?
