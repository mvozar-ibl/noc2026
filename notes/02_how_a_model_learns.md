# Notes to notebook 2: how a model learns

These notes hold the background and the derivations; the workshop runs the notebook only. The sections follow the notebook. Every code snippet runs in the notebook after its last cell; the numbers quoted are what they print.

## Words

The names of notebook 1 stay (see its notes): input $x$, target $y$, forecast $\hat y$, error $\hat y - y$, and the line $\hat y = w\,x + b$ with weight $w$ and bias $b$.

| Word | Symbol | In the code | Meaning |
| --- | --- | --- | --- |
| parameters | $\theta$ ("theta") | `theta` | all weights and biases of a model together |
| loss | $L$ | `loss` | the mean squared error on the training hours: smaller is better |
| gradient | $\partial L/\partial w$, $\partial L/\partial b$ | `dw`, `db` | for each parameter, how fast the loss grows when that parameter grows |
| learning rate | $\eta$ ("eta") | `lr` | the size of one step against the gradient |
| curvature | $\kappa$ ("kappa") | `kappa` | how sharply the loss bends along one direction: the second derivative along it |
| Hessian | $H$ | `H` | the matrix of all second derivatives of the loss; its eigenvalues are the curvatures $\kappa$ along the bowl's main axes |
| batch | | `batch_size` | the samples used for one step |
| epoch | | `epochs` | one pass over all training samples |

## Why u, and not the speed of notebook 1

Notebook 1 forecast the wind speed. This notebook forecasts the components u and v, for three reasons. The networks of section 5 need both components, to tell a westerly from an easterly. The speed is never negative, so its average is far from zero, and its loss valley is long and thin: gradient descent with the same learning rate is still off in the third decimal after 200 steps. And the speed has a daily cycle, so its memory does not fade along one exponential: the weight drops from 24 to 36 hours and rises again at 48, when the target is once more the same hour of the day as now. A check, with the code of sections 1 and 4:

```python
speed_ba = np.hypot(u[:, centre], v[:, centre]).astype(float)
xs, ys = speed_ba[hours], speed_ba[hours + 1]
kappa_s = np.linalg.eigvalsh(2 * np.array([[np.mean(xs * xs), np.mean(xs)], [np.mean(xs), 1.0]]))
w_s, b_s = 0.0, 0.0
for _ in range(200):
    dw, db = gradient(w_s, b_s, xs, ys)
    w_s, b_s = w_s - lr * dw, b_s - lr * db
w_hs = np.array([best_line(speed_ba[train_hours], speed_ba[train_hours + h])[0] for h in horizons])
print(f"speed: curvatures {kappa_s[0]:.2f} and {kappa_s[1]:.2f}; after 200 steps w = {w_s:.4f}, b = {b_s:.4f}; "
      f"least squares {best_line(xs, ys)[0]:.4f}, {best_line(xs, ys)[1]:.4f}; weights at 12, 24, 36, 48 h: {np.round(w_hs[[11, 23, 35, 47]], 2)}")
```

For the speed the curvatures are 0.41 and 27.96 (for u: 1.85 and 11.72, see section 1), gradient descent ends at $w = 0.9547$, $b = 0.1455$ instead of least squares' 0.9550 and 0.1442, and the weights at 12, 24, 36 and 48 hours are 0.34, 0.23, 0.06 and 0.15.

## 1. Gradient descent

**Where the gradient comes from.** For one hour the squared error is $(\hat y - y)^2$ with $\hat y - y = w x + b - y$. By the chain rule, the slope of a square is twice the thing inside the square, times the slope of that thing, and the slope of $w x + b - y$ is $x$ when $w$ changes and 1 when $b$ changes:

$$\frac{\partial}{\partial w}\left(\hat y - y\right)^2 = 2\left(\hat y - y\right)x, \qquad \frac{\partial}{\partial b}\left(\hat y - y\right)^2 = 2\left(\hat y - y\right).$$

The loss is the average over the $N$ hours, and the slope of an average is the average of the slopes, which gives the formulas of the notebook; `gradient` computes them. Read $\partial L/\partial b$ as twice the average error: forecasts that are too high on average make it positive, and the step lowers $b$. $\partial L/\partial w$ weights each error by $x$: if the forecast is too high whenever $x$ is large, $w$ is too big.

**Why the step goes downhill.** For a small change of the parameters the loss changes by about $\Delta L \approx \frac{\partial L}{\partial w}\Delta w + \frac{\partial L}{\partial b}\Delta b$. The step $\Delta w = -\eta\,\partial L/\partial w$, $\Delta b = -\eta\,\partial L/\partial b$ gives $\Delta L \approx -\eta\left[(\partial L/\partial w)^2 + (\partial L/\partial b)^2\right]$, which is negative as long as the step is small enough for the "about" to hold.

**How large a step may be.** Take a loss that is a parabola in one parameter, $L = \tfrac\kappa2(\theta - \theta^*)^2$, with its lowest point at $\theta^*$ and curvature $\kappa$. Its slope is $\kappa(\theta - \theta^*)$, so one step gives

$$\theta - \theta^* \;\leftarrow\; (1 - \eta\kappa)\,(\theta - \theta^*).$$

The distance to the bottom shrinks only if $|1 - \eta\kappa| < 1$, that is $\eta < 2/\kappa$; a larger step jumps over the valley, lands higher on the other side, and the walk blows up. Our loss is such a bowl in two dimensions.

**From $\kappa$ to the Hessian.** With two parameters there are four second derivatives, $\partial^2 L/\partial w^2$, $\partial^2 L/\partial w\,\partial b$, $\partial^2 L/\partial b\,\partial w$ and $\partial^2 L/\partial b^2$. Differentiating the gradient of the notebook once more gives them, and they form the **Hessian** $H = 2\begin{pmatrix}\overline{x^2} & \bar x \cr \bar x & 1\end{pmatrix}$. Its diagonal holds the curvature along $w$ alone and along $b$ alone. Along any direction, a unit vector $d$, the loss is a one-parameter bowl with curvature $d^\top H\,d$; this is largest and smallest along the two eigenvectors of $H$, the main axes of the bowl, where it equals the eigenvalues. Those eigenvalues are the curvatures $\kappa$ of the steepest and the flattest direction:

```python
kappa = np.linalg.eigvalsh(H)  # the two curvatures, flattest first (H from the notebook)
print(f"curvatures {kappa[0]:.2f} and {kappa[1]:.2f}: stable for lr < {2 / kappa[1]:.3f}; "
      f"with lr = {lr} the flat direction shrinks by {1 - lr * kappa[0]:.3f} per step")
```

The diagonal of $H$ is 11.57 and 2: u is several metres per second, so $\overline{x^2}$ is several times larger than 1, and that is what makes the bowl long and thin. The curvatures are 1.85 and 11.72, so gradient descent is stable for $\eta < 0.171$: `lr = 0.2` blows up. With $\eta = 0.05$ the distance along the flat direction shrinks by the factor 0.907 per step, which is why the walk creeps at the end.

**One step: Newton, and a rescaled input.**

- *Newton lands in one step.* The loss of a line is exactly a quadratic bowl around its lowest point $\theta^*$, $L(\theta) = L(\theta^*) + \tfrac12(\theta - \theta^*)^\top H\,(\theta - \theta^*)$, with $H$ the Hessian above. Its gradient is therefore $\nabla L(\theta) = H\,(\theta - \theta^*)$, so $\theta - H^{-1}\nabla L(\theta) = \theta^*$ from any start. This is the normal equations of section 6, solved in one step.
- *A rescaled input needs only a plain step.* For $\tilde x = (x - \bar x)/\sigma_x$, the average of $\tilde x$ is 0 and that of $\tilde x^2$ is 1, so $H = 2\begin{pmatrix}1 & 0 \cr 0 & 1\end{pmatrix}$: both curvatures are 2, and $\eta = 1/2$ makes both factors $1 - \eta\kappa$ zero.
- *Back to the original units.* From $\hat y = \tilde w\,\tilde x + \tilde b$ follow $w = \tilde w/\sigma_x$ and $b = \tilde b - \tilde w\,\bar x/\sigma_x$.

On the raw bowl, no single step size does both directions at once:

```python
kappa = np.linalg.eigvalsh(H)
for eta in (1 / kappa[1], 1 / kappa[0]):
    print(f"lr = {eta:.3f}: per step the flat direction keeps {1 - eta * kappa[0]:+.2f}, the steep one {1 - eta * kappa[1]:+.2f}")
```

With $\eta = 1/11.72 = 0.085$ the steep direction is done in one step, but the flat one keeps 84 % of its distance per step. With $\eta = 1/1.85 = 0.540$ the flat one is done, but the steep one is multiplied by $-5.32$ per step and blows up.

**First order and second order.** Gradient descent, SGD and AdamW use only the first derivatives of the loss, one slope per parameter: they are *first-order* methods. Newton's step also needs all the second derivatives, the matrix $H$: it is a *second-order* method. That is what makes it exact in one step on a bowl, and what makes it rarely feasible:

- *Memory.* With $n$ parameters, $H$ has $n^2$ entries. For the 5,122 parameters of the network in section 5 that is 26 million numbers, 210 MB, which a laptop still holds; for a million parameters it is 8 TB, and for a billion $8 \times 10^{18}$ bytes.
- *Time.* $H$ must first be computed: one row of second derivatives per parameter, each row at least as costly as a gradient over all the data. For our network that is at least 5,122 passes over the data, more than 500 times its whole training of 10 epochs. Solving $H\,\Delta\theta = \nabla L$ then takes about $n^3/3$ operations, $4.5 \times 10^{10}$; on the test laptop that alone takes about as long as the whole training (the check below). And $H$ changes after every step, so all this would be repeated at each one.
- *Shape.* The loss of a line is one clean bowl, but the loss of a network is not: it has saddles and ridges, where some curvatures are negative, and there the Newton step can point uphill.
- *Noise.* With batches, $H$ would be estimated from 64 or 1,024 samples, and inverting a noisy matrix amplifies the noise.

The time of one solve, on a matrix of the size of $H$ for our network, against the whole training of section 5:

```python
import time as clock  # "time" already holds the hours of the data
H_network = np.eye(5122) + 0.001  # a matrix of the size of H for the network of section 5
start = clock.perf_counter()
np.linalg.solve(H_network, np.ones(5122))
solve_seconds = clock.perf_counter() - start
start = clock.perf_counter()
train_mlp(Y_ba[train] - now_ba[train])
print(f"H: {H_network.nbytes / 1e6:.0f} MB; one solve {solve_seconds:.1f} s; "
      f"the whole training of section 5 {clock.perf_counter() - start:.1f} s")
```

On the test laptop one solve and the whole training each take under a second: a single Newton step costs about as much as all 350 steps of AdamW even before $H$ is computed.

Where $H$ is cheap, it is used. For a linear model the loss is one fixed bowl, the Newton step is the normal equations (section 6), and `least_squares` solves them directly: for the 2 × 13 parameters of the linear forecast in section 6, and the 2 × 487 of notebook 3. So in practice networks are trained by first-order methods, which borrow the idea cheaply. Rescaling the inputs evens out their sizes before training starts (section 5, which also shows its limit). AdamW's step size per parameter is a rough relative of Newton's: it rescales each parameter's step by the typical size of its own gradient, without ever computing $H$. Quasi-Newton methods such as L-BFGS build a small approximation of $H^{-1}$ from the last few gradients; they work well for smooth problems with full batches, less so with the noisy batches of SGD.

## 2. SGD and AdamW

**Why SGD: data that do not fit in memory.** A gradient-descent step needs the gradient over all the training data. Our year, 8,760 numbers, fits in memory easily, but real training sets often do not: decades of weather fields, millions of images. They stay on disk and are read in pieces. Gradient descent is still possible then, by adding up the gradients piece by piece, but it takes only one step per pass over the data. SGD steps after every piece, a random batch, so one pass gives many steps. A network adds a second memory cost: backpropagation keeps the intermediate results of every sample in the step (section 3), so the batch size, not the size of the data, sets the memory a step needs.

**Why a small batch is enough.** Samples repeat each other, so a small random batch points almost the same way as all the data:

```python
full = np.array(gradient(0.0, 0.0, x, y))
agreement = [np.dot(g, full) / (np.linalg.norm(g) * np.linalg.norm(full))
             for g in (np.array(gradient(0.0, 0.0, x[i], y[i])) for i in batches)]
print(f"agreement of a 64-hour batch gradient with the full one at the start: {min(agreement):.3f} or more")
```

The agreement (the cosine of the angle between the two directions; 1 means the same direction) is 0.991 or more for every one of the 200 batches. Averaged over all possible batches, the batch gradient equals the full gradient; a single one is off by a random amount that shrinks like $1/\sqrt{\text{batch size}}$. Near the bottom the full gradient is zero but the batch gradients are not, so SGD keeps jittering; a smaller learning rate makes the jitter smaller.

**What it saves.** The data read until the error on the new year is within 1 % of least squares:

```python
x_new, y_new = data["2024 (new year)"]
target = 1.01 * rmse(w_best * x_new + b_best, y_new)
for name, path in paths.items():
    first = next((k for k, (w_k, b_k) in enumerate(path[1:], 1) if rmse(w_k * x_new + b_k, y_new) <= target), None)
    per_step = len(x) if name == "gradient descent" else 64
    print(f"{name}: within 1 % after {first} steps = {first * per_step:,} hours read")
```

Gradient descent needs 7 steps, 61,320 hours read; SGD 12 steps, 768 hours; AdamW 79 steps, 5,056 hours (it moves each parameter by only about $\eta$ per step, see below).

**SGD on new data.** In the notebook's table, SGD's end point is a little worse on the new year but a little better in Košice. One run is one random sample of its jitter, so repeat it with 20 batch orders:

```python
ends = []
for seed in range(20):
    rng_seed = np.random.default_rng(seed)
    w_s, b_s = 0.0, 0.0
    for _ in range(200):
        i = rng_seed.integers(0, len(x), size=64)
        dw, db = gradient(w_s, b_s, x[i], y[i])
        w_s, b_s = w_s - lr * dw, b_s - lr * db
    ends.append([rmse(w_s * xs + b_s, ys) for xs, ys in data.values()])
print("SGD, 20 batch orders:", "  ".join(f"{name} {m:.3f} ± {s:.3f}" for name, m, s in zip(data, np.mean(ends, 0), np.std(ends, 0))))
```

On average SGD scores 0.533 ± 0.003 on 2023, 0.500 ± 0.004 on 2024 and 0.396 ± 0.003 in Košice, against 0.530, 0.497 and 0.395 for gradient descent: slightly worse everywhere, so the gain in Košice was luck. SGD's end point jitters around the bottom and costs about 0.003 m/s on every data set, whichever batches it draws. With 2 parameters and 8,760 hours there is nothing to overfit: for this line, the optimizer changes the cost, not what is learned.

**And for the network?** The same comparison for the delta network of section 5, trained with full batches (gradient descent) for the same 350 steps as the notebook's batches of 1,024, three random starts each:

```python
import time as clock  # "time" already holds the hours of the data
for name, epochs, size in [("batches of 1,024", 10, 1024), ("full batches", 350, int(train.sum()))]:
    scores, start = [], clock.perf_counter()
    for seed in range(3):
        model, _ = train_mlp(Y_ba[train] - now_ba[train], epochs=epochs, batch_size=size, seed=seed)
        scores.append((rmse(now_ba[test] + predict(model, X_ba[test]), Y_ba[test]), rmse(now_ke + predict(model, X_ke), Y_ke)))
    scores = np.array(scores)
    print(f"{name}: Bratislava 2024 {scores[:, 0].min():.3f}-{scores[:, 0].max():.3f}, "
          f"Kosice {scores[:, 1].min():.3f}-{scores[:, 1].max():.3f}; {(clock.perf_counter() - start) / 3:.1f} s per run")
```

With batches of 1,024 the three runs score 1.492–1.493 in Bratislava 2024 and 1.209–1.216 in Košice; with full batches 1.481–1.484 and 1.217–1.224, at about ten times the time per run on the test laptop. For every random start, the full batches fit Bratislava a little closer and do a little worse in Košice: for a network, the optimizer can change what is learned, if only slightly.

**AdamW, line by line.**

- *Momentum.* Unrolled, $m_k = (1-\beta_1)\left(g_k + \beta_1 g_{k-1} + \beta_1^2 g_{k-2} + \dots\right)$: recent gradients count most, roughly the last $1/(1-\beta_1) = 10$. Random errors partly cancel, while a steady direction adds up.
- *The typical size.* $s$ is the same kind of average of $g^2$, over roughly the last $1/(1-\beta_2) = 1{,}000$ steps, so $\sqrt{s}$ is the typical size of each parameter's gradient.
- *The correction of the start.* $m$ and $s$ start at 0. If every gradient were the same $g$, the geometric series gives $m_k = (1-\beta_1)(1 + \beta_1 + \dots + \beta_1^{k-1})\,g = (1 - \beta_1^k)\,g$, so dividing $m$ by $1 - \beta_1^k$ returns exactly $g$; likewise for $s$ with $\beta_2$.
- *The step.* If a parameter's gradient keeps its sign, the corrected $m$ is about $g$ and the corrected $\sqrt{s}$ about $|g|$, so the parameter moves by about $\eta$ per step, however large or small its gradient is; $\epsilon = 10^{-8}$ only prevents a division by zero. That pays off in networks, where the gradients of different weights differ by orders of magnitude. On our line it does not: $w$ has to travel 0.97 and $b$ only 0.016, but both move by about $\eta$ per step, so $b$ overshoots and needs many steps to settle. That is why AdamW is the slowest of the three per step in the notebook; per hour of data read it still beats gradient descent.
- *Weight decay.* $-\eta\lambda\,\theta$ shrinks every parameter by the fraction $\eta\lambda$ per step, keeping weights small unless the data need them. A network with many weights can use large ones to follow the noise of its training data (**overfitting**), and the pull towards zero makes that costly. On the line, with 2 parameters, it only shifts $w$ and $b$ a little. It is "decoupled", the W in AdamW: applied directly to $\theta$, not through the gradient, so the division by $\sqrt{s}$ does not weaken it.

**Is AdamW worse, then?** Not where it is meant to work. Its step does not depend on the size of the gradient, so one learning rate suits problems of any scale, while SGD's learning rate must be tuned to each problem: on the line, `lr = 0.2` blows SGD up (the limit is 0.171, section 1) but not AdamW (exercise 3 of the notebook). The networks of section 5 show the same thing. Trained with plain SGD, they match AdamW only after the learning rate has been tuned:

```python
def train_sgd(target, lr, epochs=10, batch_size=1024, seed=0):
    # As train_mlp, with plain SGD instead of AdamW
    torch.manual_seed(seed)
    model = mlp(X_ba[train], n_out=2)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    fit(model, X_ba[train], target, opt, epochs=epochs, batch_size=batch_size,
        scheduler=warmup_cosine(opt, epochs * math.ceil(train.sum() / batch_size)), seed=seed)
    return model

def skill(forecast):  # against persistence, Bratislava 2024
    return 1 - (rmse(forecast, Y_ba[test]) / rmse(now_ba[test], Y_ba[test])) ** 2

for lr in (0.01, 0.1, 0.3):
    absolute_sgd, delta_sgd = train_sgd(Y_ba[train], lr), train_sgd(Y_ba[train] - now_ba[train], lr)
    print(f"SGD, lr = {lr}: skill absolute {skill(predict(absolute_sgd, X_ba[test])):+.3f}, "
          f"delta {skill(now_ba[test] + predict(delta_sgd, X_ba[test])):+.3f}")
print(f"AdamW, lr = 0.002 (the notebook): skill absolute {skill(predict(absolute, X_ba[test])):+.3f}, "
      f"delta {skill(now_ba[test] + predict(delta, X_ba[test])):+.3f}")
```

With `lr = 0.01`, SGD's absolute network does not even reach persistence (skill −0.030); with `lr = 0.3` it scores +0.198 and +0.192, as good as AdamW's +0.179 and +0.191 at the notebook's learning rate. Tuned, plain SGD does as well on this small network. AdamW's advantage is that it needs no such search, and that is what makes it the usual choice for networks.

## 3. Autodiff

Autodiff applies the chain rule mechanically. Take one hour with $x = 2$ and $y = 3$, starting from $w = b = 0$. **Forward**, recording every intermediate result: $\hat y = w\,x + b = 0$, the error $\hat y - y = -3$, the loss $(\hat y - y)^2 = 9$. **Backward**, from the end, multiplying the slopes of the recorded steps: the loss changes by $2 \cdot (-3) = -6$ per unit of error, the error by 1 per unit of $\hat y$, and $\hat y$ by $x = 2$ per unit of $w$ and by 1 per unit of $b$:

$$\frac{\partial L}{\partial w} = -6 \cdot 1 \cdot 2 = -12, \qquad \frac{\partial L}{\partial b} = -6 \cdot 1 \cdot 1 = -6.$$

`backward()` does exactly this for any formula built from PyTorch operations, with any number of parameters, at roughly the cost of one more forward pass:

```python
w_one, b_one = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
((w_one * 2.0 + b_one - 3.0) ** 2).backward()
print(f"dL/dw = {w_one.grad.item():.0f}, dL/db = {b_one.grad.item():.0f}")
```

## 4. The memory curve

**Why an exponential.** If every hour keeps the same fraction $r$ of the deviation from the average, after $h$ hours $r^h$ is left, and $r = e^{-1/\tau}$ turns this into $e^{-h/\tau}$. After $\tau$ hours, $e^{-1} \approx 37\,\%$ is left.

**The slope.** With $c_h = e^{-h/\tau}$, the chain rule gives

$$\frac{dL}{d\tau} = \frac1{48}\sum_h 2\left(c_h - w_h\right)\frac{dc_h}{d\tau}, \qquad \frac{dc_h}{d\tau} = e^{-h/\tau}\cdot\frac{d}{d\tau}\left(-\frac h\tau\right) = e^{-h/\tau}\,\frac{h}{\tau^2},$$

which `curve_gradient` computes.

**Why there is no formula.** For the line, $w$ and $b$ appeared only multiplied by data and added, so the zero-slope conditions were linear equations, which algebra solves. Here $\tau$ sits inside 48 exponentials, and $dL/d\tau = 0$ can only be solved numerically.

Logarithms seem to give a formula after all. All 48 weights are positive, and if $w_h = e^{-h/\tau}$, then $\ln w_h = -h/\tau$: a line through the origin, whose least-squares slope gives $1/\tau = -\sum h \ln w_h / \sum h^2$. But that answers a different question. It minimizes the errors in $\ln w$, not in $w$, and since $\ln w$ changes by $\Delta w / w$, it weighs the small weights of the long horizons far more than our loss does:

```python
print(f"smallest weight {w_h.min():.3f}; tau from the logarithms {-np.sum(horizons**2) / np.sum(horizons * np.log(w_h)):.2f} h, "
      f"gradient descent on L {tau:.2f} h")
```

The logarithms give 22.15 hours, gradient descent on our loss 21.29 hours. Which is right depends on which errors matter; for the loss we chose, there is no formula.

**Why a learning rate of 500, and why it is safe.** The right step depends on how sharply the loss bends in the units of the parameter: near the bottom, the curvature in $\tau$ is about $4 \times 10^{-4}$ per hour² (the check below), against 1.85 to 11.72 for the line of section 1, so the stable learning rate is thousands of times larger. Near the bottom the loss curve is a parabola, so the rule of section 1 applies, $\eta < 2/\kappa$:

```python
kappa_tau = (curve_loss(tau + 1e-3) - 2 * curve_loss(tau) + curve_loss(tau - 1e-3)) / 1e-3**2
print(f"curvature {kappa_tau:.1e} per hour²: stable for lr_tau < {2 / kappa_tau:.0f}; kept per hour {np.exp(-1 / tau):.3f}, measured {w_h[0]:.3f}")
```

The curvature is $4.0 \times 10^{-4}$ per hour², so the walk is stable for a learning rate below about 5,000. The fit keeps 0.954 of the deviation per hour, while the measured one-hour weight is 0.974: over the first hours the wind is smoother than this simple law says, as the first points of the plot show.

**One number instead of 48 lines.** With the weight $w = e^{-h/\tau}$ and notebook 1's bias $b = \bar y - w\,\bar x$, where $\bar x \approx \bar y$, one formula forecasts every horizon: $\hat y = \bar y + e^{-h/\tau}\,(x - \bar y)$, the present deviation from the average, faded by the memory.

```python
test_48 = np.flatnonzero(year(time[:-48]) == 2024)
y_bar = u_ba[train_hours].mean()
for h in (1, 6, 24):
    x_t, y_t = u_ba[test_48], u_ba[test_48 + h]
    w_l, b_l = best_line(u_ba[train_hours], u_ba[train_hours + h])
    print(f"{h:2d} h: persistence {rmse(x_t, y_t):.3f}, one line per h {rmse(w_l * x_t + b_l, y_t):.3f}, "
          f"memory curve {rmse(y_bar + np.exp(-h / tau) * (x_t - y_bar), y_t):.3f}")
```

On u in 2024 (RMSE in m/s), the one number does as well as a separate line for every horizon, except one hour ahead, where one $\tau$ cannot fit the first hours perfectly:

| Horizon | Persistence | One line per horizon | Memory curve |
| --- | --- | --- | --- |
| 1 h | 0.501 | 0.498 | 0.503 |
| 6 h | 1.568 | 1.497 | 1.497 |
| 24 h | 2.803 | 2.340 | 2.339 |

## 5. The network

**Counting the parameters.** Layer 1 has 64 lines through 12 inputs: $64 \times 12 + 64 = 832$. Layer 2: $64 \times 64 + 64 = 4{,}160$. The output layer: $2 \times 64 + 2 = 130$. Together 5,122.

**Why the bends.** Without them, lines of lines are one line: $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2)$, so the network could do no more than least squares. The bends let it treat a strong wind differently from a weak one, or a westerly differently from an easterly. GELU multiplies $z$ by the share of a standard bell curve that lies below $z$: almost 0 for $z < -2$, almost $z$ for $z > 2$, and smooth in between, so it has a slope everywhere.

**Why rescale the inputs, and where that stops helping.** Section 1 showed that the largest safe learning rate is set by the steepest curvature, which grows with the size of the inputs ($\overline{x^2}$); section 4 showed the opposite extreme. Rescaling every input to $(x - \bar x)/\sigma_x$, with $\sigma_x$ its standard deviation, removes the differences in size between the inputs, so one learning rate suits them better. `windlab.Standardize` does this inside the model, with the averages of the training data.

It makes the bowl round only for a single input, as in section 1. Our 12 inputs are six hours of u and v, nearly copies of each other, and for correlated inputs the bowl stays long and thin along the directions in which they differ. The curvatures of the loss of a linear model on these inputs:

```python
def curvature_ratio(X):
    A = np.c_[X, np.ones(len(X))]  # the inputs, and a column of ones for the bias
    kappa = np.linalg.eigvalsh(2 * A.T @ A / len(A))  # the curvatures, flattest first
    return kappa[-1] / kappa[0]

X_train = X_ba[train]
print(f"steepest / flattest curvature: the line of section 1 {curvature_ratio(x[:, None]):.1f}; 12 inputs, "
      f"raw {curvature_ratio(X_train):,.0f}, rescaled {curvature_ratio((X_train - X_train.mean(0)) / X_train.std(0)):,.0f}")
```

The line of section 1 has a ratio of 6.3. The 12 inputs have 1,762 raw and 1,847 rescaled: u and v are both in m/s with similar spreads, so rescaling changes almost nothing here, and the ratio is about 300 times that of the bowl gradient descent crept along in section 1. Rescaling matters when the inputs come in different units and sizes, a pressure in pascals next to a wind in m/s. To make the bowl of correlated inputs round, they would have to be decorrelated as well ("whitening"), with the matrix of their correlations: second-order information again.

**Why the delta form starts well.** At the start $f(x;\theta) \approx 0$, so $\hat y \approx$ the wind now: training starts from the loss of persistence and only has to learn corrections. The absolute network needs about one and a half epochs just to reach that loss (the plot of section 5); after that, the start no longer matters much on the training town. The safe start matters most when training is short, for example with little data.

**The schedule.** With batches of 1,024 hours, one epoch is 35 steps. For the first 5 % of the steps the learning rate rises linearly from almost 0, while AdamW's averages are still rough; then it falls as $\eta \cdot \tfrac12\left(1 + \cos \pi p\right)$, with $p$ going from 0 to 1, so the jitter of section 2 dies out at the end.

## 6. Forecasts against the truth

**Least squares with many inputs.** With $n$ inputs the linear forecast is $\hat y = w_1 x_1 + \dots + w_n x_n + b$. Write the training data as a table $X$ with one row per hour and one column per input, add a column of ones for the bias, and collect the parameters in $\theta = (w_1, \dots, w_n, b)$. All forecasts at once are $[X\;\mathbf 1]\,\theta$. The slope of the loss with respect to each parameter is twice the average of the errors times that parameter's column, as in section 1, and setting all $n + 1$ slopes to zero gives the **normal equations**

$$[X\;\mathbf 1]^\top [X\;\mathbf 1]\;\theta = [X\;\mathbf 1]^\top y.$$

For a single input they are the two equations of notebook 1. `windlab.least_squares` solves them with `np.linalg.lstsq` and returns the weights and the bias separately.

**The limits in the plots.**

- *Sudden changes stay late.* Every forecast turns after the truth has turned, and the error of every model grows with the size of the change (for persistence by definition: the plot groups the hours by the change, which is its error): six hours of wind at one place do not announce a front.
- *The network beats the line only slightly.* With these 12 inputs most of what is predictable is linear; more parameters cannot add information that the inputs do not contain.
- *A new town costs skill.* The models learned Bratislava's habits. With the notebook's 12 inputs, the plain linear model keeps the most skill in Košice, and the delta network more than the absolute one, for every random start below: a learned correction to persistence seems to carry over better than a forecast learned from scratch. Both are observations for these inputs, not rules. With the whole 3 × 3 field as input (`FIELD = True`, exercise 6), every model falls below persistence in Košice, the linear one furthest, and the delta form is no longer reliably better than the absolute one:

```python
for FIELD in (False, True):  # samples() reads FIELD
    X_ba, Y_ba, now_ba, t_ba = samples(u, v)
    X_ke, Y_ke, now_ke, _ = samples(u_ke, v_ke)
    W_lin, b_lin = least_squares(X_ba[train], Y_ba[train])
    print(f"FIELD = {FIELD}, {X_ba.shape[1]} inputs: Kosice skill linear "
          f"{1 - (rmse(X_ke @ W_lin + b_lin, Y_ke) / rmse(now_ke, Y_ke)) ** 2:+.3f}")
    for seed in range(3):
        absolute_s, _ = train_mlp(Y_ba[train], seed=seed)
        delta_s, _ = train_mlp(Y_ba[train] - now_ba[train], seed=seed)
        skills = [1 - (rmse(forecast, Y_ke) / rmse(now_ke, Y_ke)) ** 2
                  for forecast in (predict(absolute_s, X_ke), now_ke + predict(delta_s, X_ke))]
        print(f"FIELD = {FIELD}, seed {seed}: Kosice skill absolute {skills[0]:+.3f}, delta {skills[1]:+.3f}")
```

With the notebook's 12 inputs, the linear model scores +0.106 in Košice and the delta network beats the absolute one for all three starts (+0.073 to +0.083, against +0.044 to +0.054). With the whole field, all seven models fall below persistence there: the linear one to −0.504, the networks to between −0.055 and −0.129, and the delta form is worse in two of the three starts.

**Why the linear model falls furthest: correlated inputs.** The 108 inputs of the field are neighbouring points and neighbouring hours, nearly copies of each other, so their bowl is extremely long and thin (section 5). Least squares then finds large weights of opposite sign whose small differences fit Bratislava; in Košice those differences no longer fit. A penalty on large weights, $\lambda \sum w^2$ added to the loss (**ridge regression**, the linear model's weight decay, section 2), keeps the weights small:

```python
x_mean, x_std = X_ba[train].mean(0), X_ba[train].std(0)  # FIELD = True, from the loop above: 108 inputs

def ridge(penalty):
    # Least squares on the rescaled inputs, with penalty * (sum of squared weights) added to the loss; the bias is free
    A = np.c_[(X_ba[train] - x_mean) / x_std, np.ones(train.sum())]
    free_bias = np.diag(np.r_[np.ones(A.shape[1] - 1), 0.0])
    weights = np.linalg.solve(A.T @ A + penalty * len(A) * free_bias, A.T @ Y_ba[train])
    return weights, lambda X: np.c_[(X - x_mean) / x_std, np.ones(len(X))] @ weights

for penalty in (0.0, 0.001, 0.1):
    weights, forecast = ridge(penalty)
    print(f"lambda = {penalty}: sum of |weights| {np.abs(weights[:-1]).sum():.1f}; skill Bratislava 2024 "
          f"{1 - (rmse(forecast(X_ba[test]), Y_ba[test]) / rmse(now_ba[test], Y_ba[test])) ** 2:+.3f}, "
          f"Kosice {1 - (rmse(forecast(X_ke), Y_ke) / rmse(now_ke, Y_ke)) ** 2:+.3f}")
```

Without a penalty the weights add up to 156 (in the units of the rescaled inputs), and the model scores +0.284 in Bratislava but −0.504 in Košice. A tiny penalty, $\lambda = 0.001$, shrinks them to 56 and lifts Košice to −0.084, at almost no cost in Bratislava (+0.275); with $\lambda = 0.1$ they add up to 12, and Košice reaches +0.039, above persistence. A model that fits its training town a little less closely carries over better: that is the job of weight decay in AdamW (section 2).

## More exercises

1. Section 1: `lr = 0.2` and `lr = 0.01`. Compare with the limit of 0.171 above.
2. Section 2: replace `lr` by `lr * 0.98**k` in the SGD loop (with `for k, i in enumerate(batches)`): the step shrinks as training goes on. Does the jitter go away?
3. Section 4: start from `tau = 60.0`, then try `lr_tau = 6000`. Does gradient descent still find 21.3 hours? Compare with the limit above.
4. Section 5: train both networks with `epochs=20`. Does the absolute network catch up? What did the delta form buy, then?
5. Section 5: train with full batches, as above, and compare Bratislava and Košice.
6. **Zero-shot:** keep `HORIZON = 6`, train only on Bratislava, and raise the skill in Košice. With `FIELD = True` the networks get better in Bratislava but worse in Košice: why?
7. Section 2: set `lr = 0.2` (notebook exercise 3), then try AdamW with `lr = 0.01` and `lr = 0.5`. How far can AdamW's learning rate move before its walk gets worse, and SGD's?
