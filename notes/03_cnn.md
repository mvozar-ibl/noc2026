# Notes to notebook 3: a small CNN on a wind field

These notes hold the background; the notebook runs on its own. The sections follow the notebook, and the code snippet runs in the notebook after its last cell.

## 1. The wind field as an image

A photo is a grid of pixels with three channels, red, green and blue. The wind on a 9 × 9 grid is a grid of points with two channels, u and v, and with three hours stacked, six channels. Everything a convolutional network does to photos, it can do to such a field. The grid spans 2° × 2° around Bratislava: about 150 km east–west and 220 km north–south.

## 3. A convolutional block

**What the formula says.** $x_{c,i,j}$ is input channel $c$ at grid point $(i, j)$, and $p, q$ run over the 3 × 3 neighbourhood. Each output channel $c'$ has its own stencil of weights $w_{c',c,p,q}$, here $6 \times 3 \times 3 = 54$ of them in the first convolution, and one bias $b_{c'}$. A stencil with positive weights on the west side and negative ones on the east side, for example, responds to "wind in the west minus wind in the east", a hint of air piling up.

**The drawing in the notebook** is one such stencil on one channel, with fixed weights: "wind in the west minus wind in the east" on u, divided by 3 so that it is the average over the three rows. Its output is proportional to the convergence of the wind from west to east, $-\partial u/\partial x$: positive where air piles up. The network starts from random stencils and learns its own by gradient descent; nobody tells it to look for convergence.

**Pictures and animations.** V. Dumoulin and F. Visin, *A guide to convolution arithmetic for deep learning* ([arXiv:1603.07285](https://arxiv.org/abs/1603.07285)), with animations of stencils sliding over a map, with and without padding, at [github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic). The Stanford CS231n notes on [convolutional networks](https://cs231n.github.io/convolutional-networks/) include an interactive convolution demo.

**Weight sharing.** The same stencil is used at all 81 grid points, so a pattern is recognised wherever it appears, and the number of parameters does not grow with the size of the map.

**Counting the parameters.**

- First convolution: 16 output channels × (6 × 9 weights + 1 bias) = 880.
- Second convolution: 16 × (16 × 9 + 1) = 2,320.
- The final linear layer reads 16 channels × 81 points = 1,296 numbers for each of the 2 outputs: 2 × 1,296 + 2 = 2,594.
- Together 5,794. The MLP of notebook 2 on the same field needs 35,458, mostly in its first layer, 486 × 64 + 64 = 31,168: six times more, because it learns a separate weight for every point.

**What one output sees.** After one 3 × 3 convolution a grid point has seen its 3 × 3 neighbourhood; after two, a 5 × 5 neighbourhood, ±0.5° or about 40–55 km. The final linear layer then combines all 81 points, so the forecast for the centre can use the whole field. `padding=1` adds a frame of zeros around the map, so that the edge points have neighbours and the map keeps its 9 × 9 size.

## 4. Forecasts against the truth

**Why the whole field helps, and why only a little earlier.** Weather systems move with the winds a few kilometres up, typically 30–50 km per hour, so a change that reaches the centre in 6 hours starts 200–300 km upstream. The window reaches only 74 km to the west and east of the centre, and 111 km to the north and south: it sees what arrives within about 1.5 to 4 hours. The field therefore holds part of the centre point's future, which no single-point model can see, and it is the only input that makes the forecasts earlier, not just smaller in error; but only by about an hour, 5 hours late instead of 6 (see the notes to notebook 1). Changes that start outside the window remain invisible to every model here. A larger window, or winds from higher levels, would be the next step.

**The linear models.** "Linear, whole field" is least squares on all 486 numbers of the three stacked fields, 974 weights and biases, fitted with the same formula as in notebook 2. It already gets most of the gain (skill +0.377, against +0.164 for the centre point alone). The CNN adds a little on top (+0.391), while the MLP on the flat field does slightly worse than the linear model (+0.368) with 35,458 parameters.

## More exercises

1. Train on January and February 2023 only: add `train &= time[t].astype("datetime64[M]").astype(int) % 12 < 2` after the samples cell. Which model copes best with little data?
2. Count the parameters of a CNN with `channels=8` by hand, then check it with the notebook.

The parameters of both networks, as printed by the notebook:

```python
for name, model in [("MLP", mlp_model), ("CNN", cnn_model)]:
    print(name, [p.numel() for p in model.parameters()])
```
