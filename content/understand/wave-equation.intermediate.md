## Overview

Ripples on a pond, sound in a room and light passing through two slits all follow the same law, the
**wave equation**. This example solves it on a 512 × 384 grid and draws the result as a lit water
surface. You can drop ripples, draw walls, and load a double slit or an elliptical mirror. It also shows
one of the most important limits in computer simulation: take steps that are too big, and the answer
does not just get worse, it explodes.

## The idea

**The equation.** Let $u$ be the height of the water at each point. The wave equation says the
acceleration of the height is proportional to how much the point sits below the average of its
surroundings:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

where $c$ is the wave speed and $\nabla^2 u$ (the Laplacian) measures that difference from the
surroundings. A point lower than its neighbours is pulled up; a point higher is pulled down. The example
adds a small damping term (the **Damping** slider) so waves slowly fade.

**On a grid.** The program stores one height per cell and moves time forward in steps. It uses the
heights at the current step and the step before to get the next one:

$$u_{\text{new}} = 2u - u_{\text{old}} + C^2 \left(u_{\text{left}} + u_{\text{right}} + u_{\text{up}} + u_{\text{down}} - 4u\right)$$

The first two terms say "keep moving the way you were moving". The last term is the neighbours' pull.
This is the leapfrog scheme, and every new value depends only on old ones, so each cell can be updated
on its own.

**The Courant number.** $C$ (the **Wave speed (C)** slider) is how many cells a wave travels in one step.
In two dimensions the scheme is only stable for $C \le 1/\sqrt{2} \approx 0.707$. Past that, the
pattern that flips sign from cell to cell, like a checkerboard, grows at every step instead of
oscillating. This is the CFL condition, found by Courant, Friedrichs and Lewy in 1928.

## How the GPU does it

Each GPU thread handles one cell, in groups of 8 × 8.

- **Two buffers, three time levels.** The scheme needs the current and the previous heights. Each thread
  reads the current heights of its cell and its four neighbours, reads the previous height of its own
  cell only, and writes the new height over that previous value. No other thread needs that value, so
  this is safe. Then the two buffers swap roles.
- **Many steps per frame.** One step over 196,608 cells is so fast that the program runs 1 to 16 steps
  per drawn frame (the **Steps / frame** slider), all in one batch of GPU work.
- **Drawing.** The renderer reads the heights straight from the same buffer, works out the slope of the
  surface for lighting, and colors the height from dark navy in the troughs to pale aqua on the crests.

> **GPU:** Each cell does about a dozen arithmetic operations but moves about 16 bytes of memory. Like
> most grid updates, the speed is set by memory bandwidth, not by arithmetic.

## How good is the answer

- **Energy.** Without damping, the scheme keeps an exact discrete energy constant, for any $C$. In the
  CPU tests, run in 32-bit floats, it drifts by only about $2 \times 10^{-7}$ over 1,500 steps at
  $C = 0.5$ and 0.7.
- **The limit.** In CI, the shipped shader on a software GPU ran 1,000 steps at every $C \le 0.705$ with
  no growth. Just past the limit it blew up: the height passed 1,000 times its starting value after 110
  steps at $C = 0.71$, 50 steps at 0.72 and 30 steps at 0.75. Tiny rounding errors start the checkerboard,
  and it grows by a fixed factor each step (1.2 at $C = 0.71$).
- **Short waves are slow.** Waves only a few cells long travel slightly slower than they should. At
  $C = 0.5$, the double-slit wave should be 10 cells long; the CPU test measures 9.87 cells.

## Try this

- In [Explore](#wave-equation), load **Ellipse** and press **Start**. The ripple from one focus reflects
  off the wall and meets again at the other focus.
- Set **Wave speed (C)** to 0.70, as close to the limit as the slider goes. Nothing goes wrong. Then run
  **Courant number vs stability (CFL)** in [Measure](#wave-equation/measure) to see what happens just past it.
- Load **Double slit** and count the bright stripes on the right. Press **Absorbing edges: on** to turn
  them off, and watch the reflections spoil the pattern.
- Set **Damping** to 0 and turn the rain on. Does the water ever settle?

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) derives the stability
limit from a Fourier analysis, gives the exact discrete energy, explains the sponge layer at the edges,
and shows how the uniform buffer gives each step its own parameters.
