## Overview

The example follows up to 262,144 stars that all pull on each other by gravity. No shape is built in.
Spiral arms, clumps and tidal tails come out of one rule, applied to every pair of stars, many times a
second. It is a small version of how astronomers study galaxies and star clusters.

## The idea

**Gravity between two stars.** Two masses $m_1$ and $m_2$ at distance $r$ pull on each other with a
force of size

$$F = \frac{G m_1 m_2}{r^2 + \varepsilon^2}$$

Without the small extra term $\varepsilon$ (the **Softening** slider), two stars that pass very close
would get a huge kick, and the step size could not keep up. With it, a near miss gives a strong but
limited push.

**Small time steps.** The program cannot move time forward smoothly. It jumps ahead by a step $\Delta t$
(the **Time step** slider). In each step it finds the force on every star, changes each velocity by
force over mass times $\Delta t$, and then moves each star by its velocity times $\Delta t$. The method
used, leapfrog, does the velocity update in two halves around the move. That small change makes it
much more accurate: halving $\Delta t$ cuts the error about four times, not two.

**Too many pairs.** With $N$ stars there are about $N^2$ pairs. For 16,384 stars that is 268 million
force sums per step. Double the stars and the work goes up four times.

**The Barnes-Hut shortcut.** A far group of stars pulls almost like a single star with the group's total
mass, placed at its center of mass. The program sorts the stars into a tree of boxes inside boxes. For
each star it walks the tree: if a box of size $s$ at distance $d$ has $s / d < \theta$, the whole box
counts as one mass; if not, the program opens the box and looks inside. Now each star only does a few
hundred to a few thousand force sums, and the total work grows only a little faster than $N$.

## How the GPU does it

A GPU runs thousands of threads at once. Here each thread handles one star.

- **Brute force:** each thread adds up the forces from all stars. To save trips to memory, a group of
  threads (a workgroup) first copies a block of 256 stars into fast shared memory, then every thread in
  the group uses that block before the next one is loaded.
- **Barnes-Hut:** the tree is built again on the GPU every step, in about 30 short passes: find the box
  around all stars, give each star a code from its position, sort the stars by that code, link them into
  a tree, then add up the mass of each box from the bottom up. After that, each thread walks the tree for
  its own star.
- **Moving the stars:** one more pass updates every position and velocity, and the picture is drawn
  straight from the same buffer, without copying the stars back to the CPU.

## How good is the answer

The shortcut and the time steps both bring in error. The example lets you see both.

- **Energy.** For real gravity, the total energy (motion plus position) never changes. The readout shows
  how far it drifts from its starting value. With brute force and a small step it stays very small.
- **The size of θ.** In the [Measure](#nbody/measure) tab, the first experiment compares Barnes-Hut forces
  with exact ones. The error grows roughly like $\theta^2$: double θ and the error gets about four times
  bigger. On an Apple M4 Pro the measured slope was 2.14, close to the predicted 2.
- **The size of the step.** The energy experiment halves $\Delta t$ again and again. With exact forces
  the energy error drops about four times each time, as leapfrog promises. With Barnes-Hut it stops
  dropping at a floor: below that, the error comes from θ, not from the step.

## Try this

- In [Explore](#nbody), pick **Galaxy collision** and press **Start**. Then try **Cold collapse**: all
  stars start at rest and fall together.
- Raise **θ** from 0.5 to 1.0 and watch the frame rate and the energy drift. Which one changes more?
- Switch **Mode** to **Brute force** at 16,384 and then 32,768 bodies. Does the time per step go up by
  about four?
- Run the three experiments in [Measure](#nbody/measure) and read the **Measured vs predicted** table.

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) shows the full force
law, the leapfrog equations, every step of the tree build and why it is safe on a GPU, and how the
measured slopes and their uncertainties are worked out.
