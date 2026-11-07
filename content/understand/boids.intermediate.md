## Overview

Boids are simple agents that follow three steering rules, invented by Craig Reynolds in 1987. Each boid
only reacts to the boids near it, and none of them knows the shape of the flock. Still, flocks form,
split, merge and wheel around. The example moves up to 10,000 boids in a 1000 × 700 pixel world. It is
one of the first and best-known cases of simple local rules giving group behaviour, and the same idea is
used to model fish schools and crowds.

## The idea

**Three rules.** For each boid, the program looks at its neighbours and computes three steering forces:

- **Separation:** steer away from boids closer than 25 pixels. Each one pushes harder the closer it is.
  This rule has weight 1.5.
- **Alignment:** steer toward the average direction of the boids within 50 pixels (weight 1).
- **Cohesion:** steer toward the average position of the boids within 50 pixels (weight 1).

**Desired velocity.** Each rule picks a direction and asks: what if I flew that way at top speed? The
steering force is the difference between that desired velocity and the boid's real velocity, capped at a
small maximum ($F_{\max} = 0.1$ pixels per frame per frame). The cap is what makes turns smooth.

**The update.** The three forces, times their weights, plus the mouse force, give the acceleration
$\mathbf{a}$. Then, with one step per frame,

$$\mathbf{v} \leftarrow \operatorname{limit}(\mathbf{v} + \mathbf{a},\ v_{\max}), \qquad \mathbf{x} \leftarrow \mathbf{x} + \mathbf{v}$$

where limit shortens the velocity if it is longer than the top speed $v_{\max}$ (4 pixels per frame at
the start). A boid that leaves one edge comes back at the opposite one.

**The modes.** **Add Predator** makes boid 0 (always drawn red) steer to the center, and the others flee
it within 150 pixels. **Scatter** weakens the three rules to 10% and pushes everyone away from the center.
**Vortex** adds a strong swirl around the center.

## How the GPU does it

A GPU runs thousands of threads at once. Here each thread handles one boid, in groups of 64.

- **Two buffers.** The boids live in two buffers. Each step reads the old positions from one and writes
  the new ones to the other, then they swap. So no thread ever reads a boid that another thread is
  changing.
- **Checking every boid.** Each rule loops over all $N$ boids and tests the distance against its radius.
  That is $3N^2$ distance tests per step: 12 million at the default 2,000 boids, 300 million at 10,000.
  All threads in a group read the same boid at the same moment, which the hardware handles well.
- **Drawing.** The same buffer is drawn as one triangle per boid, turned to face its velocity and colored
  from green (slow) to yellow (fast). Nothing is copied back to the CPU.

> **GPU:** Most of those tests are wasted: a boid has only a few neighbours within 50 pixels. A grid of
> 50-pixel cells would let each boid check only the 9 cells around it, so the work would grow like $N$
> instead of $N^2$. That is why the **Boids** slider stops at 10,000.

## How good is the answer

Boids are not real physics, so nothing is conserved and there is no energy to check. The motion is also
chaotic: two runs that start a rounding error apart soon differ boid by boid. What can be checked is one
step, and the statistics of the flock.

- **One step.** From the same start, one GPU step agrees with a careful CPU step in double precision to
  $3 \times 10^{-5}$ pixels. That is just the rounding of the stored numbers. This was measured in CI,
  on a software GPU.
- **Order.** The polarization $\phi$ is the length of the average heading: 1 when all boids fly the same
  way, and about $1/\sqrt{N}$ when headings are random. In CI, 500 boids went from $\phi = 0.08$ to 0.36
  after 150 steps, 0.65 after 300 and 0.92 after 600. Small flocks line up first, and $\phi$ rises as they
  merge.
- **The step is one frame.** Speeds are in pixels per frame, so a 120 Hz screen runs the flock twice as
  fast as a 60 Hz one.

## Try this

- In [Explore](#boids), press **Reset** and **Start**, and watch the flocks form. Hold the right mouse
  button in the middle of a flock, then let go.
- Try **Add Predator**, **Scatter** and **Vortex**. Which one overrides the flocking rules?
- Watch a flock cross an edge. Neighbour distances do not wrap, so for a moment it is two flocks.
- In [Measure](#boids/measure), run **Emergence of order** with 500, 2,000 and 4,000 boids. Then run
  **Boids vs time per step**: does the time go up about four times when $N$ doubles?

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) has the exact
steering formulas, the buffer layout, why the polarization is an order parameter, and the link to the
Vicsek model from physics.
