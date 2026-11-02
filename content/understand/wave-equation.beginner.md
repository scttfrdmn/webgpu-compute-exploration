## What you see

The picture is the top of a pond, seen from above. When you click, a drop falls in and a ring of ripples
spreads out. The ripples bounce off the edges and off walls, and where two ripples cross, they add up or
cancel out.

The high parts of the water, the crests, are pale blue-green. The low parts are dark navy blue. Flat
water is a middle blue. The light makes the slopes shine, so the ripples look like real water. Walls are
a sandy tan color.

The same math is used for sound in a room and for light. So when you play with these ripples, you are
also seeing how sound and light waves act.

## What is happening

The pond is a grid of 512 by 384 little squares, called cells. Each cell holds one number: how high the
water is there. The program moves time forward in tiny steps. In each step, every cell does the same
thing:

1. Look at the four cells next to it, up, down, left and right.
2. If they are higher than it, its water speeds up going up. If they are lower, it speeds up going down.
3. Keep moving the way it was already moving, and work out the new height.

That is all. Nobody tells the water to make rings. Each cell only listens to its four neighbors, and the
rings spread out by themselves, like a wave going through a crowd at a game.

The **Wave speed (C)** slider sets how far a wave moves in one step. It is a share of one cell, from 0.10
to 0.70.

> **Numerics:** Why does the slider stop at 0.70? If a wave tried to move more than about 0.707 cells per
> step, it would outrun the rule that moves it. Then tiny errors grow and grow, and the whole pond blows up.
> Just past the edge it takes about a hundred steps. A bit further, and it takes only thirty. The
> [Measure](#wave-equation/measure) tab lets you go past that edge and watch.

**Damping** makes the waves slowly fade, like real water. With damping at 0, a wave would keep going
forever.

## Why a GPU helps

A GPU is like a huge team of workers that all work at the same time. Here each worker takes one cell. It
reads its own height and its four neighbors, and writes its new height. Every cell needs the same simple
job, so the whole team works at once.

> **GPU:** The pond has 196,608 cells, so each step has 196,608 small jobs. A GPU does one step so fast that
> the program runs up to 16 steps for each picture it draws. The **Steps / frame** slider sets how many.

## Try this

- Open [Explore](#wave-equation), press **Start**, and click on the water. Hold the mouse button down to
  keep dropping ripples.
- Hold Shift and drag to draw a wall. Drop a ripple next to it and watch it bounce.
- Load **Double slit**. A long straight wave comes from the left and goes through two gaps in a wall. On
  the right side, the two new waves cross and make a pattern of strong and weak stripes.
- Load **Ellipse** and press **Start**. The ripple starts at one special point. It bounces off the curved
  wall and comes back together at a second point on the other side.
- Press **Rain: off** to turn the rain on, and set **Damping** to 0. What happens to the pond?
- In [Measure](#wave-equation/measure), run **Courant number vs stability (CFL)** to see the blow-up just
  past 0.707.
