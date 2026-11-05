## What you see

Each dot is a star. Every star pulls on every other star, and the pull gets weaker the farther apart
they are. That is all the rules there are. From them, swirls, clumps and spiral arms form by themselves.

The colors show how fast each star moves. Slow stars are deep blue, faster ones turn white, then gold,
and the fastest are orange-red. Try the
**Spiral galaxy** and **Galaxy collision** presets to see two very different results of the same rules.

## What is happening

The program moves time forward in tiny steps, again and again. In each step it does two things:

1. For each star, add up the pulls from all the other stars. That gives the push on that star right now.
2. Move every star a little bit, using that push to speed it up or turn it.

Adding up every pull is a lot of work. With 16,384 stars, each star feels 16,383 others. That is about
268 million pulls for one step, and the program does many steps each second.

> **Numerics:** A real star feels the pull at every instant, but the program only checks at the end of
> each step. Small steps keep that error small. Big steps are faster, but stars can fly off the wrong way.

There is a clever shortcut, called Barnes-Hut. Think of a far-away city at night. From far away you do
not see each light. You see one bright glow. In the same way, a far group of stars pulls almost like one
big star at its middle. So the program puts the stars into boxes, and boxes inside boxes. A star then
adds up the pull of whole far boxes at once, and only looks at single stars that are close.

The **θ** slider sets how far away a box must be before the shortcut is used. A small θ is more exact but
slower. A big θ is faster but less exact.

How do we know the answer is any good? The total energy of all the stars should stay the same. The
readout under the picture shows how much it drifts. A tiny drift means the steps are small enough and the
shortcut is close enough.

## Why a GPU helps

A GPU is like a huge team of workers. Each one is not very smart, but there are thousands of them, and
they all work at the same time. Here each worker takes one star and adds up the pulls on it. Every star
needs the same simple job done, so the team can do all of them at once.

> **GPU:** One worker per star means 16,384 workers for one step. A normal processor has only a few of
> these workers, so it would take turns and be much slower.

## Try this

- Open [Explore](#nbody), press **Start**, and pick **Galaxy collision**. Watch the two galaxies pass
  through each other and throw out long tails.
- Move the **θ** slider to the right. Does it run faster? Does the energy drift get bigger?
- Set **Mode** to **Brute force**. Now every star adds up every pull, with no shortcut. How much slower is it?
- In [Measure](#nbody/measure), run the first experiment. It checks how much the shortcut changes the pulls.
