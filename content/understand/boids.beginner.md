## What you see

Each small triangle is a bird, called a boid. The tip of the triangle points the way it flies. At the
start, every boid is in a random place and flies a random way. After a little while, they gather into
flocks that fly together, split up, join again and turn as one.

No boid knows where the flock is going. No boid is the leader. Each one only looks at the boids close to
it and follows three simple rules. The flock is what you get when thousands of boids all do that at once.

The colors show speed. Slow boids are green, and faster ones turn yellow. One boid is always drawn red.
It matters when you press **Add Predator**.

## What is happening

The program moves time forward one small step for each picture it draws. In each step, every boid looks
at its neighbors and follows three rules:

1. **Don't crowd.** If a boid is very close, steer away from it. This rule looks at boids within 25 pixels,
   and it counts a bit more than the other two.
2. **Fly the same way.** Turn toward the way the neighbors fly. This rule looks at boids within 50 pixels.
3. **Stay together.** Steer toward the middle of the neighbors, also within 50 pixels.

Each rule gives a small push. The program adds up the three pushes, turns the boid a little, and moves
it. A boid can only turn a little in each step, and it has a top speed. That is why the turns look
smooth, like real birds.

The world wraps around. A boid that flies off the right edge comes back on the left, like in some video
games.

How does a boid find its neighbors? In this program it checks every other boid, one by one, for each of
the three rules. With 2,000 boids that is 12 million checks in one step. Most of those checks find a boid
that is far away and does not count.

> **Numerics:** The flock does not repeat. Start it twice, and the boids end up in different places, since
> tiny changes grow over time. But the flocks still form each time. You can count on the big picture,
> not on where one boid will be.

## Why a GPU helps

A GPU is like a huge team of workers that all work at the same time. Here each worker takes one boid.
It checks all the other boids, adds up the three pushes, and moves its boid. Every boid needs the same
job done, so the whole team can work at once.

> **GPU:** With 10,000 boids, there are 10,000 workers in each step. Each one does 30,000 checks, and they
> all finish in about the same time, since they all do the same work.

There is a smarter way. Split the world into squares, like a checkerboard. Then a boid only has to check
the boids in its own square and the ones next to it. This program does not do that yet, which is why the
**Boids** slider stops at 10,000.

## Try this

- Open [Explore](#boids), press **Reset** and then **Start**. Watch the flocks form. How long does it
  take?
- Hold the left mouse button on the picture to pull boids toward you. Hold the right button to push them
  away. Try it in the middle of a flock.
- Press **Add Predator**. The red boid heads for the middle, and the others get out of its way. Then try
  **Scatter** and **Vortex**, and go back with **Normal Flocking**.
- Watch a flock fly over an edge. For a moment it is two flocks, since boids on one side do not see the
  boids that came out on the other side.
- In [Measure](#boids/measure), run **Emergence of order**. It shows how the boids go from flying every
  which way to flying the same way.
