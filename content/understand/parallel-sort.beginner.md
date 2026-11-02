## What you see

Each bar is one number, and its height is its value. Short bars are dark blue, middle ones are teal, and
tall ones are yellow. At the start the bars are mixed up. Press **Play**, and the GPU puts them in order,
shortest on the left and tallest on the right.

The strip under the bars shows which bars get compared next. Each orange mark is paired with a cyan
mark some distance to its right. The dark background shows blocks of bars: in a bluish block the next
step sorts up, and in a purple block it sorts down. When the job is done, the strip turns green.

## What is happening

The bars are sorted with bitonic sort. It is a fixed plan of swaps. Picture a line of kids. At each
step, pairs of kids compare heights, and if the pair is in the wrong order, they swap places. The plan
says which kids pair up at each step, and it never changes. It does not matter what the heights are.

First the plan sorts tiny groups of two. Then it joins them into sorted groups of four, then eight, and
so on, until the whole line is one group. Some groups are sorted up and some down on purpose, because
joining an up group with a down group is easy. With 1024 bars the plan has 55 steps. Press **Step** to
run one step at a time, and watch the pairs in the strip.

This plan needs a number of bars like 512 or 1024, made by doubling. Pick 1000 bars and the program
adds 24 gray bars. They stand for the largest number there is, so they are drawn as tall as the
tallest bar, and they end up at the far right.

The benchmark (**Run benchmark**) sorts up to about four million numbers. It also tries a second method,
radix sort, which never compares two numbers at all. Think of sorting mail by zip code. First you put
the letters in piles by the last digit, then stack the piles in order. Then you sort by the next digit,
keeping the order inside each pile. When every digit is done, the mail is in order. Radix sort here
looks at 4 bits at a time, so it uses 16 piles and 8 rounds.

> **Numerics:** Sorting has no rounding at all. The answer is either right or wrong. Each benchmark run
> checks the GPU result against the browser's own sort, and the two must match exactly.

## Why a GPU helps

A GPU is like a huge team of workers who all work at the same time. In bitonic sort, every pair in a
step can be compared at once, because no two pairs share a bar. So each worker takes one pair.

> **GPU:** Bitonic sort does more comparing than the best one-at-a-time sorts, about 5 times more for a
> million numbers. On a long list it can still beat a one-at-a-time sort, because a whole step
> happens at once.

Radix sort has a harder job: each worker must know where its number goes. To find out, the team counts
how many numbers land in each pile. Then each worker adds up the counts before it, a trick called a scan.

On an Apple M4 Pro laptop in Chrome, the browser's sort took about 72 ms for 4,194,304 numbers. Radix
sort on the GPU took about 6 ms. Bitonic sort took about 29 ms, or 13 ms in the version that keeps
blocks of 512 numbers in fast shared memory.

## Try this

- Open [Explore](#parallel-sort), press **Step** a few times, then **Play**. Slow it down with the
  **Stages / s** slider.
- Set **Data** to **Reversed**. Does bitonic sort take fewer steps when the bars start in reverse? (The
  plan is fixed, so count them.)
- Set **Bars** to **1000 (padded)** and find the gray bars at the end.
- Press **Run benchmark** with each **Algorithm**. Which one is fastest?
- In [Measure](#parallel-sort/measure), run the experiment. At which size does the GPU start to beat the
  browser's sort?
