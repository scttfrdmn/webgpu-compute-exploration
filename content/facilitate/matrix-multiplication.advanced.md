## Goal

One learner, about 25 minutes, at the Advanced level (college). The session uses the matrix multiplication
reference investigation, and asks the same question as the Beginner and Intermediate sessions in its
grown-up form:

*Tiling is a shortcut for memory: each number is fetched from far memory less often. What does it buy on
this GPU, does the GPU beat one CPU core, and how do you know the answers are right and the claims hold?*

The learner states predictions with numbers before running anything, runs **CPU vs naive GPU vs tiled GPU**
in the **Measure** tab, and judges each claim in **Measured vs predicted** against its verdict and its
uncertainty. By the end they should be able to say which predictions held on this device, which stayed
undecided and why, what the verification checks did and did not establish, and what the result does *not*
show (it is one device and one browser, not a GPU ranking).

You are watching how the learner reasons from a measurement, not testing their linear algebra.

## Setup

- *Computer and browser.* A laptop or desktop with a recent Chrome or Edge, or Safari 26 or later; the
  header should say **WebGPU ready**, and it also says whether GPU timestamps are available. Plug the
  laptop in and close other tabs that use the GPU.
- *The links.* The reference card on the catalog page,
  [index.html?level=advanced#reference-investigations](index.html?level=advanced#reference-investigations),
  then the example,
  [index.html?level=advanced#matrix-multiplication/measure](index.html?level=advanced#matrix-multiplication/measure).
  The **Level** menu should read **Advanced (college)**: it shows the full claims, the verification list and
  the scope notes.
- *Try it yourself first* (10 minutes): a run at **Largest n** 1,024 and one at 2,048. On a slow GPU the
  2,048 run may stop at the 20-second budget with fewer sizes; that is a result too, but know it in advance.
- *What the learner will use.* In **Explore**: **Run Matrix Multiplication** (one 512 × 512 product, with
  wall and GPU time, GFLOPS and a CPU check). On the card: **Run it** and **How it works**. In **Measure**:
  **This shows** and **This does not show**, the setting **Largest n**, **Run**, the charts (**Time vs n,
  with an n³ line** is the one to read), the results table, the verification list, **Measured vs
  predicted**, **Download result (JSON)** and **Compare with a result file…**.
- *Paper.* For the learner's numeric predictions, and a copy of the rubric for you.

## What to say

The Say boxes are for the learner; adapt the wording freely.

> **Say:** Multiplying two n by n matrices takes about 2n³ operations. The card lists the predictions this
> investigation tests. Before you run anything, write down your own numbers: how much longer should a
> product twice as wide take, how much faster should the tiled kernel be than the naive one, and from what
> size should the GPU beat one CPU core?

Before the first run:

> **Say:** Read "This shows" and "This does not show" first. Then run it. When it finishes, go through the
> claims one by one: for each, is it supported, contradicted or undecided, and how sure are you?

Before the second run:

> **Say:** Change exactly one setting and run again. Which one, and what do you expect to change?

## What not to explain first

Let the learner find these out and say them; explain only if asked, after they have tried.

- What tiling does in workgroup memory. They should predict its effect from "fewer fetches from far
  memory" and check it, before reading **How it works**.
- Why the measured tiling speedup is much smaller than the factor of 16 in the prediction text (caches
  already serve many of the naive kernel's loads; see the claim **The naive kernel is bound by memory
  traffic, not arithmetic**).
- Why small sizes are left out of the slope fits (fixed launch cost, a GPU that is not full yet). The notes
  under the table say so.
- What √n·u means in the verification checks.
- That an "undetermined" verdict is not a failure.

## Plan

About 25 minutes; the second run and the comparison can be dropped if time is short.

1. *Read the card (0–4 min).* On the catalog page, read the matrix multiplication card under **Reference
   investigations**: its question, protocol and the two predictions under **Tests:**. The learner writes
   numeric predictions.
2. *Quick look (4–6 min).* In **Explore**, press **Run Matrix Multiplication** and read the report:
   computation time, GFLOPS and the CPU check. Ask what the wall time includes that the GPU time does not.
3. *First run (6–13 min).* In **Measure**, read **This shows** and **This does not show**, leave **Largest
   n** at 1,024 and press **Run**. While it runs, ask what they expect the chart **Time vs n, with an n³
   line** to look like.
4. *Judge the claims (13–19 min).* In **Measured vs predicted**, the learner goes row by row: the three
   "time grows like n³" claims, **Tiling speeds the GPU kernel up** and **The GPU beats one CPU core from
   some n on**. For each: prediction, measurement with its interval, verdict, and whether they agree with
   the verdict. Then the verification list: what does a PASS establish?
5. *One change (19–24 min).* **Download result (JSON)**, set **Largest n** to 2,048 only, **Run** again,
   then **Compare with a result file…** with the first file: do the sizes both runs share give the same
   times? Which claims changed verdict, and why?
6. *Conclusion (24–28 min).* The questions in the ending below.

## Prompts if stuck

Use the lightest prompt that works; give the learner time to think before the next.

- No numeric prediction: "Twice as wide: how many more multiply-adds per output, and how many more outputs?"
- Reads only the verdict column: "What interval does the measured slope have? Would 2 be inside it?"
- Treats "undetermined" as failure: "What would you need to decide it? More sizes, less noise, a longer run?"
- Over-reads one run: "Would another browser on this machine give the same numbers? What does the card's
  caveat say?"
- Confuses the verification checks with the claims: "If the products were wrong, which part of the page
  would tell you? If they were right but slow?"
- Changes several things between runs (another tab, another browser, another setting): "Which difference
  caused the change in the comparison?"
- Surprised by a small tiling gain: "Where else could the naive kernel get its numbers from, if not from
  far memory every time?"
- Wants to compare with a friend's GPU: "What would make that a fair comparison? Look at **This does not
  show**."

## Ending

> **Say:** Which predictions held on this GPU, which did not, and which stayed open? What would you need to
> do to make a claim about GPUs in general rather than this one?

A strong answer separates correctness (the verification checks against a float64 reference) from the
performance claims (slopes and ratios with intervals), names at least one claim that stayed undecided and
what would decide it, and limits the conclusion to this device and browser. If they want more, point them
to **How it works** (the Understand tab) and to the validation page, where the same runs from several
devices are collected.

Thank the learner and name something concrete they did well. Fill in the rubric straight after.

## Observation rubric

One level per row; note a quote or an action as evidence.

### Forms a prediction

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Runs first and explains the result afterwards. | "It grows fast, makes sense." |
| Partly | Predicts directions but no numbers, or numbers without a reason. | "Tiled will be faster." |
| Clearly | Writes numeric predictions with reasons before running, including where they might fail. | "Slope 3 from 2n³; tiled 2–5× faster, less than 16 because of caches; GPU wins from about 256." |

### Changes one thing at a time

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Changes the setting and the conditions together, or compares runs from different browsers as if alike. | Opens a video in another tab between runs, then compares. |
| Partly | Changes one setting but does not use the saved first result to compare. | Remembers the first slope "roughly" instead of comparing files. |
| Clearly | Changes only Largest n, keeps the conditions, and compares the two result files. | "Same tab, same browser, only the largest n. The shared sizes repeat within a few percent, and the slope interval got narrower." |

### Reads the measurement correctly

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Reads single points off the chart, or takes the verdict without the interval. | "At 1,024 it took 4 ms, so it's fast." |
| Partly | Reads slopes and verdicts, but mixes up kernel time with wall time, or the checks with the claims. | "The check passed, so tiling works." |
| Clearly | Reads each claim's measured value with its interval, tells verification from interpretation, and notes the timing source. | "Slope 2.9, interval 2.7 to 3.1, contains 3: agrees. The checks only say the products are right." |

### Explains the conclusion in their own words

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Restates the verdict column without saying what it means. | "It says agrees." |
| Partly | Explains the result for this device but generalises beyond it, or ignores undecided claims. | "So GPUs are always 50 times faster than CPUs." |
| Clearly | Explains what held, what stayed open and why, and the limits of one device and one browser. | "On this laptop tiling gave about 2×, the GPU won from 256; it's one run on one machine, the scope note says so." |

## Privacy and consent

- The site collects nothing: no account, no tracking, nothing is sent. A downloaded result file contains
  the browser, GPU and settings, not the person; it stays wherever you save it.
- Sessions with minors need the usual consent from a parent or guardian, or the school's own process. With
  adults, say what you are noting and that they can stop at any time.
- Write your observations on paper or in your own notes; do not record names, faces or voices unless your
  consent covers it.
- Share only anonymised findings: the level, the session, the rubric marks and what was confusing, with an
  age range at most. The [learning session form](https://github.com/scttfrdmn/webgpu-compute-exploration/issues/new?template=learning-session.md)
  asks for exactly that, and it is public.
