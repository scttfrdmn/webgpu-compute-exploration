## Goal

One learner, about 25 minutes, at the Intermediate level (high school). The question carries on from the
Beginner session, one step further:

*A simulation can be off for two reasons: it moves time forward in steps, and it takes a shortcut when it
adds up the pulls. Does a smaller time step always make the answer better, and how can you tell which of
the two errors you are looking at?*

The N-body example moves stars that all pull on each other, in steps of size **Time step**. It adds up the
pulls exactly (**Brute force**) or with the **Barnes-Hut** shortcut. With exact pulls, halving the time step
should make the energy error about four times smaller. With the shortcut, the error stops falling at a floor
set by the shortcut itself, so a smaller step only costs time. By the end, the learner should be able to say
that, show the measurement that proves it, and say what it means for choosing settings: there is no point
paying for tiny steps when the shortcut's error is bigger.

You are not teaching physics or numerical methods. You are watching how the learner tests an idea.

## Setup

- *Computer and browser.* A laptop or desktop with a recent Chrome or Edge, or Safari 26 or later. The page
  header should say **WebGPU ready**. Plug the laptop in and close other tabs that play video or games.
- *The link.* Open [index.html?level=intermediate#nbody](index.html?level=intermediate#nbody). It sets the
  **Level** menu to **Intermediate (high school)**, which shows the **Softening** and **Time step** sliders
  next to **θ** and **Speed**.
- *Try it yourself first* (10 minutes, the day before): both parts of the plan, so you know your computer's
  numbers. The two Measure runs take up to 20 seconds each.
- *What the learner will use.* In **Explore**: **Start**, **Pause**, **Reset**, **Mode**, **θ**, **Time step**,
  **Speed** and the preset **Plummer sphere**. In the readout under the picture: **Energy drift**, which is
  (E − E₀)/|E₀| in percent "over t = …", and the configuration line that shows the current dt. In
  **Measure**: the experiment **Time step vs energy error** with its settings **Bodies** and **Forces**
  (**Brute force (exact)** or **Barnes-Hut θ = 0.5**), **Run**, the chart, the results table and
  **Measured vs predicted**.
- *Three facts about the readout to know before you start.*
  1. The **Energy drift** restarts from zero when **θ**, **Softening** or **Mode** changes, and after
     **Reset** or a preset, but *not* when **Time step** changes. After moving **Time step**, press **Reset**
     so the drift measures only the new step.
  2. The drift moves up and down and grows for a while, so compare two drifts at the same "over t = …"
     (say t = 5). A smaller step takes longer to reach the same t; **Speed** 4× helps.
  3. The readout shows three decimals of a percent, so +0.000% only means "below 0.0005%". With
     **Brute force** the drift is often that small; the **Measure** experiment shows it in full.
- *Paper.* A sheet for the learner's predictions, and a copy of the rubric for you.

## What to say

The Say boxes are written for the learner; use your own words if you like.

> **Say:** This program moves stars forward in time, one small jump at a time. Each jump it works out the
> pull on every star, either exactly or with a shortcut. Both of those can make the answer a bit wrong.
> Today the question is: does a smaller jump always give a better answer?

Before the first Explore test:

> **Say:** Total energy should stay the same in real gravity, so the energy drift tells us how wrong the
> model is. Before you move the time step, write down a guess. If the step is half as big, what happens
> to the drift? Half as big? A quarter? No change?

Before the Measure runs:

> **Say:** This experiment tries four time steps and records the worst energy error for each. First it
> will use exact pulls. Then change only the forces to the shortcut and run it again. Write down what you
> expect each time.

## What not to explain first

Let the learner find these out; explain only after they have tried, and only if they ask.

- That halving the step makes the error about four times smaller with exact forces. That is the
  prediction they are testing; the **Understand** tab says it, so keep it for afterwards.
- That the shortcut sets a floor. This is the discovery of the session.
- How leapfrog works, or the words "second order" and "symplectic".
- The log scales on the chart. Ask what one grid line means before explaining it.
- The other numbers in the readout (E₀, K, W, the virial ratio).

## Plan

About 25 minutes. Let the learner take longer where they are curious, and cut the Explore part short if
time runs out: the Measure part matters most.

1. *Warm-up (0–3 min).* **Start**, then **Plummer sphere**: a round ball of stars in balance. Point at
   the **Energy drift** line and ask what it says.
2. *Explore: one change at a time (3–11 min).* In **Barnes-Hut** mode, the learner writes a guess, sets
   **Time step** to 0.020, presses **Reset**, waits until t is about 5 and writes down the drift. Then
   **Time step** to 0.005, **Reset**, the same wait, the same reading. Ask what changed, and whether
   anything else changed too. If they want to, they repeat it in **Brute force** mode (the drift may show
   +0.000%; see the facts above).
3. *Measure, exact pulls (11–16 min).* Open the **Measure** tab (the picture stops; that is expected). In
   **Time step vs energy error**, leave **Bodies** at 2,048 and **Forces** at **Brute force (exact)**, and
   press **Run**. The learner reads the row **Exact forces: energy error vs time step** in **Measured vs
   predicted**, and the column max |ΔE/E₀| in the table.
4. *Measure, the shortcut (16–22 min).* The learner predicts first, then changes only **Forces** to
   **Barnes-Hut θ = 0.5** and runs again. They read **Barnes-Hut forces: energy error vs time step** and
   **Barnes-Hut forces: the size of that floor**, and compare the two charts or tables.
5. *Conclusion (22–27 min).* The questions in the ending below.

## Prompts if stuck

Use the lightest prompt that works, and wait ten seconds before the next one.

- No prediction: "Imagine walking a curved path with your eyes shut, opening them every step. Do shorter
  steps keep you closer to the path?"
- Forgets to press **Reset**: "Is the drift you see now about the new step only, or the old one too?"
- Compares at different t: "Look at the 'over t = …' part of both readings."
- Struggles with numbers like 3.4e-5: "e-5 means move the point five places: 0.000034. Which is bigger,
  e-5 or e-6?"
- Reads the log chart as linear: "How far apart are 0.01 and 0.001 on this axis? And 0.001 and 0.0001?"
- Changes **Bodies** and **Forces** together: "If both change and the result changes, which one did it?"
- Says the shortcut result "failed": "The verdict column says whether the prediction held. What was the
  prediction for the shortcut?"
- Stuck on why the floor exists: "With the shortcut, is there still an error even if time stood still?"

## Ending

> **Say:** So, does a smaller time step always help? When does it help, and when is it a waste of time?
> Which numbers showed you?

Follow-ups: "If you had to run this for a whole day of computer time, how would you pick the time step and
θ together?" and "How would you check a new setting before trusting it?" A strong answer names both
sources of error, says which measurement separates them (changing only **Forces**), and draws a rule:
make the step small enough that its error is below the shortcut's floor, and no smaller.

Thank the learner and name something concrete they did well. Fill in the rubric straight after.

## Observation rubric

One level per row, from what you saw and heard; note a quote or an action as evidence.

### Forms a prediction

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Runs the experiment or moves a slider without a stated expectation. | Presses Run straight away; "let's see". |
| Partly | Predicts the direction but not the size, or only for exact forces. | "Smaller step, smaller drift." |
| Clearly | Predicts direction and size before each run, and a different outcome for the shortcut, with a reason. | "About four times smaller with exact forces; with Barnes-Hut maybe less, because the tree is off anyway." |

### Changes one thing at a time

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Changes several settings between runs, or forgets that the drift needs a Reset after a new Time step. | Switches Forces and Bodies together, then compares. |
| Partly | Changes one setting, but compares readings at different t or without a Reset. | Reads the drift at t = 2 for one step and t = 8 for the other. |
| Clearly | Changes only the setting under test, presses Reset, and reads at the same t; in Measure changes only Forces. | "Same bodies, same preset, only Forces is different this time." |

### Reads the measurement correctly

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Misreads the scale or the column, or takes a smaller number in e-notation for a larger one. | "3e-6 is bigger than 3e-5, it has a 6." |
| Partly | Reads single values right, but not the trend on the log chart or the size of the change. | "It went down" without saying by how much. |
| Clearly | Reads the worst energy error per step, the factor per halving, and the flat floor with the shortcut, and links them to the Measured vs predicted rows. | "Exact: about four times smaller each halving. Barnes-Hut: stays near 6e-4, the floor." |

### Explains the conclusion in their own words

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Repeats the verdict column or gives a rule with no evidence. | "Smaller is better." |
| Partly | Explains one source of error, or the floor without why it matters for choosing settings. | "The shortcut has its own error." |
| Clearly | Explains both errors, the evidence that separates them, and a sensible rule for choosing the step. | "With the shortcut the step stops mattering below some size, so a tiny step is wasted work; pick it so its error is under the floor." |

## Privacy and consent

- The site collects nothing: no account, no tracking, nothing is sent. Results stay in the browser unless
  someone downloads them.
- Sessions with minors need the usual consent from a parent or guardian, or the school's own process. Tell
  the learner what you are noting and that they can stop at any time.
- Write your observations on paper or in your own notes; do not record names, faces or voices unless your
  consent covers it.
- Share only anonymised findings: the level, the session, the rubric marks and what was confusing, with an
  age range at most. The [learning session form](https://github.com/scttfrdmn/webgpu-compute-exploration/issues/new?template=learning-session.md)
  asks for exactly that, and it is public.
