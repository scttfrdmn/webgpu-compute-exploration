## Goal

One learner, about 25 minutes, at the Beginner level (middle school). The session is built around one
question:

*What changes when you let the computer take shortcuts, and how can you tell whether the shortcuts are
worth it?*

The N-body example moves thousands of stars that all pull on each other. It can add up every pull
(**Brute force**) or take a shortcut (**Barnes-Hut**), where a far-away group of stars counts as one big
star. The **θ** slider sets how much of the shortcut it takes. By the end, the learner should be able to
say in their own words that the shortcut makes the program faster but a little less exact, and that you
find out whether it is worth it by measuring both: the speed, and how far off the answer is.

You are not teaching gravity or programming. You are watching how the learner tests an idea.

## Setup

- *Computer and browser.* A laptop or desktop with a recent Chrome or Edge, or Safari 26 or later. When
  the page has loaded, the header says **WebGPU ready**. If it does not, the examples cannot run in that
  browser; try another one before the session. Plug the laptop in and close other tabs that play video or
  games.
- *The link.* Open [index.html?level=beginner#nbody](index.html?level=beginner#nbody). It opens the
  N-body example with the **Level** menu set to **Beginner (middle school)**, which hides the controls this
  session does not need. Keep this one browser tab for the whole session.
- *Try it yourself first* (5 minutes, the day before). Every computer gives different numbers, so note
  your own: the **Frame rate** with **Barnes-Hut** and with **Brute force**, and the **Energy drift** at
  θ 0.30 and at θ 1.00. If both modes show the same frame rate (fast computers stay at the screen's limit,
  often 60 FPS), use the **GPU** time in the same readout line instead, or set **Bodies** to **32,768**.
- *What the learner will use.* The **Explore** tab: **Start**, **Pause**, **Reset**, **Bodies**,
  **Mode**, **θ**, **Speed** and the presets **Spiral galaxy**, **Galaxy collision**, **Plummer sphere**
  and **Cold collapse**. The readout under the picture is long; only two of its lines matter today:
  **Frame rate** (pictures per second) and **Energy drift** (how far the total energy has moved from its
  starting value, in percent). Later, the **Measure** tab and its experiment **Opening angle vs force
  error**.
- *Two facts about the readout to know before you start.* The **Energy drift** starts again from zero
  every time **θ** or **Mode** changes, and after **Reset** or a preset. It also keeps growing for a while,
  so only compare two drift numbers taken at about the same time: the line says "over t = …", and you can
  wait until that t is about 10 each time. **Speed** changes how fast t grows, so leave **Speed** alone.
- *Paper.* A sheet for the learner to write guesses on, and a copy of the rubric below for you.

## What to say

Open with this, in your own words if you like. The Say boxes in this guide are written for the learner.

> **Say:** This is a model of stars in space. Every star pulls on every other star. The computer has to add
> up all those pulls, for every star, many times a second. That is a lot of work. Today we will find out
> what happens when we let the computer take a shortcut, and how we can tell if the shortcut is a good idea.

Then let them press **Start** and pick **Galaxy collision**.

> **Say:** Just watch for a minute. What do you see? You can drag the picture to turn it.

When they are ready for the first test:

> **Say:** Right now the computer uses a shortcut. Far-away stars are lumped together, like a far city
> looks like one glow at night. The **Mode** menu can turn the shortcut off. Before you try it, write down
> your guess. Will it run faster or slower without the shortcut? Why do you think so?

Before they move the **θ** slider:

> **Say:** This slider sets how much shortcut to take. More to the right means more stars get lumped
> together. Write down two guesses. Will it run faster or slower? Will the answer get better or worse?

Before they look at the **Energy drift** line:

> **Say:** Real stars never gain or lose energy in total. So if this number moves a lot, the model is
> making mistakes. Small is good. Big is bad.

Before the **Measure** tab:

> **Say:** Now the computer will test the shortcut for us. It works out the pulls the exact way and the
> shortcut way, and checks how far apart they are. What do you think it will find?

## What not to explain first

Let the learner find these out. Explain them only after the learner has tried, and only if they ask.

- Which mode or which θ is faster. The point is that they predict it and then check.
- How the shortcut works inside (boxes inside boxes, the tree). "Far stars count as one" is enough.
- What θ means exactly, or the word "opening angle". "How much shortcut" is enough.
- What energy is, or why it should stay the same. "It should not change, so a change means a mistake" is
  enough.
- The other numbers in the readout (E₀, K, W, the virial, interactions). They are for older learners.
- That the computer is "wrong". The shortcut is a trade, not a bug: the learner's job is to judge it.

## Plan

About 25 minutes. The times are a guide; let the learner take longer on a part that interests them and
shorten the last part if needed.

1. *Look (0–3 min).* **Start**, then **Galaxy collision**. Ask what they see. Accept any answer.
2. *First test: shortcut on or off (3–8 min).* The learner writes a guess, then sets **Mode** to
   **Brute force** and reads **Frame rate**. Then back to **Barnes-Hut**. Ask which was faster and by how
   much. If both show the same frame rate, point to the **GPU** time in the readout, or set **Bodies** to
   **32,768** (the largest that **Brute force** allows) and try again.
3. *Second test: how much shortcut (8–15 min).* In **Barnes-Hut** mode, the learner writes two guesses
   for moving **θ** to the right (speed, and how good the answer is). They move **θ** to 1.00, wait
   until "over t = …" reaches about 10, and write down the **Energy drift**. Then **θ** to 0.30, wait
   the same, write it down. For a "no shortcut" result, set **Mode** to **Brute force** and do it once more.
   Watch whether they change only **θ**, or several things at once.
4. *The computer's own test (15–22 min).* Open the **Measure** tab (the picture stops; that is
   expected). Under **Opening angle vs force error**, leave the settings as they are and press **Run**. It
   takes up to 20 seconds. Ask the learner to read the **Measured vs predicted** table out loud, especially
   the rows **A bigger θ: how far off the pulls are** and **A bigger θ: how much work each star does**.
   The chart and **Show the numbers** are optional.
5. *Is the shortcut worth it? (22–27 min).* Ask the questions in the ending below.

## Prompts if stuck

Use the lightest prompt that gets them moving again, and wait at least ten seconds before the next one.

- Not sure what to guess: "There is no wrong guess. What does your gut say?" or "If you had to add up a
  million numbers, would a shortcut make it faster or slower?"
- Changes several things at once: "If you change two things and the number moves, which one moved it?"
- Reads the drift too early: "Look at the 'over t = …' part. Was it the same both times?"
- Confused by plus and minus in the drift: "Look at how big the number is, not the sign in front."
- Lost in the readout: put a finger on the screen next to **Frame rate** and **Energy drift**.
- The frame rate does not change: "Your computer is so fast that both hit the screen's limit. Let's give
  it more stars." Set **Bodies** to **32,768**.
- Does not see why energy matters: "If a ball you threw came back faster than you threw it, would you
  trust that game?"
- Says the shortcut is "bad": "Bad for what? What if you only want a nice picture of a galaxy?"

## Ending

Ask, and let them answer without help:

> **Say:** So, is the shortcut worth it? When would you use it, and when would you turn it off? How did you
> find out?

Good follow-up questions: "If you were making a film of two galaxies, which θ would you pick?" and "If you
were planning the path of a space probe, would you pick the same?" There is no single right answer: a
good answer names both sides (faster, less exact) and says which number showed it.

Close by thanking them and saying what they did well, in concrete terms ("you changed only the slider",
"you waited for the same t"). Then fill in the rubric on your own, straight after the session.

## Observation rubric

Mark one level per row, from what you saw and heard. Note a quote or an action as evidence.

### Forms a prediction

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Changes a control without saying what will happen, or says "I don't know" after a prompt. | Moves θ straight away; afterwards: "Oh, it got faster." |
| Partly | Says what will happen, but not why, or only for one of speed and correctness. | "It'll be slower with Brute force." |
| Clearly | Says what will happen and why, before trying, for speed and for correctness. | "Faster, because it lumps more stars together, but the drift will be bigger." |

### Changes one thing at a time

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Changes several controls between two readings, or switches preset in the middle of a test. | Moves θ and Speed, picks Cold collapse, then compares the drift. |
| Partly | Changes one control, but compares readings taken at different times or after a Reset. | Reads the drift at t = 3 for one θ and at t = 20 for the other. |
| Clearly | Changes only the control being tested and keeps everything else, including the time of the reading. | Waits for "over t = 10" both times; puts θ back before trying Mode. |

### Reads the measurement correctly

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Reads the wrong line, or treats a bigger drift as better. | Reads E₀ as the drift; "0.4 is more than 0.01 so it's better." |
| Partly | Reads the right line, but is thrown by the sign or the size of small numbers. | "This one is minus, so it's smaller." |
| Clearly | Reads Frame rate and Energy drift correctly, compares sizes, and reads the Measured vs predicted row in their own words. | "The drift went from 0.01 to 0.2 percent, and the table says doubling θ made the error about 4 times bigger." |

### Explains the conclusion in their own words

| Level | What you see | For example |
| --- | --- | --- |
| Not yet | Repeats your words, or gives an answer without a reason. | "The shortcut is good." |
| Partly | Names one side of the trade, or the trade without the evidence. | "It's faster but worse." |
| Clearly | Names both sides, the evidence for each, and when the shortcut is worth it. | "With θ at 1 it ran faster but the energy moved a lot more. For a picture that's fine, for a real space probe I'd use Brute force." |

## Privacy and consent

- The site collects nothing: no account, no tracking, nothing is sent. The learner's settings stay in the
  browser.
- Sessions with minors need the usual consent from a parent or guardian, or the school's own process.
  Tell the learner what you are writing down and why, and that they can stop at any time.
- Write your observations on paper or in your own notes. Do not record names, faces or voices unless your
  consent covers it.
- Share only anonymised findings: the level, the session, the rubric marks and what was confusing, with an
  age range at most. The [learning session form](https://github.com/scttfrdmn/webgpu-compute-exploration/issues/new?template=learning-session.md)
  asks for exactly that, and it is public.
