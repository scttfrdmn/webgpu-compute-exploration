## Overview

The Mandelbrot set is drawn by running one short rule for every pixel of an 800 × 600 image, 480,000
pixels in all. Each pixel is independent of the others, which makes it a classic job for a GPU. This
example also shows how three kinds of code work together in a web page: **Rust compiled to
WebAssembly** sets up the view and colors the result, a **WebGPU compute shader** does the counting for
every pixel, and **JavaScript** moves the data between them.

## The idea

**The rule.** Each pixel stands for a complex number $c = c_x + i\,c_y$. Starting from $z_0 = 0$, the
program repeats

$$z_{n+1} = z_n^2 + c$$

which, written with real numbers, is $x \leftarrow x^2 - y^2 + c_x$ and $y \leftarrow 2xy + c_y$. If
$|z|$ ever reaches 2, the orbit is sure to run off to infinity, so the program stops and records how many
steps it took. If it is still below 2 after 256 steps, the point is counted as part of the set and drawn
black. The other pixels get a color from their count: few steps give red, and more steps move through
yellow, green and blue to purple.

**The view.** The image is centered at $-0.5$ on the real axis, with each pixel $4 / 600$ wide at zoom 1,
so the picture is 4 units tall.

**What WebAssembly is.** WebAssembly (WASM) is a compact, low-level format that browsers run at close to
the speed of a normal compiled program, safely inside the page. You can write code in a language like
Rust and compile it to WASM. WASM runs on the CPU, not the GPU, and here it cannot talk to WebGPU
directly, so JavaScript owns the GPU and passes data back and forth.

## How the GPU does it

1. **WASM** builds the view settings (size, iteration limit, zoom, center) and hands them to JavaScript
   as six numbers.
2. **JavaScript** copies them into a small GPU buffer and launches the shader: one thread per pixel, in
   groups of 8 × 8, so 100 × 75 groups.
3. **Each thread** runs the rule for its pixel and writes the count to a GPU buffer.
4. **JavaScript** copies the 1.9 MB of counts back to the CPU.
5. **WASM** turns each count into a color and draws the image on the canvas.

> **GPU:** Neighbouring threads run in lockstep, but a pixel outside the set may stop after 2 steps while
> one inside runs all 256. The group keeps going until its slowest pixel is done, so threads near the
> edge of the set waste some time.

**What the comparison shows.** In [Explore](#wasm-mandelbrot) the readout splits the time in two: the GPU
part (which includes setting up the shader and reading the counts back, not only the counting) and the
WASM coloring. The [Measure](#wasm-mandelbrot/measure) tab makes a fairer race. It runs the same counting
loop on the GPU and on one CPU core, both in Rust compiled to WASM and in JavaScript, at four sizes from
80 × 60 to 640 × 480. For small images, the fixed cost of sending work to the GPU and reading it back is
larger than the counting itself, so the CPU can win. As the image grows, the GPU pulls ahead.

## How good is the answer

The shader uses 32-bit floats, which keep about 7 digits. The CPU versions use 64-bit floats, about 16
digits. Near the edge of the set, the rule is chaotic: a tiny difference in the last digit grows at each
step until the two versions stop at different counts.

- **The default view.** In CI, on a software GPU, 0.085% of the pixels of a 640 × 480 view differed from
  the 64-bit answer. Every one of them had a count of at least 38. Pixels that escape quickly always
  agree.
- **Deep zoom.** At the Seahorse valley ($-0.7436 + 0.1318\,i$), with a 256 × 192 view, CI measured 0.13%
  of pixels different at zoom 1, 9.2% at $10^3$ and 40% at $10^5$. By zoom $10^6$ the pixels are closer
  together than 32-bit floats can tell apart: only 90 of the 256 pixels in a row get their own value of
  $c_x$, and the image turns into blocks.

(CI's "GPU" runs on the CPU, so its timings say nothing about a real GPU; only the counts are checked
there.)

## Try this

- Press **Run WASM + WebGPU** in [Explore](#wasm-mandelbrot) and compare the GPU time with the WASM
  coloring time.
- Run **Resolution vs time: WebGPU, WASM and JavaScript** in [Measure](#wasm-mandelbrot/measure). At what
  size does the GPU, readback included, beat the CPU?
- Run it again with **Max iterations** at 64 and at 1024. Which side slows down more?
- Run **Zoom vs agreement with float64** and watch the column of distinct values per row shrink.

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) shows why the escape
radius is 2, gives the exact uniform layout, and names perturbation theory, the trick deep-zoom
programs use to get past the limits of floating point.
