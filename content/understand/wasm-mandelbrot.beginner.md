## What you see

Press the button, and a famous shape appears: the Mandelbrot set. It is the black shape in the middle,
with a big round body, a smaller round head, and fuzzy edges full of tiny copies of itself. Around it,
the colors show how close each point is to being part of the set.

Most of the picture outside the set is red or orange. Those points leave fast. Points close to the edge
of the set take longer to leave, and the colors go from red to yellow, green, blue and purple the longer
they take.

The shape comes from one short rule, used again and again for each pixel. There are 480,000 pixels in
the picture, so that is a lot of work.

## What is happening

Each pixel stands for a point, given by two numbers: how far right and how far up it is. For each pixel,
the program plays a game:

1. Start at zero.
2. Square the number and add the pixel's point. (Squaring a point like this is a special way to mix its
   two numbers.)
3. Do it again, with the new number. And again.

For some points the number stays small forever. Those points are in the set, and they are drawn black.
For other points the number runs away and gets huge. As soon as it gets far enough from zero, it can
never come back, so the program stops. The color of the pixel shows how many turns it took. The program
gives up after 256 turns and calls the point black.

Three helpers share the work, a bit like a kitchen:

- **Rust and WebAssembly.** Rust is a programming language. WebAssembly is a compact form of a program
  that the browser can run almost as fast as a normal app on your computer. Here, Rust code turned into
  WebAssembly sets up the picture: its size, where it is centered, how far it is zoomed in. Then, at the
  end, it turns each count into a color and paints the picture.
- **The GPU** does the hard part: it plays the game for every pixel.
- **JavaScript**, the usual language of web pages, carries the data between the two.

## Why a GPU helps

A GPU is like a huge team of workers that all work at the same time. Here each worker takes one pixel and
plays the game for it. No pixel needs to know about any other pixel, so the whole team can work at once.

> **GPU:** Not every pixel takes the same time. A red pixel is done in a few turns, but a black one takes
> all 256. Workers who finish early have to wait for slow ones in their group.

The GPU has a cost too. The job has to be sent to the GPU, and the counts have to be sent back so the
WebAssembly code can color them. Much of that wait is the same for a small picture as for a big one. So
for a small picture, it can take longer than the work itself. Then one normal computer core, running the
same game in WebAssembly, can be faster than the GPU.

> **Numerics:** The GPU keeps about 7 digits for each number. The check that runs on the computer's main
> chip keeps about 16. Near the edge of the set, a tiny difference in the last digit can grow and grow,
> so a few pixels there come out with a different count.

## Try this

- Open [Explore](#wasm-mandelbrot) and press **Run WASM + WebGPU**. Read the times under the picture. One
  is for the GPU part, which also includes setting up the GPU and copying the counts back. The other is
  for the WebAssembly coloring.
- In [Measure](#wasm-mandelbrot/measure), run **Resolution vs time: WebGPU, WASM and JavaScript**. It
  plays the same game on the GPU and on one computer core, from tiny pictures to big ones. At what size
  does the GPU start to win?
- Run **Zoom vs agreement with float64**. As you zoom in deeper, the GPU runs out of digits. Next-door
  pixels end up with the same number, and the picture turns into blocks.
