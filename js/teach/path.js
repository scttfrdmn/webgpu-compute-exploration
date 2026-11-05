// The guided learning path (learn.html), per visitor level (#94): at Advanced, two ordered tracks through
// all 25 examples (TRACKS); at Intermediate, one core track through the main ideas; at Beginner, a short
// visual sequence of examples that have a Beginner Understand page. Pure data, rendered by js/teach/learn.js.
// test/learn-path.test.mjs checks that every example id is a section in index.html, that every example
// appears in an Advanced track, that Beginner steps only use examples with Beginner pages, that the Beginner
// and Intermediate text passes that level's reading rules (js/teach/readability.js), and that every *text*
// in `tryThis` (an experiment or control name, rendered as <em>) is a label in index.html or an experiment.

// Example numbers and titles, as in the <h3> of each section of index.html (the test checks they match)
export const EXAMPLES = {
    'vector-addition': { num: 1, title: 'Vector Addition' },
    'matrix-multiplication': { num: 2, title: 'Matrix Multiplication' },
    'gaussian-blur': { num: 3, title: 'Image Processing: Gaussian Blur' },
    'game-of-life': { num: 4, title: "Conway's Game of Life" },
    'molecular-dynamics': { num: 5, title: 'Molecular Dynamics' },
    'sph-fluid': { num: 6, title: 'SPH Fluid' },
    'ray-marching': { num: 7, title: 'Ray Marching: 3D Fractals' },
    'boids': { num: 8, title: 'Boids Flocking' },
    'dla': { num: 9, title: 'Diffusion-Limited Aggregation (3D)' },
    'reaction-diffusion': { num: 10, title: 'Reaction-Diffusion (Gray-Scott)' },
    'wave-equation': { num: 11, title: 'Wave Equation' },
    'falling-sand': { num: 12, title: 'Falling Sand' },
    'noise': { num: 13, title: 'Procedural Noise' },
    'fft-audio': { num: 14, title: 'FFT Audio Visualizer' },
    'parallel-sort': { num: 15, title: 'Parallel Sorting' },
    'kmeans': { num: 16, title: 'K-Means Clustering' },
    'voronoi': { num: 17, title: 'Voronoi (Jump Flooding)' },
    'maze': { num: 18, title: 'Maze Generation & Solving' },
    'cloth': { num: 19, title: 'Cloth Simulation' },
    'path-tracer': { num: 20, title: 'Path Tracer' },
    'mnist-train': { num: 21, title: 'MNIST: Train' },
    'mnist-recognize': { num: 22, title: 'MNIST: Recognize' },
    'nbody': { num: 23, title: 'N-Body Gravity (Barnes-Hut)' },
    'convolution-reverb': { num: 24, title: 'Convolution Reverb' },
    'wasm-mandelbrot': { num: 25, title: 'WASM Integration: Mandelbrot Set' },
};

export const TRACKS = [
    {
        id: 'gpu',
        title: 'GPU programming',
        intro: 'For people learning to program GPUs: how work is dispatched, how data moves, and how threads ' +
            'share results without racing each other. Each step adds one parallel pattern to the ones before it, ' +
            'from independent threads to reductions, scans, trees and constraint solvers, then real-time ' +
            'deadlines, a neural network, and the question of what to leave on the CPU.',
        steps: [
            {
                title: 'Dispatch and memory',
                examples: ['vector-addition', 'matrix-multiplication'],
                why: 'Start here. One thread per output and no communication between threads, so every cost you ' +
                    'see is buffers, transfers and launch overhead. Matrix multiplication adds 2D workgroups and ' +
                    'data reuse, the first sign that memory access, not arithmetic, sets the speed.',
                tryThis: 'In Vector Addition, run *Size vs bandwidth and overhead* and find the N where the ' +
                    "kernel's GB/s stops growing: below it, your GPU is not full.",
            },
            {
                title: 'Stencils and images',
                examples: ['gaussian-blur', 'game-of-life', 'reaction-diffusion'],
                why: 'Still one thread per output, but now each thread reads its neighbours. Inputs overlap, and a ' +
                    'time step needs two buffers (ping-pong) so that no thread reads a cell another thread has ' +
                    'already overwritten.',
                tryThis: 'In Gaussian Blur, run *Blur radius vs cost* and check that the time grows with the ' +
                    'number of taps (the reference line), not with σ.',
            },
            {
                title: 'Reductions and atomics',
                examples: ['kmeans', 'nbody'],
                why: 'A stencil writes one value per thread; a reduction does the opposite and combines many ' +
                    "threads' values into one: a cluster mean, a bounding box, a total energy. That takes " +
                    'workgroup memory, barriers and atomics, and WGSL has atomics only for integers.',
                tryThis: 'In K-Means, run *Clusters vs time per iteration*. The distance work is linear in K; is ' +
                    'the time? What is left over is the per-cluster reduction.',
            },
            {
                title: 'Scans and sorting',
                examples: ['parallel-sort'],
                why: 'A scan (prefix sum) is a reduction that keeps every partial result, and it turns "where ' +
                    'does this key go?" into a parallel question: radix sort is a histogram, a scan and a ' +
                    'scatter. Bitonic sort shows the other route, a fixed network with no data-dependent branches.',
                tryThis: 'Run *Sorting N keys: bitonic, bitonic with workgroup memory, radix, CPU* with u32 keys ' +
                    'and find the N at which radix sort passes the CPU sort on your GPU.',
            },
            {
                title: 'Spatial structures and trees',
                examples: ['voronoi', 'nbody', 'maze'],
                why: 'Reductions and sorting are what spatial structures are built from: Barnes-Hut takes a ' +
                    'bounding box, sorts the bodies by Morton code and builds its tree from the sorted order. Jump ' +
                    'flooding and BFS show two other shapes of spatial work: a fixed number of passes, and a ' +
                    'frontier that grows level by level.',
                tryThis: 'In N-Body, run *Scaling: Barnes-Hut vs brute force* and find the N at which the tree ' +
                    'starts to win on your GPU; the tree-build column shows how much of it is fixed overhead.',
            },
            {
                title: 'Graph coloring and constraint solving',
                examples: ['cloth'],
                why: 'Trees let many threads read a shared structure; a constraint solver has them write to ' +
                    'shared particles. Graph coloring splits the constraints into sets that share no particle, so ' +
                    'each set runs without races and Gauss-Seidel becomes parallel.',
                tryThis: 'In Cloth, choose *Hang from two corners* and drag *Iterations* down to 1, then run ' +
                    '*Solver iterations vs stretch* and compare the 99th percentile with the 1/N line.',
            },
            {
                title: 'Streaming and latency',
                examples: ['fft-audio', 'convolution-reverb'],
                why: 'Until now the question was throughput. Audio adds a deadline, a block of samples every few ' +
                    'milliseconds without fail, so launch overhead and readback latency matter more than peak ' +
                    'speed. The FFT also brings a new access pattern: butterflies whose stride changes every stage.',
                tryThis: 'In Convolution Reverb, run *IR length vs time per block and the real-time deadline* at ' +
                    'both block sizes. How much of the deadline does your GPU use at 6 s?',
            },
            {
                title: 'ML training and inference',
                examples: ['mnist-train', 'mnist-recognize'],
                why: 'A neural network puts the earlier patterns together: convolutions are stencils, dense ' +
                    'layers are matrix products, the loss and the gradients are reductions, and a training step is ' +
                    'dozens of small dispatches in a row, which makes dispatch overhead visible again.',
                tryThis: 'In MNIST: Train, run *Batch size vs time per mini-batch*. If the time per batch hardly ' +
                    'changes from 8 to 64, your GPU is limited by the number of dispatches, not by arithmetic.',
            },
            {
                title: 'Splitting work between CPU and GPU',
                examples: ['wasm-mandelbrot'],
                why: 'The last question is what not to put on the GPU. Rust compiled to WebAssembly, a compute ' +
                    'shader and JavaScript share one task here, and the timing breakdown shows when moving the ' +
                    'data costs more than the GPU saves.',
                tryThis: 'Run *Resolution vs time: WebGPU, WASM and JavaScript* and find the image size at which ' +
                    'the GPU starts to win.',
            },
        ],
    },
    {
        id: 'science',
        title: 'Computational science',
        intro: 'For people learning computational science: how equations become computations, and how to ' +
            'tell when the numbers can be trusted. The GPU is the instrument; the subject is discretization, ' +
            'stability, convergence, accuracy and scaling, from grids and particles to random sampling, ' +
            'spectral methods, optimization and procedural fields.',
        steps: [
            {
                title: 'Stencils and PDEs',
                examples: ['wave-equation', 'reaction-diffusion'],
                why: 'Start here. Replace derivatives with differences on a grid, and a partial differential ' +
                    'equation becomes the same local update at every cell, one GPU thread each. The wave equation ' +
                    'is the classic second-order case; Gray-Scott couples diffusion with a nonlinear reaction.',
                tryThis: 'In Reaction-Diffusion, run *Single precision vs float64* with Coral and then with ' +
                    'Chaos: same equations, same start, but only one stays close to the float64 run.',
            },
            {
                title: 'Stability and the CFL condition',
                examples: ['wave-equation', 'sph-fluid'],
                why: 'An explicit scheme is only as good as its time step: past the Courant-Friedrichs-Lewy ' +
                    'limit, errors grow every step instead of dying out. The grid of the wave equation and the ' +
                    'particles of SPH obey the same kind of limit.',
                tryThis: 'In Wave Equation, run *Courant number vs stability (CFL)* and find where it blows up, ' +
                    'and how quickly.',
            },
            {
                title: 'Particles and integrators',
                examples: ['molecular-dynamics', 'nbody', 'boids'],
                why: 'From fields on a grid to particles moving under forces. Stability is now a property of the ' +
                    'integrator: how the energy error scales with the time step, and whether it stays bounded. ' +
                    'Boids are the limiting case, rules with no energy at all, where what you measure is ' +
                    'emergent order.',
                tryThis: 'In Molecular Dynamics, run *Time step vs energy error* with the thermostat off: the ' +
                    'maximum error should halve with Δt. Then run the N-Body experiment of the same name, where ' +
                    'leapfrog makes it fall 4×.',
            },
            {
                title: 'Stochastic growth and cellular automata',
                examples: ['dla', 'falling-sand', 'game-of-life'],
                why: 'Deterministic integrators give way to random walks and discrete rules. Simple local updates ' +
                    'grow fractals, move and transform materials, and compute patterns; the useful measurements ' +
                    'become statistics: fractal dimension, conserved counts, agreement with a CPU reference.',
                tryThis: 'In DLA, run *Stickiness vs structure* and compare your radius of gyration and fractal ' +
                    'dimension D with the table on the Understand page.',
            },
            {
                title: 'Monte Carlo',
                examples: ['path-tracer'],
                why: 'Randomness becomes the method itself: each light path is one random sample of an integral, ' +
                    'and the average converges like 1/√N in any dimension. Path tracing makes the error visible ' +
                    'as noise, and shows how importance sampling reduces it and how clamping trades it for bias.',
                tryThis: 'Run *Samples per pixel vs noise* on the Cornell box with and without light sampling, and ' +
                    'compare both curves with the 1/√N line.',
            },
            {
                title: 'Spectral methods',
                examples: ['fft-audio', 'convolution-reverb'],
                why: 'Instead of working cell by cell, transform to frequencies, where convolution becomes ' +
                    'multiplication. The FFT does it in O(N log N), and its rounding error grows only like log N; ' +
                    'convolution reverb uses it to apply a 6-second impulse response in real time.',
                tryThis: 'In FFT Audio Visualizer, run *FFT size vs accuracy* with white noise and check whether ' +
                    'the error follows the log₂ N reference.',
            },
            {
                title: 'Optimization and learning',
                examples: ['kmeans', 'mnist-train'],
                why: 'Iterations that minimize an objective instead of stepping through time. Lloyd\'s algorithm ' +
                    'reaches a local minimum that depends on where it starts; stochastic gradient descent depends ' +
                    'on its learning rate. Both are checked against a float64 run on the CPU.',
                tryThis: 'In MNIST: Train, run *Learning rate vs test accuracy*, find the rate where test ' +
                    "accuracy peaks, and compare your GPU's accuracies with CI's.",
            },
            {
                title: 'Procedural fields',
                examples: ['noise', 'ray-marching'],
                why: 'Last, functions evaluated everywhere at once. Noise synthesizes a random field with ' +
                    'controlled statistics, and ray marching finds the surface of a distance function by root ' +
                    'finding along every ray, so accuracy is a question of convergence and step budgets.',
                tryThis: 'In Ray Marching, run *Step budget vs cost and convergence* for the Mandelbulb and then ' +
                    'the Menger sponge. Which one converges in fewer steps, and why?',
            },
        ],
    },
];

// Intermediate (high school): one track through the core ideas, mostly examples with Intermediate pages
export const INTERMEDIATE_TRACKS = [
    {
        id: 'core',
        title: 'Core ideas',
        intro: 'One route through the main ideas of GPU computing, in order: threads that work alone, grids ' +
            'where each cell reads its neighbours, particles and forces, sorting, random sampling and a ' +
            'neural network. Each step uses the ideas of the steps before it.',
        steps: [
            {
                title: 'One thread per answer',
                examples: ['matrix-multiplication', 'wasm-mandelbrot'],
                why: 'Start here. A GPU runs thousands of small threads at the same time. Here each thread ' +
                    'works out one answer, one number of the product or one pixel of the picture, and never ' +
                    'needs the result of another thread. The cost to watch is moving data: to the GPU, from its ' +
                    'memory to the threads, and back.',
                tryThis: 'In Matrix Multiplication, run *CPU vs naive GPU vs tiled GPU* on the Measure tab. ' +
                    'When n doubles, does the time grow about eight times, as the n³ line says?',
            },
            {
                title: 'Grids and neighbours',
                examples: ['game-of-life', 'falling-sand'],
                why: 'Now each thread reads the cells next to it. A cell of the Game of Life counts its eight ' +
                    'neighbours, and a grain of sand looks at the cells below it. Threads must not change a cell ' +
                    'that another thread is still reading: the Game of Life keeps two copies of the grid and ' +
                    'swaps them, and Falling Sand gives each thread its own block of 2 by 2 cells.',
                tryThis: 'In Falling Sand, pour *Water* onto the sand heap, then *Oil* on top. Which ends up ' +
                    'where? Then run *What is conserved* in Measure.',
            },
            {
                title: 'Waves and the time step',
                examples: ['wave-equation'],
                why: 'The same neighbour update, now with physics: each point of the water is pulled toward its ' +
                    'neighbours, and that makes waves. The size of the time step matters. Past a limit, small ' +
                    'errors grow at every step until the numbers blow up.',
                tryThis: 'Set *Wave speed (C)* to 0.70, the top of the slider: nothing goes wrong. Then run ' +
                    '*Courant number vs stability (CFL)* in Measure and see what happens just past 0.707.',
            },
            {
                title: 'Particles and forces',
                examples: ['boids', 'nbody'],
                why: 'Leave the grid: now each thread moves one particle. A boid steers by the boids near it. A ' +
                    'star is pulled by every other star, which is N² pairs, so N-Body groups far-away stars ' +
                    'into one and builds a tree of groups at every step.',
                tryThis: 'In N-Body, switch *Mode* to *Brute force* at 16,384 and then 32,768 bodies. Does the ' +
                    'time per step go up by about four?',
            },
            {
                title: 'Sorting in parallel',
                examples: ['parallel-sort'],
                why: 'Sorting looks like a job for one worker, but it can be split up. Bitonic sort compares ' +
                    'fixed pairs of keys in stages, so every thread knows its partner ahead of time. Radix sort ' +
                    'counts the keys digit by digit and sends each key straight to its place.',
                tryThis: 'In Parallel Sorting, press *Run benchmark* with each *Algorithm* at 1,048,576 keys. ' +
                    'Which one is fastest on your GPU?',
            },
            {
                title: 'Random samples',
                examples: ['path-tracer'],
                why: 'Some answers are averages of many random tries. The path tracer follows one random path ' +
                    'of light per pixel in each frame and keeps a running average. The noise in the picture is ' +
                    'the error, and it shrinks slowly: four times the samples cut it in half.',
                tryThis: 'In Path Tracer, run *Samples per pixel vs noise* on the Measure tab, once with each ' +
                    '*Light sampling* choice. Which one starts with less noise?',
            },
            {
                title: 'Learning from examples',
                examples: ['mnist-train', 'mnist-recognize'],
                why: 'A neural network puts the earlier ideas together. Its first layers slide small filters ' +
                    'over the image, like a stencil on a grid, its last layer is a matrix product, and training ' +
                    'adds up the errors over a batch of images. The GPU runs every part.',
                tryThis: 'In MNIST: Train, press *Start* and watch the test accuracy after each epoch. Then press ' +
                    '*Use in Recognize* and draw a digit in MNIST: Recognize.',
            },
        ],
    },
];

// Beginner (middle school): a short visual sequence, only examples with a Beginner Understand page
export const BEGINNER_TRACKS = [
    {
        id: 'first-look',
        title: 'First look',
        intro: 'Five pictures that move, in order. Each one shows a new thing a GPU is good at. Open the ' +
            'example, try the one thing below, then read its Understand tab.',
        steps: [
            {
                title: 'The same rule in every square',
                examples: ['game-of-life'],
                why: 'Start here. The picture is a grid of small squares. Each square looks at the eight ' +
                    'squares around it and follows the same simple rule. The GPU gives every square its own ' +
                    'small worker, and they all work at the same time.',
                tryThis: 'Press *Glider*, then *Start*. Follow the little shape as it walks across the grid.',
            },
            {
                title: 'Sand, water and fire',
                examples: ['falling-sand'],
                why: 'Now each square holds sand, water, oil or fire, and each one has its own rules. Sand ' +
                    'falls, water flows to the side, and fire burns oil. The GPU still works on all the squares ' +
                    'at once.',
                tryThis: 'Press *Start*, pick *Water* and pour it onto the sand heap. Then pour *Oil* on top. ' +
                    'Which one ends up on top?',
            },
            {
                title: 'Ripples on a pond',
                examples: ['wave-equation'],
                why: 'Each point of the water moves up and down, pulled by the points next to it. That one ' +
                    'rule is enough to make ripples spread out, bounce off walls and cross each other.',
                tryThis: 'Load *Double slit* and press *Start*. Where the two waves cross, look for stripes of ' +
                    'strong and weak waves.',
            },
            {
                title: 'A flock of birds',
                examples: ['boids'],
                why: 'Now we leave the grid. Each bird, called a boid, looks at the boids near it. It steers to ' +
                    'stay close, to fly the same way, and to not bump into them. No one leads, but flocks form ' +
                    'all the same.',
                tryThis: 'Press *Start* and watch the flocks form. Then press *Add Predator* and watch the ' +
                    'others get out of its way.',
            },
            {
                title: 'Stars and galaxies',
                examples: ['nbody'],
                why: 'Every star pulls on every other star. With so many stars, that is a huge number of pulls ' +
                    'at each step. So the GPU takes a shortcut: a group of stars far away counts as one big star.',
                tryThis: 'Pick *Galaxy collision* and press *Start*. Then set *Mode* to *Brute force*, which ' +
                    'skips the shortcut, and see how much slower it gets.',
            },
        ],
    },
];

// The tracks at each visitor level (Advanced: the two full tracks)
export const TRACKS_BY_LEVEL = { beginner: BEGINNER_TRACKS, intermediate: INTERMEDIATE_TRACKS, advanced: TRACKS };

export const tracksForLevel = (level) => TRACKS_BY_LEVEL[level] ?? TRACKS;

// The page intro at the lower levels, rendered by learn.js (learn.html holds the Advanced one as static HTML)
export const LEVEL_INTROS = {
    beginner: 'Five examples to look at and play with, one after the other. Each step says why it comes next ' +
        'and gives you one thing to try. The links open the example, and its Understand tab tells you how it ' +
        'works in plain words.',
    intermediate: 'One route through the main ideas of GPU computing, from threads that work alone to grids, ' +
        'particles, sorting, random sampling and a neural network. Each step says why it comes next and gives ' +
        "you one thing to try, with links to the example's Explore, Understand and Measure tabs.",
};

// At Beginner the reference investigations are not the place to start: this line points at the first step
export const BEGINNER_START = 'New here? Start with step 1 below and go in order.';
