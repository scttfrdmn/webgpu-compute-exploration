// The guided learning path (learn.html): two ordered tracks through the 25 examples.
// Pure data, rendered by js/teach/learn.js. test/learn-path.test.mjs checks that every example id is a
// section in index.html, that every example appears in at least one track, and that every step has its
// "why" and "try this" text. In `tryThis`, *text* marks an experiment or control name (rendered as <em>).

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
