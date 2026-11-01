// Wall-clock phase timing for compute-only measurement APIs (e.g. measureForces in js/examples/nbody.js),
// so every experiment reports the same phases with the same meaning (see the protocol at the top of
// js/teach/measure.js):
//
//   compile   creating shader modules, pipelines and the buffers they bind (createComputePipeline is
//             synchronous to call; a driver may finish compiling lazily at the first dispatch, which is why
//             the measurement protocol discards a warm-up run)
//   upload    queue.writeBuffer calls for the inputs (the copy into the queue's staging memory; the
//             transfer itself overlaps the next submit)
//   dispatch  queue.submit of the measured work until onSubmittedWorkDone resolves
//   readback  copying results to a MAP_READ buffer, mapAsync, and the copy out of the mapped range
//
// These are wall-clock times (performance.now), including JavaScript, driver and queue latency. GPU
// execution time from timestamp queries (js/gpu-timer.js) is a separate quantity, reported as gpuMs.
//
//     const clock = new PhaseClock();
//     const sim = clock.time('compile', () => new Simulation(device));
//     clock.time('upload', () => sim.load(data));
//     await clock.time('dispatch', () => sim.run());      // async functions are awaited
//     clock.phases   // { compile, upload, dispatch } in ms (summed when a phase is timed more than once)

export const PHASES = ['compile', 'upload', 'dispatch', 'readback'];

export class PhaseClock {
    constructor({ now = () => performance.now() } = {}) {
        this.now = now;
        this.totals = {};
    }

    add(phase, ms) {
        this.totals[phase] = (this.totals[phase] ?? 0) + ms;
    }

    // Runs fn, adds its duration to `phase` and returns its result (a Promise if fn is async; the time is
    // added when it settles, also when it rejects)
    time(phase, fn) {
        const t0 = this.now();
        let result;
        try {
            result = fn();
        } catch (error) {
            this.add(phase, this.now() - t0);
            throw error;
        }
        if (result && typeof result.then === 'function') {
            return result.finally(() => this.add(phase, this.now() - t0));
        }
        this.add(phase, this.now() - t0);
        return result;
    }

    get phases() {
        return { ...this.totals };
    }
}
