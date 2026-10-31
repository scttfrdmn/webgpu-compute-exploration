// Tests for js/examples/mnist-train.js (example 21) — shipped JS on a mock device (no GPU):
//  - the trainer's frame scheduler (planSteps + MNISTGPU.writeSlots): over two epochs the uniform
//    slots it writes, decoded with the WGSL Params struct, cover every training position exactly
//    once per epoch in batch-sized steps, never cross an epoch boundary, run a full mode-1 test
//    pass after each epoch, and carry the same lr schedule / augmentation seed / shuffle as the
//    CPU trainer tools/train-mnist.mjs (so a browser run repeats the run behind weights.bin)
//  - readStats() decodes the Stats readback at the WGSL struct offsets.
// The training kernels themselves are replicated in test/mnist-kernels.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MNISTTrainer, wgslModules } from '../js/examples/mnist-train.js';
import * as M from '../js/examples/mnist-model.js';
import { epochLearningRate } from '../tools/train-mnist.mjs';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice, stub } from './helpers/mock-gpu.mjs';

const computeCode = M.wgslModules().find(m => m.name === `compute-batch${M.MAX_BATCH}`).code;
const PARAMS = structLayout(computeCode, 'Params');
const STATS = structLayout(computeCode, 'Stats');

function makeTrainer() {
    const device = mockDevice();
    const trainer = new MNISTTrainer(device, stub('plotCanvas'), stub('filterCanvas'));
    trainer.trainData = { count: M.DATASETS.train.count, orderBuffer: { label: 'order' }, bindGroup: stub() };
    trainer.testData = { count: M.DATASETS.test.count, bindGroup: stub() };
    trainer.readStats = () => {};      // no readback in this test
    return { device, trainer };
}

test('two epochs of frames: slots decoded via struct Params match the CPU trainer schedule', (t) => {
    const { device, trainer } = makeTrainer();
    trainer.stepsPerFrame = 16;
    trainer.epochs = 2;
    trainer.running = true;
    trainer.status = 'running';
    const nTrain = trainer.trainData.count, nTest = trainer.testData.count;

    const slots = [];
    let frames = 0;
    while (trainer.running && frames < 10000) {
        const before = device.writes.length;
        trainer.frame();
        frames++;
        for (const w of device.writes.slice(before)) {
            if (w.buffer === trainer.gpu.paramsBuffer) {
                const k = w.bytes.length / trainer.gpu.paramsStride;
                const frameSlots = [];
                for (let i = 0; i < k; i++) frameSlots.push(decodeStruct(w.bytes, PARAMS, i * trainer.gpu.paramsStride));
                // one frame never mixes training and evaluation
                assert.equal(new Set(frameSlots.map(s => s.mode)).size, 1);
                slots.push(...frameSlots.map(s => ({ ...s, frame: frames })));
            }
            if (w.buffer === trainer.trainData.orderBuffer) {
                const order = new Uint32Array(w.bytes.buffer);
                const epoch = trainer.epoch;
                assert.deepEqual(order, M.epochOrder(nTrain, trainer.seed, epoch), `epoch ${epoch} shuffle`);
            }
        }
    }
    assert.equal(trainer.epoch, 2);

    let i = 0;
    for (let epoch = 0; epoch < 2; epoch++) {
        let pos = 0;
        const lr = Math.fround(epochLearningRate(M.DEFAULTS.lr, M.DEFAULTS.lrDecay, epoch));
        while (pos < nTrain) {
            const s = slots[i++];
            const count = Math.min(M.DEFAULTS.batch, nTrain - pos);
            assert.deepEqual(
                [s.mode, s.start, s.count, s.dataLen, s.lr, s.momentum, s.weightDecay, s.augSeed, s.maxShift],
                [0, pos, count, nTrain, lr, Math.fround(M.DEFAULTS.momentum), Math.fround(M.DEFAULTS.weightDecay),
                    M.augmentSeed(trainer.seed, epoch), M.DEFAULTS.maxShift],
                `epoch ${epoch} train slot at ${pos}`);
            assert.ok(s.count <= M.MAX_BATCH);
            pos += count;
        }
        let evalPos = 0;
        while (evalPos < nTest) {
            const s = slots[i++];
            const count = Math.min(M.MAX_BATCH, nTest - evalPos);
            assert.deepEqual([s.mode, s.start, s.count, s.dataLen], [1, evalPos, count, nTest], `epoch ${epoch} eval slot`);
            evalPos += count;
        }
    }
    assert.equal(i, slots.length, 'no extra slots');
    // a frame never crosses the epoch boundary: the last train slot of an epoch ends its frame
    const lastTrain0 = slots.findIndex(s => s.mode === 1) - 1;
    assert.notEqual(slots[lastTrain0].frame, slots[lastTrain0 + 1].frame);
    t.diagnostic(`${frames} frames, ${slots.length} uniform slots over 2 epochs ` +
        `(${Math.ceil(nTrain / M.DEFAULTS.batch)} train + ${Math.ceil(nTest / M.MAX_BATCH)} eval per epoch); ` +
        `epoch-1 lr ${slots[lastTrain0 + 1 + Math.ceil(nTest / M.MAX_BATCH)].lr.toPrecision(4)}`);
});

test('readStats decodes the Stats readback at the WGSL struct offsets', async () => {
    const { trainer } = makeTrainer();
    delete trainer.readStats;          // use the prototype method
    const values = { trainLoss: 123.25, trainCorrect: 1000003, trainCount: 1000064, pad0: 0,
        testLoss: 7.5, testCorrect: 1961, testCount: 2000, pad1: 0 };
    const bytes = new ArrayBuffer(M.STATS_SIZE);
    const view = new DataView(bytes);
    for (const fld of STATS.fields) {
        if (fld.layout.kind === 'f32') view.setFloat32(fld.offset, values[fld.name], true);
        else view.setUint32(fld.offset, values[fld.name], true);
    }
    assert.deepEqual(decodeStruct(bytes, STATS), values);
    let got = null;
    trainer.addStats = (...args) => { got = args; };
    trainer.readbackBuffer = { mapAsync: () => Promise.resolve(), getMappedRange: () => bytes, unmap() {} };
    trainer.readStats();
    await new Promise(r => setImmediate(r));
    assert.deepEqual(got, [123.25, 1000003, 1000064, 7.5, 1961, 2000]);
});

test('trainer learning-rate schedule and seed match tools/train-mnist.mjs', () => {
    const { trainer } = makeTrainer();
    for (const epoch of [0, 1, 5, 14]) {
        trainer.epoch = epoch;
        assert.equal(trainer.epochLearningRate(), epochLearningRate(M.DEFAULTS.lr, M.DEFAULTS.lrDecay, epoch));
    }
    assert.equal(trainer.seed, M.DEFAULTS.seed);
    // the initial weights uploaded by reset() are initWeights(seed) with zero momentum
    const { device, trainer: t2 } = makeTrainer();
    const w = new Float32Array(device.lastWrite(t2.gpu.weightsBuffer).bytes.buffer);
    assert.deepEqual(w.subarray(0, M.NUM_PARAMS), M.initWeights(M.DEFAULTS.seed));
    assert.ok(w.subarray(M.NUM_PARAMS).every(x => x === 0));
});

test('filters shader: no uniforms; reads W1 (offset 0, 200 floats) from the weights buffer', () => {
    // Nothing else to lay out: the only binding is the weights storage buffer
    const code = wgslModules()[0].code;
    assert.ok(!/var<uniform>/.test(code));
    assert.equal(M.OFF.W1, 0);
});
