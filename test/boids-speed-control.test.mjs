import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { initBoids } from '../js/examples/boids.js';
import { mockDevice, stub } from './helpers/mock-gpu.mjs';

test('boids speed control matches the simulation on load and reset', async () => {
    const elements = new Map();
    const element = (id, value = '') => {
        const el = {
            id, value, textContent: '', innerHTML: '', listeners: {},
            addEventListener(type, listener) { this.listeners[type] = listener; },
            click() { this.listeners.click(); },
        };
        elements.set(id, el);
        return el;
    };
    const reset = element('btn-boids-reset');
    element('btn-boids-start');
    element('btn-boids-pause');
    const count = element('boids-count', '2000');
    element('boids-count-value');
    const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
    const speedMarkup = html.match(/<input type="range" id="boids-speed"[^>]*>/)?.[0];
    assert.ok(speedMarkup, 'boids speed slider should exist');
    const speedDefault = speedMarkup.match(/\bvalue="(\d+)"/)?.[1];
    const speed = element('boids-speed', speedDefault);
    const speedLabel = html.match(/<span id="boids-speed-value">(\d+)<\/span>/)?.[1];
    const speedValue = element('boids-speed-value');
    speedValue.textContent = speedLabel;
    element('canvas-boids').getContext = () => stub('ctx');
    element('output-boids');
    globalThis.document = {
        getElementById: id => elements.get(id),
        querySelectorAll: () => [],
    };
    globalThis.window ??= { addEventListener() {} };

    const simulation = await initBoids({ device: mockDevice() });
    assert.equal(speed.value, '40', 'slider should reflect the initial simulation speed');
    assert.equal(speedValue.textContent, '40');
    assert.equal(simulation.maxSpeed, 4);

    speed.value = '80';
    speed.listeners.input({ target: speed });
    reset.click();
    assert.equal(speed.value, '40', 'reset should restore the slider to the default speed');
    assert.equal(speedValue.textContent, '40');
});
