// pauseExample (js/example-control.js): pauses a running example through its own buttons and returns a
// resume function. Uses a tiny fake DOM container; no browser.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pauseExample, NO_AUTO_RESUME } from '../js/example-control.js';
import { setExampleInstance } from '../js/teach/app-state.js';

function button(id, { primary = false, onClick } = {}) {
    return { id, disabled: false, className: primary ? 'primary' : '', clicks: 0, click() { this.clicks++; onClick?.(); } };
}

// Supports the selectors pauseExample uses: 'canvas[id]', 'button[id$="-pause"], button[id$="-stop"]',
// 'button[id$="-start"], button[id$="-resume"]' and 'button.primary'
function container(canvasId, buttons) {
    return {
        querySelectorAll: (sel) => (sel === 'canvas[id]' ? [{ id: canvasId }] : []),
        querySelector(sel) {
            if (sel === 'button.primary') return buttons.find(b => b.className === 'primary') ?? null;
            const suffixes = [...sel.matchAll(/id\$="([^"]+)"/g)].map(m => m[1]);
            return buttons.find(b => suffixes.some(s => b.id.endsWith(s))) ?? null;
        },
    };
}

test('separate Start/Pause buttons: pause clicks Pause, resume clicks Start only if still paused', () => {
    const sim = { running: true };
    const pause = button('btn-x-pause', { onClick: () => { sim.running = false; } });
    const start = button('btn-x-start', { primary: true, onClick: () => { sim.running = true; } });
    setExampleInstance('canvas-x', sim);
    const resume = pauseExample(container('canvas-x', [start, pause]));
    assert.equal(pause.clicks, 1); assert.equal(sim.running, false);
    resume();
    assert.equal(start.clicks, 1); assert.equal(sim.running, true);
    // if the user already restarted it, resume does nothing
    sim.running = true;
    resume();
    assert.equal(start.clicks, 1);
});

test('toggle button (▶ Animate / ⏸ Pause): pause and resume both click the toggle', () => {
    const sim = { running: true };
    const toggle = button('btn-y-animate', { primary: true, onClick: () => { sim.running = !sim.running; } });
    setExampleInstance('canvas-y', sim);
    const resume = pauseExample(container('canvas-y', [toggle]));
    assert.equal(sim.running, false); assert.equal(toggle.clicks, 1);
    resume();
    assert.equal(sim.running, true); assert.equal(toggle.clicks, 2);
});

test('no buttons: falls back to the instance pause()/start()', () => {
    const sim = { running: true, pause() { this.running = false; }, start() { this.running = true; } };
    setExampleInstance('canvas-z', sim);
    const resume = pauseExample(container('canvas-z', []));
    assert.equal(sim.running, false);
    resume();
    assert.equal(sim.running, true);
});

test('nothing running (or not initialized): returns null and clicks nothing', () => {
    const pause = button('btn-w-pause');
    setExampleInstance('canvas-w', { running: false });
    assert.equal(pauseExample(container('canvas-w', [pause])), null);
    assert.equal(pauseExample(container('canvas-uninitialized', [pause])), null);
    assert.equal(pause.clicks, 0);
});

test('audio examples are never auto-resumed', () => {
    assert.ok(NO_AUTO_RESUME.has('fft-audio') && NO_AUTO_RESUME.has('convolution-reverb'));
});
