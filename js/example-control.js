// Pausing and resuming an example from outside it: used when its Explore tab is hidden (js/teach/tabs.js)
// and when its canvas scrolls off screen (js/main.js). Goes through the example's own buttons whenever
// it has them, so labels and state stay consistent with what the user would see after clicking.

import { getExampleInstance } from './teach/app-state.js';

// Audio examples are stopped but never restarted automatically: sound coming back on its own is jarring.
export const NO_AUTO_RESUME = new Set(['fft-audio', 'convolution-reverb']);

function runningInstance(container) {
    for (const canvas of container.querySelectorAll('canvas[id]')) {
        const instance = getExampleInstance(canvas.id);
        if (instance?.running) return instance;
    }
    return null;
}

// Pauses the example inside `container` if it is running. Returns a function that resumes it (only if it
// is still paused), or null if nothing was running.
export function pauseExample(container) {
    const instance = runningInstance(container);
    if (!instance) return null;

    // Separate Start / Pause (or Stop) buttons
    const stop = container.querySelector('button[id$="-pause"], button[id$="-stop"]');
    if (stop && !stop.disabled) {
        stop.click();
        const start = container.querySelector('button[id$="-start"], button[id$="-resume"]');
        return () => {
            if (!instance.running && start && !start.disabled) start.click();
        };
    }
    // A single toggle (▶ Animate / ⏸ Pause): its click handler flips the state and the label
    const toggle = container.querySelector('button.primary');
    if (toggle && !toggle.disabled) {
        toggle.click();
        return () => {
            if (!instance.running && !toggle.disabled) toggle.click();
        };
    }
    if (typeof instance.pause === 'function') {
        instance.pause();
        return () => {
            if (!instance.running && typeof instance.start === 'function') instance.start();
        };
    }
    return null;
}
