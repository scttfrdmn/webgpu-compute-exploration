// Small dependency-free SVG line chart for Measure results.
//
// Follows the site's data-viz rules: one y-axis per chart (a second measure gets its own chart), series
// colors in a fixed order validated for color-vision deficiency on the dark surface (--series-1…4 in
// css/site.css), 2px lines, markers r=4 with a 2px surface ring, recessive 1px solid gridlines, a legend
// plus direct end labels for 2+ series, text in text colors (never the series color), and a crosshair
// tooltip that snaps to the nearest x and lists every series. The results table is the table view.

const SVG = 'http://www.w3.org/2000/svg';
const MAX_SERIES = 4;

// ---- Pure helpers (tested in test/teach-measure.test.mjs) ------------------------------------------

// Round, evenly spaced ticks covering [min, max]
export function niceTicks(min, max, count = 5) {
    if (!(Number.isFinite(min) && Number.isFinite(max))) return [];
    if (min === max) { const pad = Math.abs(min) || 1; min -= pad / 2; max += pad / 2; }
    const raw = (max - min) / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
    return ticks;
}

// Decade ticks (1, 10, 100…) for a positive range; adds 2× and 5× when the range spans few decades
export function logTicks(min, max) {
    if (!(min > 0 && max > 0)) return [];
    const lo = Math.floor(Math.log10(min)), hi = Math.ceil(Math.log10(max));
    const ticks = [];
    const dense = hi - lo <= 2;
    for (let e = lo; e <= hi; e++) {
        for (const m of dense ? [1, 2, 5] : [1]) {
            const v = m * 10 ** e;
            if (v >= min * (1 - 1e-9) && v <= max * (1 + 1e-9)) ticks.push(+v.toPrecision(12));
        }
    }
    return ticks.length >= 2 ? ticks : [min, max];
}

export function makeScale({ min, max, log = false }, [r0, r1]) {
    if (log) {
        const a = Math.log10(min), b = Math.log10(max);
        const span = b - a || 1;
        return v => r0 + (Math.log10(v) - a) / span * (r1 - r0);
    }
    const span = max - min || 1;
    return v => r0 + (v - min) / span * (r1 - r0);
}

// Data extent of the given keys; log axes ignore non-positive values
export function extent(rows, keys, log) {
    let min = Infinity, max = -Infinity;
    for (const row of rows) for (const k of keys) {
        const v = row[k];
        if (typeof v !== 'number' || !Number.isFinite(v) || (log && v <= 0)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    if (min === Infinity) return null;
    if (log && min === max) return { min: min / 2, max: max * 2 };
    return { min, max };
}

export function formatTick(v) {
    const abs = Math.abs(v);
    if (abs === 0) return '0';
    if (abs >= 1e6 || abs < 1e-3) return v.toExponential(0).replace('e+', 'e');
    if (abs >= 1e4) return `${+(v / 1e3).toPrecision(3)}K`;
    return String(+v.toPrecision(3));
}

// ---- Rendering --------------------------------------------------------------------------------------

function el(name, attrs = {}, parent) {
    const node = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    parent?.append(node);
    return node;
}

// spec: { x, xLabel, logX, y: [keys], yLabel, logY, labels: {key: text}, reference: {label, points: [[x, y]]},
//         rows, columns, title? }
export function renderLineChart(holder, spec) {
    const keys = spec.y.slice(0, MAX_SERIES);
    const label = (k) => spec.labels?.[k] ?? spec.columns?.find(c => c.key === k)?.label ?? k;
    const rows = spec.rows.filter(r => typeof r[spec.x] === 'number' && (!spec.logX || r[spec.x] > 0));
    const caption = document.createElement('figcaption');
    caption.textContent = spec.title ?? `${spec.yLabel} vs ${spec.xLabel}`;
    holder.append(caption);
    const plottable = rows.some(r => keys.some(k => typeof r[k] === 'number' && Number.isFinite(r[k]) && (!spec.logY || r[k] > 0)));
    if (!plottable) {
        if (spec.rows.length) {
            const note = document.createElement('p');
            note.className = 'hint chart-empty';
            note.textContent = spec.emptyNote ?? 'No values to plot for this chart.';
            holder.append(note);
        }
        return;
    }

    // A reference may be a function of the measured rows, e.g. a θ² trend anchored to the visitor's own data
    const reference = typeof spec.reference === 'function' ? spec.reference(rows) : spec.reference;
    const refPoints = reference?.points ?? [];
    const xs = extent([...rows, ...refPoints.map(([x]) => ({ [spec.x]: x }))], [spec.x], spec.logX);
    const ys = extent([...rows, ...refPoints.map(([, y]) => ({ __ref: y }))], [...keys, '__ref'], spec.logY);
    if (!xs || !ys) return;
    const W = 560, H = 260, m = { l: 56, r: keys.length > 1 ? 96 : 20, t: 10, b: 40 };
    const xScale = makeScale({ ...xs, log: spec.logX }, [m.l, W - m.r]);
    const yScale = makeScale({ ...ys, log: spec.logY }, [H - m.b, m.t]);

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img',
        'aria-label': `${caption.textContent}. Values are listed in the table below.` }, holder);

    // Gridlines and ticks (recessive, solid)
    const yTicks = spec.logY ? logTicks(ys.min, ys.max) : niceTicks(ys.min, ys.max);
    for (const t of yTicks) {
        const y = yScale(t);
        if (y < m.t - 0.5 || y > H - m.b + 0.5) continue;
        el('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: 'chart-grid' }, svg);
        el('text', { x: m.l - 6, y: y + 4, 'text-anchor': 'end', class: 'chart-tick' }, svg).textContent = formatTick(t);
    }
    const xTicks = spec.logX ? logTicks(xs.min, xs.max) : niceTicks(xs.min, xs.max, 6);
    for (const t of xTicks) {
        const x = xScale(t);
        if (x < m.l - 0.5 || x > W - m.r + 0.5) continue;
        el('line', { x1: x, x2: x, y1: H - m.b, y2: H - m.b + 4, class: 'chart-axis' }, svg);
        el('text', { x, y: H - m.b + 16, 'text-anchor': 'middle', class: 'chart-tick' }, svg).textContent = formatTick(t);
    }
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: 'chart-axis' }, svg);
    el('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'chart-label' }, svg).textContent = spec.xLabel;
    el('text', { x: 12, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', class: 'chart-label',
        transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, svg).textContent = spec.yLabel;

    // Reference line (CI baseline or expected slope), muted and labeled
    if (refPoints.length >= 2) {
        const d = refPoints.map(([x, y], i) => `${i ? 'L' : 'M'}${xScale(x).toFixed(1)},${yScale(y).toFixed(1)}`).join('');
        el('path', { d, class: 'chart-reference' }, svg);
        const [lx, ly] = refPoints[refPoints.length - 1];
        el('text', { x: xScale(lx) - 4, y: yScale(ly) - 6, 'text-anchor': 'end', class: 'chart-tick' }, svg)
            .textContent = reference.label;
    }

    // Series: 2px line, markers with a surface ring, direct label at the line end
    keys.forEach((k, i) => {
        const pts = rows.filter(r => typeof r[k] === 'number' && Number.isFinite(r[k]) && (!spec.logY || r[k] > 0))
            .map(r => [xScale(r[spec.x]), yScale(r[k])]);
        if (!pts.length) return;
        const cls = `series-${i + 1}`;
        if (pts.length > 1) el('path', { d: pts.map(([x, y], j) => `${j ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(''),
            class: `chart-line ${cls}` }, svg);
        for (const [x, y] of pts) el('circle', { cx: x, cy: y, r: 4, class: `chart-dot ${cls}` }, svg);
        if (keys.length > 1) {
            const [x, y] = pts[pts.length - 1];
            el('text', { x: x + 8, y: y + 4, class: 'chart-direct' }, svg).textContent = label(k);
        }
    });

    // Legend for 2+ series (line keys, text in text color)
    if (keys.length > 1) {
        const legend = document.createElement('div');
        legend.className = 'chart-legend';
        keys.forEach((k, i) => {
            const item = document.createElement('span');
            const key = document.createElement('span');
            key.className = `chart-key series-${i + 1}`;
            item.append(key, label(k));
            legend.append(item);
        });
        holder.append(legend);
    }

    // Crosshair + tooltip snapping to the nearest x, listing every series
    const cross = el('line', { y1: m.t, y2: H - m.b, class: 'chart-crosshair', visibility: 'hidden' }, svg);
    const tip = document.createElement('div');
    tip.className = 'chart-tooltip';
    tip.hidden = true;
    holder.append(tip);
    const column = (k) => spec.columns?.find(c => c.key === k);
    const fmt = (k, v) => (column(k)?.format ? column(k).format(v) : (typeof v === 'number' ? formatTick(v) : String(v ?? '—')));
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', (event) => {
        const box = svg.getBoundingClientRect();
        const px = (event.clientX - box.left) * W / box.width;
        let best = rows[0], bestD = Infinity;
        for (const r of rows) { const d = Math.abs(xScale(r[spec.x]) - px); if (d < bestD) { bestD = d; best = r; } }
        const x = xScale(best[spec.x]);
        cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.setAttribute('visibility', 'visible');
        tip.replaceChildren();
        const head = document.createElement('div');
        head.className = 'chart-tooltip-head';
        head.textContent = `${spec.xLabel}: ${fmt(spec.x, best[spec.x])}`;
        tip.append(head);
        keys.forEach((k, i) => {
            const line = document.createElement('div');
            const keyMark = document.createElement('span');
            keyMark.className = `chart-key series-${i + 1}`;
            const value = document.createElement('strong');
            value.textContent = fmt(k, best[k]);
            line.append(keyMark, value, ` ${label(k)}`);
            tip.append(line);
        });
        tip.hidden = false;
        tip.style.left = `${Math.min(x / W * 100, 70)}%`;
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });
}
