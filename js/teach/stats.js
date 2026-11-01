// Summary statistics for repeated measurements (the Measure protocol, js/teach/measure.js) and the
// result records built from them (js/teach/result-record.js). Pure; tested in test/teach-stats.test.mjs.
//
// Conventions:
//   - Non-numbers and non-finite values (null, undefined, NaN, ±Infinity) are dropped before anything is
//     computed; `n` counts what is left.
//   - Quantiles interpolate linearly between order statistics: for sorted x₀ … xₙ₋₁ and probability p,
//     h = (n − 1)·p and Q(p) = x⌊h⌋ + (h − ⌊h⌋)(x⌊h⌋₊₁ − x⌊h⌋). This is Hyndman & Fan's type 7, the default
//     of R's quantile() and NumPy's percentile(). The median is Q(0.5) (the mean of the middle two for even n).
//   - Spread is the interquartile range q1 … q3 = Q(0.25) … Q(0.75). With the default 5 repeats that is the
//     2nd and 4th smallest values, so a single outlier moves neither the median nor the range.
//   - std is the sample standard deviation (n − 1 in the denominator); cv = std / |mean|. Both are null
//     for n < 2 (no variability can be estimated from one value), and cv is null when the mean is 0.

export const SPREAD = 'iqr';
export const SPREAD_LABEL = 'interquartile range (q1–q3)';

export function finiteValues(values) {
    return (values ?? []).filter(v => typeof v === 'number' && Number.isFinite(v));
}

// Q(p) of an ascending array (type 7); null when empty
export function quantileSorted(sorted, p) {
    if (!sorted.length) return null;
    if (!(p >= 0 && p <= 1)) throw new RangeError(`quantile probability ${p} is outside 0 … 1`);
    const h = (sorted.length - 1) * p;
    const lo = Math.floor(h);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

export function quantile(values, p) {
    return quantileSorted(finiteValues(values).sort((a, b) => a - b), p);
}

const EMPTY = Object.freeze({ n: 0, median: null, min: null, max: null, q1: null, q3: null, mean: null, std: null, cv: null });

// { n, median, min, max, q1, q3, mean, std, cv }; every statistic is null when n = 0
export function summarize(values) {
    const v = finiteValues(values).sort((a, b) => a - b);
    const n = v.length;
    if (!n) return { ...EMPTY };
    const mean = v.reduce((s, x) => s + x, 0) / n;
    const std = n > 1 ? Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1)) : null;
    return {
        n,
        median: quantileSorted(v, 0.5),
        min: v[0],
        max: v[n - 1],
        q1: quantileSorted(v, 0.25),
        q3: quantileSorted(v, 0.75),
        mean,
        std,
        cv: std === null || mean === 0 ? null : std / Math.abs(mean),
    };
}

// The row-key convention for a quantity with spread: `<key>` is the median, `<key>Lo` / `<key>Hi` are
// q1 / q3. A missing summary (e.g. GPU time without timestamp-query) gives three nulls.
export function withSpread(key, stats) {
    const s = stats ?? EMPTY;
    return { [key]: s.median ?? null, [`${key}Lo`]: s.q1 ?? null, [`${key}Hi`]: s.q3 ?? null };
}

export const loKey = (key) => `${key}Lo`;
export const hiKey = (key) => `${key}Hi`;

// [lo, hi] when a row carries a finite spread for `key`, else null
export function rangeOf(row, key) {
    const lo = row?.[loKey(key)], hi = row?.[hiKey(key)];
    return typeof lo === 'number' && Number.isFinite(lo) && typeof hi === 'number' && Number.isFinite(hi) ? [lo, hi] : null;
}

// Whether any row has a spread for `key` (the table and CSV then show it)
export function hasRange(rows, key) {
    return (rows ?? []).some(r => rangeOf(r, key) !== null);
}
