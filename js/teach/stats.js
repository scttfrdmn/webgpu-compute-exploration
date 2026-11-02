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

// Two-sided 95% Student t quantiles for small degrees of freedom (df ≥ 30 uses 1.96 + 2.4/df)
const T975 = [NaN, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045];
export function t975(df) {
    if (!(df >= 1)) return NaN;
    return df < T975.length ? T975[Math.floor(df)] : 1.96 + 2.4 / df;
}

// Least-squares fit of log10(y) = slope · log10(x) + intercept, i.e. y ∝ x^slope. Points with non-positive or
// non-finite x or y are skipped. Returns the slope with its standard error and a 95% confidence interval
// (Student t, n − 2 degrees of freedom), the intercept and R²; slope fields are null with fewer than 3 points
// for the interval (2 points give a slope but no uncertainty).
export function fitPowerLaw(xs, ys) {
    const pts = [];
    for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
        const x = xs[i], y = ys[i];
        if (Number.isFinite(x) && Number.isFinite(y) && x > 0 && y > 0) pts.push([Math.log10(x), Math.log10(y)]);
    }
    const n = pts.length;
    const empty = { n, slope: null, intercept: null, slopeStdErr: null, ci95: null, r2: null };
    if (n < 2) return empty;
    const mx = pts.reduce((a, p) => a + p[0], 0) / n, my = pts.reduce((a, p) => a + p[1], 0) / n;
    let sxx = 0, sxy = 0, syy = 0;
    for (const [x, y] of pts) { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - my); syy += (y - my) ** 2; }
    if (sxx === 0) return empty;
    const slope = sxy / sxx, intercept = my - slope * mx;
    const ssRes = pts.reduce((a, [x, y]) => a + (y - (intercept + slope * x)) ** 2, 0);
    const r2 = syy === 0 ? 1 : 1 - ssRes / syy;
    if (n < 3) return { n, slope, intercept, slopeStdErr: null, ci95: null, r2 };
    const slopeStdErr = Math.sqrt(ssRes / (n - 2) / sxx);
    const half = t975(n - 2) * slopeStdErr;
    return { n, slope, intercept, slopeStdErr, ci95: [slope - half, slope + half], r2 };
}

// Does a predicted exponent fall inside the fit's 95% interval (with an optional extra tolerance for model
// approximations, e.g. ±0.15 when the prediction itself is only asymptotic)? null when there is no interval.
export function slopeAgrees(fit, predicted, tolerance = 0) {
    if (!fit.ci95) return null;
    return predicted >= fit.ci95[0] - tolerance && predicted <= fit.ci95[1] + tolerance;
}
