// Builds an opt-in Markdown "device report" for portability data (issue #53). The report is
// only shown in the page and copied to the clipboard when the user asks; nothing is sent anywhere.
//
// It reads browser/adapter/device properties and a small whitelist of plain numeric/boolean
// fields from the example instances. It never calls methods on the instances.

import { loadSourceInfo } from './source-info.js';

// Instance fields that are safe to read and useful for comparing devices. Only values whose
// type matches are reported, so a missing field or an accidental method is simply skipped.
const INSTANCE_FIELDS = [
    ['fps', 'number'],
    ['running', 'boolean'],
    ['lost', 'boolean'],
    ['steps', 'number'],
    ['stepsPerFrame', 'number'],
    ['iteration', 'number'],
    ['iterations', 'number'],
    ['generation', 'number'],
    ['frameCount', 'number'],
    ['sampleCount', 'number'],
    ['epoch', 'number'],
    ['numParticles', 'number'],
    ['numBodies', 'number'],
    ['numBoids', 'number'],
    ['numAtoms', 'number'],
    ['gridSize', 'number'],
    ['width', 'number'],
    ['height', 'number'],
];

// One-shot examples write their result into these output panels.
const ONE_SHOT_OUTPUTS = [
    ['Vector Addition', 'output-vector-add'],
    ['Matrix Multiplication', 'output-matrix-mul'],
    ['Gaussian Blur', 'output-image-blur'],
    ['WASM Mandelbrot', 'output-wasm-mandelbrot'],
];
const ONE_SHOT_LINES = 8;

// Fallback limit names for browsers whose GPUSupportedLimits attributes are not enumerable.
const KNOWN_LIMITS = [
    'maxTextureDimension1D', 'maxTextureDimension2D', 'maxTextureDimension3D', 'maxTextureArrayLayers',
    'maxBindGroups', 'maxBindGroupsPlusVertexBuffers', 'maxBindingsPerBindGroup',
    'maxDynamicUniformBuffersPerPipelineLayout', 'maxDynamicStorageBuffersPerPipelineLayout',
    'maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage',
    'maxStorageTexturesPerShaderStage', 'maxUniformBuffersPerShaderStage', 'maxUniformBufferBindingSize',
    'maxStorageBufferBindingSize', 'minUniformBufferOffsetAlignment', 'minStorageBufferOffsetAlignment',
    'maxVertexBuffers', 'maxBufferSize', 'maxVertexAttributes', 'maxVertexBufferArrayStride',
    'maxInterStageShaderVariables', 'maxColorAttachments', 'maxColorAttachmentBytesPerSample',
    'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension',
];

const MAX_EXPERIMENT_ROWS = 30;
const MAX_ERRORS = 20;

// Swallow exceptions from any property read: a report should never break the page.
function safe(fn, fallback = undefined) {
    try {
        const value = fn();
        return value === undefined ? fallback : value;
    } catch {
        return fallback;
    }
}

async function safeAsync(fn, fallback) {
    try {
        return (await fn()) ?? fallback;
    } catch {
        return fallback;
    }
}

function cell(value) {
    return String(value).replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}

function table(headers, rows) {
    if (rows.length === 0) return '_none_\n';
    const lines = [
        `| ${headers.join(' | ')} |`,
        `| ${headers.map(() => '---').join(' | ')} |`,
        ...rows.map(row => `| ${row.map(cell).join(' | ')} |`),
    ];
    return lines.join('\n') + '\n';
}

function fence(text) {
    return '```\n' + String(text).replace(/```/g, "'''") + '\n```\n';
}

function stripHtml(html) {
    return String(html)
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/<\/(p|div|tr|li|h\d)>/gi, '\n')
        .replace(/<[^>]*>/g, '')
        .replace(/&nbsp;/g, ' ')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&amp;/g, '&');
}

function formatNumber(value) {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
}


function limitRows(adapterLimits, deviceLimits) {
    const names = new Set();
    if (adapterLimits) {
        // GPUSupportedLimits exposes its attributes as enumerable getters on the prototype.
        safe(() => { for (const key in adapterLimits) names.add(key); });
    }
    if (names.size === 0) KNOWN_LIMITS.forEach(name => names.add(name));
    const rows = [];
    for (const name of [...names].sort()) {
        const a = safe(() => adapterLimits?.[name]);
        const d = safe(() => deviceLimits?.[name]);
        if (typeof a !== 'number' && typeof d !== 'number') continue;
        rows.push([name, typeof a === 'number' ? a : '—', typeof d === 'number' ? d : '—']);
    }
    return rows;
}

function setLikeToSortedArray(setLike) {
    return safe(() => Array.from(setLike).map(String).sort(), []);
}

function exampleName(doc, id) {
    const heading = safe(() => doc?.getElementById(id)?.closest('.example')?.querySelector('h3')?.textContent);
    return heading ? heading.replace(/\s+/g, ' ').trim() : id;
}

function instanceFields(instance) {
    const fields = [];
    if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return fields;
    for (const [key, type] of INSTANCE_FIELDS) {
        const value = safe(() => instance[key]);
        if (typeof value !== type) continue;
        if (type === 'number' && !Number.isFinite(value)) continue;
        fields.push(`${key}=${type === 'number' ? formatNumber(value) : value}`);
    }
    return fields;
}

/**
 * @param {object} options
 * @param {GPUAdapter} [options.adapter]
 * @param {GPUDevice} [options.device]
 * @param {Map<string, object>} [options.instances]  canvas id -> resolved example instance
 * @param {string[]} [options.errors]  recent WebGPU console errors
 * @param {{reason?: string, message?: string}|null} [options.deviceLost]
 * @param {string} [options.unavailableReason]  why WebGPU is not in use, if it is not
 * @param {object} [options.env]  overrides for navigator/window/document/fetch/location (tests)
 * @returns {Promise<string>} Markdown
 */
export async function buildDeviceReport({
    adapter = null,
    device = null,
    instances = new Map(),
    errors = [],
    deviceLost = null,
    unavailableReason = '',
    experiments = [],
    env = {},
} = {}) {
    const nav = env.navigator ?? globalThis.navigator;
    const win = env.window ?? globalThis.window ?? globalThis;
    const doc = env.document ?? globalThis.document;
    const loc = env.location ?? globalThis.location;
    const fetchFn = env.fetch ?? globalThis.fetch;
    const now = env.now ?? new Date();

    const out = [];
    const push = (...lines) => out.push(...lines);

    push('## WebGPU device report', '');
    push('_Generated on request by the page. Nothing is sent automatically; review before sharing._', '');

    // Page and browser
    push('### Page and browser', '');
    const ua = safe(() => nav.userAgent, 'unknown');
    const uaData = safe(() => nav.userAgentData);
    const brands = safe(() => uaData.brands.map(b => `${b.brand} ${b.version}`).join(', '));
    const screenSize = safe(() => `${win.screen.width}×${win.screen.height}`);
    const source = await safeAsync(() => loadSourceInfo(fetchFn), { version: 'unknown', commit: 'local' });
    const pageRows = [
        ['Page version', source.version],
        ['Source commit', source.commit],
        ['Page origin', safe(() => loc.origin, 'unknown')],
        ['Date', safe(() => now.toISOString(), 'unknown')],
        ['User agent', ua],
    ];
    if (brands) pageRows.push(['UA brands', brands]);
    const platform = safe(() => uaData.platform);
    if (platform) pageRows.push(['UA platform', platform]);
    const mobile = safe(() => uaData.mobile);
    if (typeof mobile === 'boolean') pageRows.push(['UA mobile', mobile]);
    const dpr = safe(() => win.devicePixelRatio);
    if (typeof dpr === 'number') pageRows.push(['devicePixelRatio', dpr]);
    if (screenSize) pageRows.push(['Screen', screenSize]);
    push(table(['Field', 'Value'], pageRows));

    // WebGPU availability
    const gpu = safe(() => nav.gpu);
    push('### WebGPU', '');
    if (!gpu) {
        push('**WebGPU is not available in this browser** (`navigator.gpu` is undefined).', '');
    } else if (!adapter) {
        push(`**No WebGPU adapter in use.**${unavailableReason ? ` ${unavailableReason}` : ''}`, '');
    }
    if (gpu) {
        const gpuRows = [];
        const format = safe(() => gpu.getPreferredCanvasFormat());
        if (format) gpuRows.push(['Preferred canvas format', format]);
        const wgsl = safe(() => gpu.wgslLanguageFeatures);
        if (wgsl) {
            const list = setLikeToSortedArray(wgsl);
            gpuRows.push(['WGSL language features', list.length ? list.join(', ') : '(none)']);
        }
        if (gpuRows.length) push(table(['Field', 'Value'], gpuRows));
    }
    if (deviceLost) {
        const reason = safe(() => deviceLost.reason, 'unknown');
        const message = safe(() => deviceLost.message, '');
        push(`**Device lost this session** (reason: \`${reason}\`)${message ? `: ${message}` : ''}`, '');
    }

    // Adapter
    if (adapter) {
        push('### Adapter', '');
        const info = safe(() => adapter.info);
        const adapterRows = [];
        if (info) {
            for (const key of ['vendor', 'architecture', 'device', 'description']) {
                const value = safe(() => info[key]);
                adapterRows.push([key, value ? value : '(empty)']);
            }
            for (const key of ['subgroupMinSize', 'subgroupMaxSize']) {
                const value = safe(() => info[key]);
                if (typeof value === 'number') adapterRows.push([key, value]);
            }
        } else {
            adapterRows.push(['info', '(adapter.info not supported)']);
        }
        const fallback = safe(() => info?.isFallbackAdapter ?? adapter.isFallbackAdapter);
        if (typeof fallback === 'boolean') adapterRows.push(['isFallbackAdapter', fallback]);
        push(table(['Field', 'Value'], adapterRows));

        const features = setLikeToSortedArray(safe(() => adapter.features, []));
        push(`#### Adapter features (${features.length})`, '');
        push(table(['Feature'], features.map(f => [f])));

        push('#### Limits', '');
        push('The page requests its device with **default limits**, so the device column is what examples ' +
            'actually get; the adapter column is what this GPU could offer.', '');
        push(table(['Limit', 'Adapter', 'Device (default)'],
            limitRows(safe(() => adapter.limits), safe(() => device?.limits))));
    }

    // Interactive examples
    push('### Interactive examples initialized this session', '');
    const exampleRows = [];
    for (const [id, instance] of safe(() => [...instances], [])) {
        const fields = instanceFields(instance);
        exampleRows.push([exampleName(doc, id), instance ? (fields.join(', ') || '(no readable fields)')
            : '(no instance — failed to initialize?)']);
    }
    push(table(['Example', 'State'], exampleRows));
    if (exampleRows.length) push('_fps is the last value each example sampled (0 if it has not been started)._', '');

    // One-shot examples
    push('### One-shot results', '');
    let anyOneShot = false;
    for (const [name, id] of ONE_SHOT_OUTPUTS) {
        const el = safe(() => doc?.getElementById(id));
        if (!el) continue;
        const raw = safe(() => el.innerText ?? el.textContent) ?? stripHtml(safe(() => el.innerHTML, ''));
        const lines = String(raw ?? '').split('\n').map(l => l.trimEnd()).filter(l => l.trim());
        if (lines.length === 0) continue;
        anyOneShot = true;
        push(`**${name}**`, '');
        push(fence(lines.slice(0, ONE_SHOT_LINES).join('\n')));
    }
    if (!anyOneShot) push('_none run_', '');

    // Measure-tab experiments run this session: the latest result record (schema v1,
    // schema/experiment-result.v1.json) per experiment, as a table of medians and the full record as JSON
    const results = safe(() => experiments.slice(), []);
    push(`### Measure experiments this session (${results.length})`, '');
    if (!results.length) push('_none run_', '');
    else push('_Each record below is a result file (schema `experiment-result.v1`); paste one into a `.json` file to '
        + 'compare it in the Measure tab._', '');
    for (const r of results) {
        safe(() => {
            const settings = Object.entries(r.settings ?? {}).map(([k, v]) => `${k}=${v}`).join(', ');
            const checks = (r.verification ?? []).map(c => c.status);
            const verified = checks.includes('fail') ? 'verification FAILED'
                : checks.includes('pass') ? `verified (${checks.filter(s => s === 'pass').length} checks pass)` : 'not verified';
            push(`**${r.experiment.exampleId} · ${r.experiment.title}** (${r.outcome?.status}; ${verified}`
                + `${settings ? `; ${settings}` : ''})`, '');
            const cols = (r.columns ?? []).map(c => ({ key: c.key, label: c.label }));
            const cell = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toPrecision(4)) : String(v ?? ''));
            push(`| ${cols.map(c => c.label).join(' | ')} |`, `| ${cols.map(() => '---').join(' | ')} |`);
            for (const row of (r.rows ?? []).slice(0, MAX_EXPERIMENT_ROWS)) push(`| ${cols.map(c => cell(row[c.key])).join(' | ')} |`);
            push('', '<details><summary>Result record (JSON)</summary>', '');
            push('```json', JSON.stringify(r).replace(/```/g, "'''"), '```', '', '</details>', '');
        });
    }

    // Errors
    const recent = safe(() => errors.slice(-MAX_ERRORS), []);
    push(`### WebGPU errors this session (${recent.length})`, '');
    push(recent.length ? fence(recent.join('\n')) : '_none_\n');

    return out.join('\n').replace(/\n{3,}/g, '\n\n').trimEnd() + '\n';
}

export { stripHtml };
