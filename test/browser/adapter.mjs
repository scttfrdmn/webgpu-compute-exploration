// Software-adapter recognition for the browser suite (test/browser/, #77). Pure: no side effects, no
// Playwright, no DOM, so the CPU suite (test/browser-guard.test.mjs) can import it.

// Mesa lavapipe reports itself as "llvmpipe"; Chrome's bundled CPU Vulkan driver is SwiftShader
export const SOFTWARE = /llvmpipe|lavapipe|swiftshader/i;

// Runs inside the page (passed to page.evaluate): describes the adapter the site itself gets from
// navigator.gpu.requestAdapter() with no options, and checks that a device can be created on it.
export async function probeAdapterInPage() {
    if (!navigator.gpu) return { ok: false, why: 'navigator.gpu is undefined' };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, why: 'requestAdapter() returned null' };
    const info = adapter.info ?? {};
    const out = {
        ok: true,
        vendor: info.vendor ?? '', architecture: info.architecture ?? '',
        device: info.device ?? '', description: info.description ?? '',
        isFallbackAdapter: Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter),
        features: [...adapter.features].sort(),
    };
    try {
        const device = await adapter.requestDevice();
        device.destroy();
    } catch (error) {
        return { ...out, ok: false, why: `requestDevice() failed: ${error.message}` };
    }
    return out;
}

// True only for a CPU (software) adapter: a fallback adapter, or one whose strings name a CPU driver
export function isSoftwareAdapter(info) {
    if (!info || !info.ok) return false;
    if (info.isFallbackAdapter) return true;
    return SOFTWARE.test([info.vendor, info.architecture, info.device, info.description].join(' '));
}

export function describeAdapter(info) {
    if (!info) return '<none>';
    if (!('vendor' in info)) return `unavailable (${info.why})`;
    const fields = ['vendor', 'architecture', 'device', 'description']
        .map(k => `${k}=${JSON.stringify(info[k] ?? '')}`).join(' ');
    const ts = info.features?.includes('timestamp-query') ? 'yes' : 'no';
    return `${fields} fallback=${info.isFallbackAdapter} timestamp-query=${ts}${info.ok ? '' : ` (${info.why})`}`;
}
