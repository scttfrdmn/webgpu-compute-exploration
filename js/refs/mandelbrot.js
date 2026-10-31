// CPU reference for the WASM Mandelbrot example: escape-iteration counts, a JS transcription of
// wasm/src/lib.rs compute_mandelbrot_cpu.
// Precision: float64 (the GPU shader works in f32, so counts near the set's boundary can differ).
// Cost: O(width·height·max_iterations) worst case; a 512² view at 256 iterations is at most ~6.7e7
// iterations (about a second in a browser, less where points escape early).

/**
 * Escape counts for c = { width, height, max_iterations, zoom, center_x, center_y } (the fields of
 * lib.rs MandelbrotConfig); returns a Float32Array of width·height iteration counts.
 */
export function mandelbrotEscape(c) {
    const { width, height, max_iterations } = c;
    const out = new Float32Array(width * height);
    const scale = 4.0 / (c.zoom * Math.min(width, height));
    for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
            const cx = c.center_x + (x - width / 2.0) * scale;
            const cy = c.center_y + (y - height / 2.0) * scale;
            let zx = 0, zy = 0, it = 0;
            while (zx * zx + zy * zy < 4.0 && it < max_iterations) {
                const t = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = t;
                it++;
            }
            out[y * width + x] = it;
        }
    return out;
}
