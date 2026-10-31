// Parallel-sort references shared by the CPU suite (test/parallel-sort.test.mjs) and the GPU kernel
// tests (test/gpu/parallel-sort.gpu.test.mjs).

export function rng(seed) {   // mulberry32
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ---- f32 key conversion replica (cs_encode_f32 / cs_decode_f32) ----
export const encode = u => (u ^ ((u & 0x80000000) !== 0 ? 0xffffffff : 0x80000000)) >>> 0;
export const decode = u => (u ^ ((u & 0x80000000) === 0 ? 0xffffffff : 0x80000000)) >>> 0;

/** Sorted / reversed / all-equal / PAD_KEY / few-unique / random u32 inputs, incl. non-powers of two. */
export function edgeCases(seed) {
    const r = rng(seed);
    const rand = n => Uint32Array.from({ length: n }, () => (r() * 0x100000000) >>> 0);
    const cases = {
        empty: new Uint32Array(0),
        one: Uint32Array.of(7),
        two: Uint32Array.of(9, 3),
        three: Uint32Array.of(5, 0xffffffff, 0),
        sorted1000: Uint32Array.from({ length: 1000 }, (_, i) => i),
        reversed1023: Uint32Array.from({ length: 1023 }, (_, i) => 1023 - i),
        allEqual513: new Uint32Array(513).fill(42),
        withPadKey600: Uint32Array.from({ length: 600 }, (_, i) => (i % 3 ? i : 0xffffffff)),
        fewUnique2049: Uint32Array.from({ length: 2049 }, () => Math.floor(r() * 4)),
        random511: rand(511),
        random512: rand(512),
        random1025: rand(1025),
        random4097: rand(4097),
    };
    return cases;
}

export const cpuSort = a => Uint32Array.from(a).sort();
