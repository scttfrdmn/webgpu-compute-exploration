// Seeded PRNG shared by the CPU references (js/refs/*) and their tests: mulberry32, a 32-bit
// generator returning floats in [0, 1). Not a reference itself; it makes reference inputs
// reproducible (same seed, same sequence, in any JS engine). Precision: exact 32-bit integer
// arithmetic. Cost: O(1) per draw.

export function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
