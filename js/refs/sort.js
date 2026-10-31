// CPU references for the parallel-sort example: the built-in sort, the f32 <-> sortable-u32 key
// mapping (cs_encode_f32 / cs_decode_f32), and a seeded set of edge-case inputs.
// Precision: exact (integer keys; the key mapping is a bijection on 32-bit patterns).
// Cost: cpuSort is O(n log n) (TypedArray sort; 1e7 keys take on the order of a second in a
// browser); the key mapping is O(1) per key; edgeCases allocates about 12k keys.

import { mulberry32 as rng } from './random.js';

export { rng };

// f32 bit pattern -> u32 whose unsigned order matches the float order, and its inverse.
export const encodeF32Key = u => (u ^ ((u & 0x80000000) !== 0 ? 0xffffffff : 0x80000000)) >>> 0;
export const decodeF32Key = u => (u ^ ((u & 0x80000000) === 0 ? 0xffffffff : 0x80000000)) >>> 0;

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
