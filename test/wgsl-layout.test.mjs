// Tests for test/helpers/wgsl-layout.mjs, the WGSL memory-layout calculator the uniform-layout tests
// rely on. Expected offsets come from the WGSL spec's "Memory Layout" section (alignment/size tables
// and its worked struct examples), computed by hand.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseStructs, structLayout, typeLayout, decodeStruct } from './helpers/wgsl-layout.mjs';

test('scalar, vector and matrix alignment/size table', () => {
    const cases = {
        f32: [4, 4], u32: [4, 4], i32: [4, 4], f16: [2, 2],
        'vec2<f32>': [8, 8], vec2f: [8, 8], 'vec3<f32>': [16, 12], vec3u: [16, 12], 'vec4<i32>': [16, 16],
        'vec2<f16>': [4, 4], 'vec3<f16>': [8, 6], vec4h: [8, 8],
        'mat2x2<f32>': [8, 16], 'mat3x3<f32>': [16, 48], mat4x4f: [16, 64], 'mat4x3<f32>': [16, 64],
        'mat2x3<f32>': [16, 32], 'mat3x2<f32>': [8, 24],
        'atomic<u32>': [4, 4],
        'array<f32,4>': [4, 16], 'array<vec3<f32>, 2>': [16, 32], 'array<vec2f,3>': [8, 24],
    };
    for (const [type, [align, size]] of Object.entries(cases)) {
        const l = typeLayout(type, new Map());
        assert.deepEqual([l.align, l.size], [align, size], type);
    }
    assert.equal(typeLayout('array<vec3<f32>>', new Map()).stride, 16, 'runtime array stride rounds vec3 up');
});

test('vec3 padding: a scalar after a vec3 packs into its 4th lane', () => {
    const l = structLayout('struct S { a: vec3<f32>, b: f32, c: vec3<f32>, d: vec2<f32> }', 'S');
    assert.deepEqual(l.offsets, { a: 0, b: 12, c: 16, d: 32 });
    assert.equal(l.size, 48);
    assert.equal(l.align, 16);
});

test('WGSL spec example: struct A / struct B with nested arrays and @align/@size', () => {
    // From the WGSL spec, "Structure Member Layout" examples.
    const code = `
        struct A {                                     //             align(8)  size(24)
            u: f32,                                    // offset(0)   align(4)  size(4)
            v: f32,                                    // offset(4)   align(4)  size(4)
            w: vec2<f32>,                              // offset(8)   align(8)  size(8)
            x: f32                                     // offset(16)  align(4)  size(4)
            // -- implicit struct size padding --      // offset(20)            size(4)
        }
        struct B {                                     //             align(16) size(160)
            a: vec2<f32>,                              // offset(0)   align(8)  size(8)
            // -- implicit member alignment padding -- // offset(8)             size(8)
            b: vec3<f32>,                              // offset(16)  align(16) size(12)
            c: f32,                                    // offset(28)  align(4)  size(4)
            d: f32,                                    // offset(32)  align(4)  size(4)
            // -- implicit member alignment padding -- // offset(36)            size(4)
            e: A,                                      // offset(40)  align(8)  size(24)
            f: vec3<f32>,                              // offset(64)  align(16) size(12)
            // -- implicit member alignment padding -- // offset(76)            size(4)
            g: array<A, 3>,    // element stride 24       offset(80)  align(8)  size(72)
            h: i32                                     // offset(152) align(4)  size(4)
            // -- implicit struct size padding --      // offset(156)           size(4)
        }
        struct C {
            @align(16) x: f32,
            @size(20) y: f32,
            z: u32,
        }`;
    const a = structLayout(code, 'A');
    assert.deepEqual(a.offsets, { u: 0, v: 4, w: 8, x: 16 });
    assert.deepEqual([a.align, a.size], [8, 24]);
    const b = structLayout(code, 'B');
    assert.deepEqual(b.offsets, { a: 0, b: 16, c: 28, d: 32, e: 40, f: 64, g: 80, h: 152 });
    assert.deepEqual([b.align, b.size], [16, 160]);
    const c = structLayout(code, 'C');
    assert.deepEqual(c.offsets, { x: 0, y: 4, z: 24 });
    assert.deepEqual([c.align, c.size], [16, 32]);
});

test('matrices and nested structs', () => {
    const code = `struct Inner { p: vec3f, q: f32 }
        struct Outer { s: f32, m: mat4x4<f32>, t: f32, n: mat3x3f, i: Inner, arr: array<Inner, 2>, last: u32 }`;
    const l = structLayout(code, 'Outer');
    assert.deepEqual(l.offsets, { s: 0, m: 16, t: 80, n: 96, i: 144, arr: 160, last: 192 });
    assert.equal(l.size, 208);
});

test('parser tolerates comments, attributes, shader IO structs and trailing commas', () => {
    const structs = parseStructs(`
        // struct Fake { nope: f32 }
        /* struct AlsoFake { x: u32 } */
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) id: u32,
        }`);
    assert.deepEqual([...structs.keys()], ['VertexOutput']);
    const l = structLayout(`struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) id: u32, }`, 'VertexOutput');
    assert.deepEqual(l.offsets, { position: 0, id: 16 });
});

test('decodeStruct reads u32 as integers and f32 as floats at computed offsets', () => {
    const l = structLayout('struct P { n: u32, r: f32, v: vec2<f32>, k: i32, arr: array<u32, 2> }', 'P');
    const buf = new ArrayBuffer(l.size);
    const f = new Float32Array(buf), u = new Uint32Array(buf), i = new Int32Array(buf);
    u[0] = 1000; f[1] = 2.5; f[2] = 3; f[3] = -4; i[4] = -7; u[5] = 11; u[6] = 12;
    const d = decodeStruct(buf, l);
    assert.deepEqual(d, { n: 1000, r: 2.5, v: [3, -4], k: -7, arr: [11, 12] });
    // The bug class this guards: a u32 written through a Float32Array view.
    f[0] = 1000;
    assert.equal(decodeStruct(buf, l).n, 1148846080);
});
