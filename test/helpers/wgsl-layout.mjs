// WGSL host-shareable memory layout calculator (WGSL spec §13.4 "Memory Layout").
//
// Parses `struct` declarations out of a WGSL source string and computes each member's byte offset,
// size and alignment, so tests can check that the bytes a JS uniform writer produces line up with
// the struct the shader declares. Supports scalars (f32/i32/u32/f16/bool), vecN<T> and the vecNf /
// vecNu / vecNi / vecNh shorthands, matCxR<T> and matCxRf, fixed and runtime-sized arrays, atomic<T>,
// nested structs, and the @align(n) / @size(n) member attributes.
//
// The layout computed is the storage/uniform layout; it does not enforce the extra uniform-address-
// space restrictions (16-byte array strides etc.) — naga (tools/validate-wgsl.mjs) does that.

const SCALARS = {
    f32: { size: 4, align: 4, kind: 'f32' },
    i32: { size: 4, align: 4, kind: 'i32' },
    u32: { size: 4, align: 4, kind: 'u32' },
    f16: { size: 2, align: 2, kind: 'f16' },
    bool: { size: 4, align: 4, kind: 'u32' },
};
const SHORTHAND = { f: 'f32', i: 'i32', u: 'u32', h: 'f16' };

const roundUp = (k, n) => Math.ceil(n / k) * k;

function stripComments(code) {
    return code.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

// Split on commas that are not inside <...> or (...).
function splitTopLevel(s, sep = ',') {
    const out = [];
    let depth = 0, cur = '';
    for (const ch of s) {
        if (ch === '<' || ch === '(') depth++;
        else if (ch === '>' || ch === ')') depth--;
        if (ch === sep && depth === 0) { out.push(cur); cur = ''; } else cur += ch;
    }
    if (cur.trim()) out.push(cur);
    return out.map(x => x.trim()).filter(Boolean);
}

/** Parse every struct in `code`: returns Map name -> [{ name, type, align?, size? }]. */
export function parseStructs(code) {
    const src = stripComments(code);
    const structs = new Map();
    const re = /struct\s+(\w+)\s*\{([^}]*)\}/g;
    let m;
    while ((m = re.exec(src))) {
        const members = [];
        for (const raw of splitTopLevel(m[2])) {
            const attrs = {};
            let rest = raw;
            let a;
            const attrRe = /^@(\w+)(?:\s*\(([^)]*)\))?\s*/;
            while ((a = attrRe.exec(rest))) {
                attrs[a[1]] = a[2];
                rest = rest.slice(a[0].length);
            }
            const colon = rest.indexOf(':');
            if (colon < 0) throw new Error(`cannot parse struct member "${raw}" in ${m[1]}`);
            const member = { name: rest.slice(0, colon).trim(), type: rest.slice(colon + 1).trim().replace(/\s+/g, '') };
            if (attrs.align !== undefined) member.alignAttr = Number(attrs.align);
            if (attrs.size !== undefined) member.sizeAttr = Number(attrs.size);
            if (attrs.builtin !== undefined || attrs.location !== undefined) member.io = true;
            members.push(member);
        }
        structs.set(m[1], members);
    }
    return structs;
}

/** Layout of a WGSL type string: { size, align, kind, ... }. `structs` is a parseStructs() map. */
export function typeLayout(type, structs) {
    type = type.replace(/\s+/g, '');
    if (SCALARS[type]) return { ...SCALARS[type], type };
    let m;
    if ((m = /^atomic<(\w+)>$/.exec(type))) return { ...typeLayout(m[1], structs), type };
    if ((m = /^vec([234])([fiuh])$/.exec(type))) return typeLayout(`vec${m[1]}<${SHORTHAND[m[2]]}>`, structs);
    if ((m = /^vec([234])<(\w+)>$/.exec(type))) {
        const n = Number(m[1]);
        const s = SCALARS[m[2]];
        const size = n * s.size;
        const align = (n === 3 ? 4 : n) * s.size;
        return { size, align, kind: 'vec', n, scalar: m[2], type };
    }
    if ((m = /^mat([234])x([234])([fh])$/.exec(type))) return typeLayout(`mat${m[1]}x${m[2]}<${SHORTHAND[m[3]]}>`, structs);
    if ((m = /^mat([234])x([234])<(\w+)>$/.exec(type))) {
        const cols = Number(m[1]), rows = Number(m[2]);
        const col = typeLayout(`vec${rows}<${m[3]}>`, structs);
        const stride = roundUp(col.align, col.size);
        return { size: cols * stride, align: col.align, kind: 'mat', cols, rows, stride, column: col, type };
    }
    if (type.startsWith('array<') && type.endsWith('>')) {
        const parts = splitTopLevel(type.slice(6, -1));
        const elem = typeLayout(parts[0], structs);
        const stride = roundUp(elem.align, elem.size);
        const count = parts.length > 1 ? Number(parts[1]) : null;
        if (parts.length > 1 && !Number.isFinite(count)) throw new Error(`array count must be a literal: ${type}`);
        return { size: count === null ? stride : count * stride, align: elem.align, kind: 'array', elem, stride, count, type };
    }
    if (structs && structs.has(type)) return structLayoutFrom(structs, type);
    throw new Error(`unknown WGSL type "${type}"`);
}

function structLayoutFrom(structs, name) {
    const members = structs.get(name);
    if (!members) throw new Error(`struct ${name} not found`);
    let offset = 0, align = 0;
    const fields = [];
    for (const mem of members) {
        const t = typeLayout(mem.type, structs);
        const a = mem.alignAttr ?? t.align;
        const s = mem.sizeAttr ?? t.size;
        offset = roundUp(a, offset);
        fields.push({ name: mem.name, offset, size: s, align: a, layout: t });
        offset += s;
        align = Math.max(align, a);
    }
    return { kind: 'struct', name, type: name, fields, align, size: roundUp(align, offset),
        offsets: Object.fromEntries(fields.map(f => [f.name, f.offset])) };
}

/** Layout of struct `name` declared in WGSL source `code`. */
export function structLayout(code, name) {
    return structLayoutFrom(parseStructs(code), name);
}

function decodeValue(view, offset, t) {
    switch (t.kind) {
        case 'f32': return view.getFloat32(offset, true);
        case 'u32': return view.getUint32(offset, true);
        case 'i32': return view.getInt32(offset, true);
        case 'f16': return view.getFloat16 ? view.getFloat16(offset, true) : view.getUint16(offset, true);
        case 'vec': {
            const s = SCALARS[t.scalar];
            return Array.from({ length: t.n }, (_, i) => decodeValue(view, offset + i * s.size, s));
        }
        case 'mat':
            return Array.from({ length: t.cols }, (_, c) => decodeValue(view, offset + c * t.stride, t.column));
        case 'array': {
            const n = t.count ?? 1;
            return Array.from({ length: n }, (_, i) => decodeValue(view, offset + i * t.stride, t.elem));
        }
        case 'struct':
            return Object.fromEntries(t.fields.map(f => [f.name, decodeValue(view, offset + f.offset, f.layout)]));
        default: throw new Error(`cannot decode ${t.type}`);
    }
}

/** Decode bytes (ArrayBuffer / TypedArray / DataView) as `layout` (from structLayout) at byteOffset. */
export function decodeStruct(bytes, layout, byteOffset = 0) {
    const view = bytes instanceof DataView ? bytes
        : ArrayBuffer.isView(bytes) ? new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
        : new DataView(bytes);
    if (view.byteLength < byteOffset + layout.size) {
        // Allow a short buffer only if every field fits; otherwise report the mismatch clearly.
        const last = layout.fields[layout.fields.length - 1];
        if (view.byteLength < byteOffset + last.offset + last.size) {
            throw new Error(`buffer of ${view.byteLength} bytes is too small for struct ${layout.name} (${layout.size} bytes)`);
        }
    }
    return decodeValue(view, byteOffset, layout);
}

/** Find the WGSL code of the module named `name` in an example's wgslModules() list. */
export function moduleCode(mod, name) {
    const found = mod.wgslModules().find(m => m.name === name);
    if (!found) throw new Error(`no WGSL module named "${name}"`);
    return found.code;
}
