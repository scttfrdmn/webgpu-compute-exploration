// Small, safe Markdown renderer for the Understand tab (#54). No dependencies, no DOM APIs.
//
// Blocks: ATX headings (# … ######, with unique id slugs), paragraphs, fenced code (``` or ~~~, optional
// language → <pre><code class="language-x"> plus a label), lists (-, *, +, 1. / 1); nesting by
// indentation; tight lists drop the <p>), GitHub tables (with an alignment row), blockquotes, horizontal
// rules, display math ($$ … $$ on its own lines, possibly multi-line), and callouts: a blockquote whose
// first line starts with **GPU:**, **Numerics:**, **Try this:**, **Note:** or **Warning:** renders as
// <aside class="callout callout-gpu|numerics|try|note|warning"> with the label in a <strong>.
//
// Inline: `code`, **bold** / __bold__, *em* / _em_ (underscores inside words stay literal: snake_case),
// [links](href "title"), ![images](src), inline math $…$, backslash escapes, hard line breaks (two
// trailing spaces or a trailing backslash).
//
// Inline math rule (so prices don't turn into math):
//   - an opening $ must be followed by a non-space character;
//   - an opening $ followed by a number and then a space, punctuation or the end ("$5 each", "$1,000.")
//     is a literal dollar sign;
//   - the closing $ must be preceded by a non-space character and not followed by a digit;
//   - \$ is always a literal dollar; nothing inside `code` is math. $$…$$ inside a line is display math.
//   If no valid closing $ exists, the $ is literal.
//
// Security: every piece of source text is HTML-escaped; there is no raw HTML passthrough; attribute values
// are escaped; link and image URLs must be http:, https:, #fragment or relative (checked after
// resolveLink), anything else (javascript:, data:, vbscript:, mailto:, …) is dropped and only the text kept.

import { texToMathML, texErrors } from './tex-mathml.js';

const CALLOUTS = { 'GPU': 'gpu', 'Numerics': 'numerics', 'Try this': 'try', 'Note': 'note', 'Warning': 'warning' };

export function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---- URL policy ----------------------------------------------------------------------------------------
export function safeUrl(url) {
    if (typeof url !== 'string') return null;
    const trimmed = url.trim();
    if (trimmed === '') return null;
    // Browsers ignore ASCII whitespace/control characters inside a scheme ("java\tscript:"), so must we.
    const squashed = trimmed.replace(/[\u0000- \u007f-\u009f]/g, '');
    if (/^[a-z][a-z0-9+.-]*:/i.test(squashed)) return /^https?:/i.test(squashed) ? trimmed : null;
    if (/^[\\/]{2}/.test(squashed) || squashed.startsWith('/\\') || squashed.startsWith('\\')) return null; // //host, \\host
    return trimmed;
}

// ---- slugs ---------------------------------------------------------------------------------------------
function plainText(s) {
    return s
        .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
        .replace(/\\(.)/g, '$1')
        .replace(/[`*$]/g, '')
        .replace(/(^|\W)_+|_+(?=\W|$)/g, '$1')
        .trim();
}

function slugify(text) {
    return text.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
        .replace(/[^a-z0-9\s-]/g, '').trim().replace(/[\s-]+/g, '-').replace(/^-+|-+$/g, '') || 'section';
}

// ---- line classification -------------------------------------------------------------------------------
const FENCE_RE = /^( {0,3})(`{3,}|~{3,})\s*([^`\s]*)[^`]*$/;
const HEADING_RE = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const HR_RE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const LIST_RE = /^( *)([-*+]|\d{1,9}[.)])( +|$)(.*)$/;
const TABLE_DELIM_RE = /^ *\|? *:?-+:? *(\| *:?-+:? *)*\|? *$/;
const isBlank = (l) => /^\s*$/.test(l);
const indentOf = (l) => /^ */.exec(l)[0].length;

function isTableStart(lines, i) {
    return i + 1 < lines.length && lines[i].includes('|') && TABLE_DELIM_RE.test(lines[i + 1]) && lines[i + 1].includes('-');
}

function isBlockStart(lines, i) {
    const l = lines[i];
    return FENCE_RE.test(l) || HEADING_RE.test(l) || HR_RE.test(l) || /^ {0,3}>/.test(l) ||
        /^\s*\$\$/.test(l) || (LIST_RE.test(l) && !isBlank(LIST_RE.exec(l)[4])) || isTableStart(lines, i);
}

// ---- table cells ---------------------------------------------------------------------------------------
// Split on | that is not escaped and not inside `code` or $math$ (so |x| works in a cell).
function splitRow(line) {
    let s = line.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
    const sc = new Scan(s);
    const cells = [];
    let cur = '';
    for (let i = 0; i < s.length; i++) {
        const c = s[i];
        if (c === '\\' && i + 1 < s.length) { cur += c + s[i + 1]; i++; continue; }
        if (c === '`') { const end = sc.skipCode(i); cur += s.slice(i, end); i = end - 1; continue; }
        if (c === '$') {
            const m = sc.math(i);
            if (m) { cur += s.slice(i, m.end); i = m.end - 1; continue; }
        }
        if (c === '|') { cells.push(cur.trim()); cur = ''; continue; }
        cur += c;
    }
    cells.push(cur.trim());
    return cells;
}

// ---- inline scanners (shared by the renderer and the table splitter) -----------------------------------
// Code spans and math delimiters are precomputed once per string (linear time), so scanning stays
// near-linear even on pathological input such as thousands of unmatched backticks or dollars.
const PRICE_RE = /\d[\d,]*(?:\.\d+)?/y;
const HARD_BREAK_RE = / {2,}\n/y;
const isSpace = (c) => c === ' ' || c === '\n' || c === '\t';

class Scan {
    constructor(s) {
        this.s = s;
        const n = s.length;
        // Backtick runs: a run closes at the next run of the same length.
        this.runLen = new Map();
        this.codeClose = new Map();
        const runs = [];
        for (let i = 0; i < n;) {
            if (s[i] !== '`') { i++; continue; }
            let m = 0;
            while (s[i + m] === '`') m++;
            runs.push([i, m]);
            i += m;
        }
        const nextOfLen = new Map();
        for (let k = runs.length - 1; k >= 0; k--) {
            const [start, len] = runs[k];
            this.runLen.set(start, len);
            this.codeClose.set(start, nextOfLen.get(len) ?? -1);
            nextOfLen.set(len, start);
        }
        // closer[j]: first valid closing $ at index >= j; dd[j]: first "$$" at index >= j.
        this.closer = new Int32Array(n + 2).fill(-1);
        this.dd = new Int32Array(n + 2).fill(-1);
        for (let j = n - 1; j >= 0; j--) {
            this.closer[j] = s[j] === '$' && this.validCloser(j) ? j : this.closer[j + 1];
            this.dd[j] = s[j] === '$' && s[j + 1] === '$' ? j : this.dd[j + 1];
        }
    }

    validCloser(j) {
        const s = this.s;
        let b = 0;
        while (j - 1 - b >= 0 && s[j - 1 - b] === '\\') b++;
        return b % 2 === 0 && j > 0 && !isSpace(s[j - 1]) && !/\d/.test(s[j + 1] ?? '');
    }

    // Code span starting at a backtick run, or null.
    code(i) {
        const n = this.runLen.get(i);
        const j = n === undefined ? -1 : this.codeClose.get(i);
        if (j < 0) return null;
        let content = this.s.slice(i + n, j).replace(/\n/g, ' ');
        if (/^ .*[^ ].* $/.test(content)) content = content.slice(1, -1);
        return { end: j + n, content };
    }

    // Index just past the code span at i, or past the unmatched backtick run.
    skipCode(i) {
        const m = this.code(i);
        if (m) return m.end;
        return i + (this.runLen.get(i) ?? 1);
    }

    // Math at s[i] === '$' (see the rule in the header), or null.
    math(i) {
        const s = this.s;
        if (s[i + 1] === '$') {
            const close = this.dd[i + 2];
            if (close < 0 || close === i + 2) return null;
            return { end: close + 2, tex: s.slice(i + 2, close), display: true };
        }
        const next = s[i + 1];
        if (next === undefined || /\s/.test(next)) return null;
        // "$5 each", "$1,000.", "$3)" — a price, not math.
        PRICE_RE.lastIndex = i + 1;
        const price = PRICE_RE.exec(s);
        if (price) {
            const after = s[i + 1 + price[0].length];
            if (after === undefined || /[\s.,;:!?)\]]/.test(after)) return null;
        }
        const j = this.closer[i + 2];
        if (j < 0) return null;
        return { end: j + 1, tex: s.slice(i + 1, j), display: false };
    }
}

// [text](dest "title") starting at s[i] === '['.
function matchLink(sc, i) {
    const s = sc.s;
    let depth = 0, j = i;
    for (; j < s.length; j++) {
        const c = s[j];
        if (c === '\\') { j++; continue; }
        if (c === '`') { j = sc.skipCode(j) - 1; continue; }
        if (c === '[') depth++;
        else if (c === ']' && --depth === 0) break;
    }
    if (j >= s.length || s[j + 1] !== '(') return null;
    const text = s.slice(i + 1, j);
    let k = j + 2;
    while (s[k] === ' ') k++;
    let href = '';
    if (s[k] === '<') {
        const close = s.indexOf('>', k);
        if (close < 0) return null;
        href = s.slice(k + 1, close);
        k = close + 1;
    } else {
        let parens = 0;
        for (; k < s.length; k++) {
            const c = s[k];
            if (c === '\\' && k + 1 < s.length) { href += s[k + 1]; k++; continue; }
            if (/\s/.test(c)) break;
            if (c === '(') parens++;
            else if (c === ')') { if (parens === 0) break; parens--; }
            href += c;
        }
    }
    while (s[k] === ' ') k++;
    let title = null;
    if (s[k] === '"' || s[k] === "'") {
        const q = s[k];
        const close = s.indexOf(q, k + 1);
        if (close < 0) return null;
        title = s.slice(k + 1, close);
        k = close + 1;
        while (s[k] === ' ') k++;
    }
    if (s[k] !== ')') return null;
    return { text, href, title, end: k + 1 };
}

const PUNCT_RE = /[!-/:-@[-`{-~]/;
const isAlnum = (c) => c !== undefined && /[\p{L}\p{N}]/u.test(c);

// Find the closing emphasis delimiter, skipping escapes, code spans, math and nested ** pairs.
function findClose(sc, from, delim) {
    const s = sc.s;
    for (let j = from; j < s.length; j++) {
        const c = s[j];
        if (c === '\\') { j++; continue; }
        if (c === '`') { j = sc.skipCode(j) - 1; continue; }
        if (c === '$') { const m = sc.math(j); if (m) { j = m.end - 1; continue; } }
        if (delim.length === 1 && c === delim && s[j + 1] === delim) {
            const inner = findClose(sc, j + 2, delim + delim);
            j = inner < 0 ? j + 1 : inner + 1;
            continue;
        }
        if (s.startsWith(delim, j) && j > from && !/\s/.test(s[j - 1])) {
            if (delim[0] === '_' && isAlnum(s[j + delim.length])) continue;
            if (delim.length === 2 && s[j + 2] === delim[0] && s[j + 3] !== delim[0]) { // "***" closing: bold ends last
                return j + 1;
            }
            return j;
        }
    }
    return -1;
}

// ---- renderer ------------------------------------------------------------------------------------------
class Renderer {
    constructor(opts) {
        this.resolveLink = typeof opts.resolveLink === 'function' ? opts.resolveLink : null;
        this.idPrefix = opts.idPrefix ? String(opts.idPrefix) : '';
        this.slugs = new Set();
        this.headings = [];
        this.problems = [];
    }

    problem(line, message) {
        this.problems.push({ line, message });
    }

    url(raw, line, what) {
        if (raw.trim() === '') { this.problem(line, `empty ${what} URL`); return null; }
        let resolved = raw;
        if (this.resolveLink) {
            try { resolved = this.resolveLink(raw); } catch { resolved = null; }
        }
        const safe = safeUrl(resolved);
        if (safe === null) this.problem(line, `unsupported or unsafe ${what} URL: ${raw}`);
        return safe;
    }

    math(tex, display, line) {
        for (const e of texErrors(tex, { display })) this.problem(line, `math: ${e} in ${display ? '$$' : '$'}${tex}${display ? '$$' : '$'}`);
        return texToMathML(tex, { display });
    }

    // ---- inline ----
    inline(s, line, inLink = false) {
        let out = '', text = '';
        const flush = () => { out += escapeHtml(text); text = ''; };
        const sc = new Scan(s);
        const at = (k) => line + (s.slice(0, k).match(/\n/g)?.length ?? 0); // source line of s[k]
        for (let i = 0; i < s.length; i++) {
            const c = s[i];
            if (c === '\\') {
                if (s[i + 1] === '\n') { flush(); out += '<br>\n'; i++; continue; }
                if (s[i + 1] !== undefined && PUNCT_RE.test(s[i + 1])) { text += s[i + 1]; i++; continue; }
                text += c;
                continue;
            }
            if (c === ' ' && s[i + 1] === ' ') {
                HARD_BREAK_RE.lastIndex = i;
                const hb = HARD_BREAK_RE.exec(s);
                if (hb) { flush(); out += '<br>\n'; i += hb[0].length - 1; continue; }
            }
            if (c === '`') {
                const m = sc.code(i);
                if (m) { flush(); out += `<code>${escapeHtml(m.content)}</code>`; i = m.end - 1; continue; }
                const end = sc.skipCode(i);
                text += s.slice(i, end); i = end - 1; continue;
            }
            if (c === '$') {
                const m = sc.math(i);
                if (m) { flush(); out += this.math(m.tex, m.display, at(i)); i = m.end - 1; continue; }
                text += c; continue;
            }
            if (c === '!' && s[i + 1] === '[') {
                const m = matchLink(sc, i + 1);
                if (m) {
                    flush();
                    const alt = plainText(m.text);
                    const src = this.url(m.href, at(i), 'image');
                    if (src !== null) {
                        const title = m.title !== null ? ` title="${escapeHtml(m.title)}"` : '';
                        out += `<img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}"${title} loading="lazy">`;
                    } else out += escapeHtml(alt);
                    i = m.end - 1;
                    continue;
                }
            }
            if (c === '[' && !inLink) {
                const m = matchLink(sc, i);
                if (m) {
                    flush();
                    const inner = this.inline(m.text, at(i), true);
                    if (m.text.trim() === '') this.problem(at(i), `empty link text for (${m.href})`);
                    const href = this.url(m.href, at(i), 'link');
                    if (href !== null) {
                        const title = m.title !== null ? ` title="${escapeHtml(m.title)}"` : '';
                        const ext = /^https?:/i.test(href) ? ' target="_blank" rel="noopener noreferrer"' : '';
                        out += `<a href="${escapeHtml(href)}"${title}${ext}>${inner || escapeHtml(href)}</a>`;
                    } else out += inner;
                    i = m.end - 1;
                    continue;
                }
            }
            if (c === '*' || c === '_') {
                const dbl = s[i + 1] === c;
                const delim = dbl ? c + c : c;
                const after = s[i + delim.length];
                const opens = after !== undefined && !/\s/.test(after) && !(c === '_' && isAlnum(s[i - 1]));
                if (opens) {
                    const close = findClose(sc, i + delim.length, delim);
                    if (close > i + delim.length) {
                        flush();
                        const tag = dbl ? 'strong' : 'em';
                        out += `<${tag}>${this.inline(s.slice(i + delim.length, close), at(i + delim.length), inLink)}</${tag}>`;
                        i = close + delim.length - 1;
                        continue;
                    }
                }
                text += delim; i += delim.length - 1; continue;
            }
            text += c;
        }
        flush();
        return out;
    }

    // ---- blocks ----
    // lines: array of strings; base: 1-based source line of lines[0]; tight: render paragraphs without <p>.
    blocks(lines, base, tight = false) {
        const out = [];
        let i = 0;
        const n = lines.length;
        while (i < n) {
            const l = lines[i];
            const ln = base + i;
            if (isBlank(l)) { i++; continue; }

            let m = FENCE_RE.exec(l);
            if (m) {
                const [, ind, fence, lang] = m;
                const body = [];
                let closed = false;
                const close = new RegExp(`^ {0,3}${fence[0]}{${fence.length},}\\s*$`);
                i++;
                for (; i < n; i++) {
                    if (close.test(lines[i])) { closed = true; i++; break; }
                    body.push(lines[i].slice(Math.min(ind.length, indentOf(lines[i]))));
                }
                if (!closed) this.problem(ln, 'unclosed code fence');
                const safeLang = lang.replace(/[^A-Za-z0-9_+#-]/g, '');
                const code = escapeHtml(body.join('\n'));
                out.push(safeLang
                    ? `<div class="code-block"><span class="code-lang">${escapeHtml(safeLang)}</span><pre><code class="language-${escapeHtml(safeLang)}">${code}</code></pre></div>`
                    : `<pre><code>${code}</code></pre>`);
                continue;
            }

            m = HEADING_RE.exec(l);
            if (m) {
                const level = m[1].length;
                const src = (m[2] ?? '').trim();
                const text = plainText(src);
                let slug = this.idPrefix + slugify(text);
                if (this.slugs.has(slug)) {
                    let k = 2;
                    while (this.slugs.has(`${slug}-${k}`)) k++;
                    slug = `${slug}-${k}`;
                }
                this.slugs.add(slug);
                this.headings.push({ level, text, id: slug });
                out.push(`<h${level} id="${escapeHtml(slug)}">${this.inline(src, ln)}</h${level}>`);
                i++;
                continue;
            }

            if (HR_RE.test(l)) { out.push('<hr>'); i++; continue; }

            if (/^ {0,3}>/.test(l)) {
                const inner = [];
                while (i < n && /^ {0,3}>/.test(lines[i])) inner.push(lines[i++].replace(/^ {0,3}> ?/, ''));
                out.push(this.blockquote(inner, ln));
                continue;
            }

            if (/^\s*\$\$/.test(l)) {
                const r = this.displayMath(lines, i, base);
                if (r) { out.push(r.html); i = r.next; continue; }
            }

            if (isTableStart(lines, i)) {
                const r = this.table(lines, i, base);
                out.push(r.html);
                i = r.next;
                continue;
            }

            m = LIST_RE.exec(l);
            if (m) {
                const r = this.list(lines, i, base);
                out.push(r.html);
                i = r.next;
                continue;
            }

            // Paragraph.
            const para = [l.replace(/^\s+/, '')];
            i++;
            while (i < n && !isBlank(lines[i]) && !isBlockStart(lines, i)) para.push(lines[i++].replace(/^\s+/, ''));
            const html = this.inline(para.join('\n').replace(/\s+$/, ''), ln);
            out.push(tight ? html : `<p>${html}</p>`);
        }
        return out.join('\n');
    }

    displayMath(lines, i, base) {
        const first = lines[i].trim();
        const parts = [];
        let j = i;
        let rest = first.slice(2);
        for (;;) {
            const close = rest.indexOf('$$');
            if (close >= 0) {
                if (rest.slice(close + 2).trim() !== '') return null; // text after $$: treat as a paragraph
                parts.push(rest.slice(0, close));
                break;
            }
            parts.push(rest);
            j++;
            if (j >= lines.length || isBlank(lines[j])) {
                this.problem(base + i, 'unclosed $$ display math');
                return null;
            }
            rest = lines[j].trim();
        }
        const tex = parts.join('\n').trim();
        if (tex === '') { this.problem(base + i, 'empty $$ display math'); return null; }
        return { html: this.math(tex, true, base + i), next: j + 1 };
    }

    blockquote(inner, ln) {
        const firstIdx = inner.findIndex((x) => !isBlank(x));
        const first = firstIdx >= 0 ? inner[firstIdx] : '';
        const m = /^\s*\*\*([^*]+?):\*\*\s*/.exec(first);
        if (m && m[1] in CALLOUTS) {
            const lines = inner.slice();
            lines[firstIdx] = first.slice(m[0].length);
            let body = this.blocks(lines, ln);
            const label = `<strong class="callout-label">${escapeHtml(m[1])}:</strong>`;
            body = body.startsWith('<p>') ? `<p>${label} ${body.slice(3)}` : `<p>${label}</p>${body ? `\n${body}` : ''}`;
            return `<aside class="callout callout-${CALLOUTS[m[1]]}">\n${body}\n</aside>`;
        }
        if (m) this.problem(ln + Math.max(firstIdx, 0), `unknown callout type "${m[1]}" (use ${Object.keys(CALLOUTS).join(', ')})`);
        return `<blockquote>\n${this.blocks(inner, ln)}\n</blockquote>`;
    }

    table(lines, i, base) {
        const header = splitRow(lines[i]);
        const aligns = splitRow(lines[i + 1]).map((d) => {
            const l = d.startsWith(':'), r = d.endsWith(':');
            return l && r ? 'center' : r ? 'right' : l ? 'left' : null;
        });
        const cols = header.length;
        const cell = (tag, src, j, ln) => {
            const a = aligns[j] ? ` style="text-align:${aligns[j]}"` : '';
            return `<${tag}${a}>${this.inline(src ?? '', ln)}</${tag}>`;
        };
        const head = `<tr>${header.map((h, j) => cell('th', h, j, base + i)).join('')}</tr>`;
        const rows = [];
        let j = i + 2;
        for (; j < lines.length && !isBlank(lines[j]) && lines[j].includes('|'); j++) {
            const cells = splitRow(lines[j]);
            rows.push(`<tr>${Array.from({ length: cols }, (_, k) => cell('td', cells[k], k, base + j)).join('')}</tr>`);
        }
        const body = rows.length ? `<tbody>${rows.join('')}</tbody>` : '';
        return { html: `<div class="table-wrap"><table><thead>${head}</thead>${body}</table></div>`, next: j };
    }

    list(lines, i, base) {
        const n = lines.length;
        const m0 = LIST_RE.exec(lines[i]);
        const ind = m0[1].length;
        const ordered = /\d/.test(m0[2]);
        const start = ordered ? parseInt(m0[2], 10) : 1;
        const items = [];
        let loose = false;
        while (i < n) {
            const m = LIST_RE.exec(lines[i]);
            if (!m || /\d/.test(m[2]) !== ordered || m[1].length < ind) break;
            const pad = m[3].length > 4 ? 1 : Math.max(m[3].length, 1);
            const contentCol = ind + m[2].length + pad;
            const itemLines = [m[4]];
            const itemBase = base + i;
            i++;
            while (i < n) {
                const l = lines[i];
                if (isBlank(l)) {
                    let k = i;
                    while (k < n && isBlank(lines[k])) k++;
                    if (k < n && indentOf(lines[k]) >= contentCol) {
                        for (; i < k; i++) itemLines.push('');
                        loose = true;
                        continue;
                    }
                    break;
                }
                const li = indentOf(l);
                if (li >= contentCol) { itemLines.push(l.slice(contentCol)); i++; continue; }
                const mm = LIST_RE.exec(l);
                if (mm && li <= ind) break; // sibling (or parent) item
                if (li > ind) { itemLines.push(l.slice(li)); i++; continue; }
                // Lazy continuation of the item's paragraph.
                if (!isBlank(itemLines[itemLines.length - 1]) && !isBlockStart(lines, i)) { itemLines.push(l.trim()); i++; continue; }
                break;
            }
            items.push({ lines: itemLines, base: itemBase });
            // Blank lines between items make the list loose.
            let k = i;
            while (k < n && isBlank(lines[k])) k++;
            if (k > i && k < n) {
                const mm = LIST_RE.exec(lines[k]);
                if (mm && mm[1].length === ind && /\d/.test(mm[2]) === ordered) { loose = true; i = k; continue; }
            }
            if (k > i) break;
        }
        const tag = ordered ? 'ol' : 'ul';
        const startAttr = ordered && start !== 1 ? ` start="${start}"` : '';
        const body = items.map((it) => `<li>${this.blocks(it.lines, it.base, !loose)}</li>`).join('\n');
        return { html: `<${tag}${startAttr}>\n${body}\n</${tag}>`, next: i };
    }
}

function run(src, opts = {}) {
    const r = new Renderer(opts);
    const lines = String(src ?? '').replace(/\r\n?/g, '\n').replace(/\t/g, '    ').split('\n');
    const html = r.blocks(lines, 1);
    return { html, headings: r.headings, problems: r.problems };
}

/** Render Markdown to an HTML string. Options: resolveLink(href) → href, idPrefix for heading ids. */
export function renderMarkdown(src, opts = {}) {
    return run(src, opts).html;
}

/** Headings in document order: [{ level, text, id }] (ids match renderMarkdown with the same options). */
export function headings(src, opts = {}) {
    return run(src, opts).headings;
}

/** Content problems: [{ line, message }] — bad math, unclosed fences/$$, unknown callouts, empty/unsafe links. */
export function lintMarkdown(src, opts = {}) {
    return run(src, opts).problems;
}
