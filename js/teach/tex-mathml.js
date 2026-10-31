// TeX subset → native MathML (MathML Core), for the Understand tab (#54).
//
// A small recursive-descent parser; no dependencies and no DOM APIs, so it runs in Node and the browser.
//
// Grammar (informal):
//   sequence := scripted*
//   scripted := atom ( "'" | "^" arg | "_" arg | \limits | \nolimits )*
//   arg      := "{" sequence "}" | single digit | single letter | atom
//   atom     := number | letter | operator | "{" sequence "}" | \command args…
//
// Supported: identifiers (<mi>, one letter each; letter runs stay together inside \mathrm / \operatorname),
// numbers with decimals (<mn>), operators (<mo>, "-" becomes U+2212), groups, ^ _ and primes,
// \frac \dfrac \tfrac \binom, \sqrt[n]{x}, \left…\middle…\right with any delimiter (incl. "."),
// \big-style fixed-size delimiters, Greek letters, common relations/arrows/operators, big operators
// (\sum \prod \int \oint …: <munderover> in display mode, <msubsup> inline; \limits / \nolimits override),
// function names (\sin \log \lim \max …: upright <mi> + U+2061 function application; \lim \max \min \sup
// \inf \det \gcd \Pr put their subscripts underneath in display mode), \operatorname{} and \operatorname*{},
// \mathbf \mathrm \mathit \mathcal \mathbb \boldsymbol, \text{} (and \textrm \textbf \textit \mbox),
// accents \hat \widehat \bar \overline \underline \vec \dot \ddot \tilde \widetilde, spaces \, \: \; \! \ ~
// \quad \qquad, and the environments aligned, align(*), gathered, cases, matrix, pmatrix, bmatrix, vmatrix.
//
// \deg follows LaTeX (the operator name "deg"); use \degree for the ° sign.
//
// Never throws: unknown commands and parse errors render in place as <merror><mtext>…</mtext></merror>,
// and texErrors(tex) returns the list of problems so tests and the content linter can reject them.

const MATHML_NS = 'http://www.w3.org/1998/Math/MathML';
const APPLY = '<mo>⁡</mo>'; // invisible FUNCTION APPLICATION

export function escapeXml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

const GREEK = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ϵ', varepsilon: 'ε', zeta: 'ζ', eta: 'η',
    theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', omicron: 'ο',
    pi: 'π', varpi: 'ϖ', rho: 'ρ', varrho: 'ϱ', sigma: 'σ', varsigma: 'ς', tau: 'τ', upsilon: 'υ',
    phi: 'ϕ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
    Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Upsilon: 'Υ',
    Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
};

// Ordinary symbols rendered as identifiers. [char, upright]
const SYMBOL_MI = {
    nabla: ['∇', true], partial: ['∂', false], infty: ['∞', false], ell: ['ℓ', false], hbar: ['ℏ', false],
    emptyset: ['∅', true], varnothing: ['∅', true], degree: ['°', false], Re: ['ℜ', false], Im: ['ℑ', false],
};

const SYMBOL_MO = {
    cdot: '⋅', times: '×', div: '÷', pm: '±', mp: '∓', le: '≤', leq: '≤', ge: '≥', geq: '≥', ne: '≠',
    neq: '≠', approx: '≈', equiv: '≡', sim: '∼', simeq: '≃', cong: '≅', propto: '∝', ll: '≪', gg: '≫',
    lt: '<', gt: '>', to: '→', rightarrow: '→', leftarrow: '←', gets: '←', Rightarrow: '⇒',
    Leftarrow: '⇐', leftrightarrow: '↔', Leftrightarrow: '⇔', longrightarrow: '⟶', implies: '⟹',
    iff: '⟺', mapsto: '↦', uparrow: '↑', downarrow: '↓', in: '∈', notin: '∉', ni: '∋', subset: '⊂',
    subseteq: '⊆', supset: '⊃', supseteq: '⊇', cup: '∪', cap: '∩', setminus: '∖', forall: '∀',
    exists: '∃', neg: '¬', lnot: '¬', land: '∧', wedge: '∧', lor: '∨', vee: '∨', ldots: '…',
    cdots: '⋯', dots: '…', vdots: '⋮', ddots: '⋱', langle: '⟨', rangle: '⟩', lfloor: '⌊', rfloor: '⌋',
    lceil: '⌈', rceil: '⌉', circ: '∘', ast: '∗', star: '⋆', bullet: '∙', perp: '⊥', parallel: '∥',
    mid: '∣', vert: '|', Vert: '‖', colon: ':', oplus: '⊕', otimes: '⊗', odot: '⊙', prime: '′',
};

const BIG_OPS = {
    sum: '∑', prod: '∏', coprod: '∐', int: '∫', iint: '∬', iiint: '∭', oint: '∮', bigcup: '⋃',
    bigcap: '⋂', bigoplus: '⨁', bigotimes: '⨂',
};

const FUNCS = new Set([
    'log', 'ln', 'lg', 'exp', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'coth',
    'arcsin', 'arccos', 'arctan', 'arg', 'deg', 'dim', 'ker', 'hom',
]);
const LIMIT_FUNCS = new Set(['lim', 'liminf', 'limsup', 'max', 'min', 'sup', 'inf', 'det', 'gcd', 'Pr', 'argmax', 'argmin']);
const LIMIT_FUNC_TEXT = { liminf: 'lim inf', limsup: 'lim sup', argmax: 'arg max', argmin: 'arg min' };

const ACCENTS = {
    hat: ['^', false], widehat: ['^', true], bar: ['¯', false], overline: ['‾', true], vec: ['→', false],
    dot: ['˙', false], ddot: ['¨', false], tilde: ['˜', false], widetilde: ['˜', true],
};

const SPACES = {
    ',': '0.1667em', ':': '0.2222em', '>': '0.2222em', ';': '0.2778em', '!': '-0.1667em', ' ': '0.3333em',
    quad: '1em', qquad: '2em', enspace: '0.5em', thinspace: '0.1667em',
};

// Single-character escapes: \{ \} \| \$ \% \& \# \_
const ESCAPED = { '{': '{', '}': '}', '|': '‖', $: '$', '%': '%', '&': '&', '#': '#', _: '_' };

const DELIMS_CHAR = { '(': '(', ')': ')', '[': '[', ']': ']', '|': '|', '/': '/', '.': '', '<': '⟨', '>': '⟩' };
const DELIMS_CMD = {
    '{': '{', '}': '}', '|': '‖', langle: '⟨', rangle: '⟩', lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉',
    vert: '|', Vert: '‖', lvert: '|', rvert: '|', lVert: '‖', rVert: '‖', lbrace: '{', rbrace: '}',
    lbrack: '[', rbrack: ']', uparrow: '↑', downarrow: '↓',
};
const BIG_SIZES = { big: '1.2em', Big: '1.8em', bigg: '2.4em', Bigg: '3em' };

const ENVS = {
    aligned: { align: 'rl' }, align: { align: 'rl' }, 'align*': { align: 'rl' }, gathered: { align: 'c' },
    cases: { align: 'l', open: '{', close: '' }, matrix: { align: 'c' },
    pmatrix: { align: 'c', open: '(', close: ')' }, bmatrix: { align: 'c', open: '[', close: ']' },
    vmatrix: { align: 'c', open: '|', close: '|' },
};

const OPERATOR_CHARS = '+-=<>,;:!|/()[]*.?@';

// Mathematical alphanumeric symbols (MathML Core renders mathvariant only via these code points).
const SCRIPT_HOLES = { B: 'ℬ', E: 'ℰ', F: 'ℱ', H: 'ℋ', I: 'ℐ', L: 'ℒ', M: 'ℳ', R: 'ℛ' };
const BB_HOLES = { C: 'ℂ', H: 'ℍ', N: 'ℕ', P: 'ℙ', Q: 'ℚ', R: 'ℝ', Z: 'ℤ' };

function styleChar(ch, variant) {
    const c = ch.codePointAt(0);
    const upper = c >= 65 && c <= 90, lower = c >= 97 && c <= 122, digit = c >= 48 && c <= 57;
    switch (variant) {
        case 'bold':
            if (upper) return String.fromCodePoint(0x1d400 + c - 65);
            if (lower) return String.fromCodePoint(0x1d41a + c - 97);
            if (digit) return String.fromCodePoint(0x1d7ce + c - 48);
            return ch;
        case 'script':
            if (upper) return SCRIPT_HOLES[ch] ?? String.fromCodePoint(0x1d49c + c - 65);
            return ch;
        case 'double-struck':
            if (upper) return BB_HOLES[ch] ?? String.fromCodePoint(0x1d538 + c - 65);
            if (digit) return String.fromCodePoint(0x1d7d8 + c - 48);
            return ch;
        default:
            return ch;
    }
}

const VARIANT_OF = {
    mathbf: 'bold', boldsymbol: 'bold', mathrm: 'normal', mathit: 'italic', mathcal: 'script',
    mathbb: 'double-struck',
};

function row(nodes) {
    return nodes.length === 1 ? nodes[0] : `<mrow>${nodes.join('')}</mrow>`;
}

class TexParser {
    constructor(src, display) {
        this.src = String(src ?? '');
        this.pos = 0;
        this.display = display;
        this.errors = [];
        this.variant = null;
    }

    error(message, raw) {
        this.errors.push(message);
        return `<merror><mtext>${escapeXml(raw ?? message)}</mtext></merror>`;
    }

    // ---- tokenizer -------------------------------------------------------------------------------------
    skipWs() {
        while (this.pos < this.src.length && /\s/.test(this.src[this.pos])) this.pos++;
    }

    peek() {
        this.skipWs();
        return this.scan();
    }

    next() {
        this.skipWs();
        const tok = this.scan();
        this.pos = tok.end;
        return tok;
    }

    scan() {
        const s = this.src, i = this.pos;
        if (i >= s.length) return { t: 'eof', v: '', start: i, end: i };
        const c = s[i];
        if (c === '\\') {
            const m = /^[A-Za-z]+/.exec(s.slice(i + 1, i + 64));
            if (m) return { t: 'cmd', v: m[0], start: i, end: i + 1 + m[0].length };
            if (i + 1 < s.length) {
                const ch = String.fromCodePoint(s.codePointAt(i + 1));
                return { t: 'cmd', v: ch, start: i, end: i + 1 + ch.length };
            }
            return { t: 'char', v: '\\', start: i, end: i + 1 };
        }
        const num = /^(?:\d+(?:\.\d+)?|\.\d+)/.exec(s.slice(i, i + 64));
        if (num) return { t: 'num', v: num[0], start: i, end: i + num[0].length };
        const ch = String.fromCodePoint(s.codePointAt(i));
        const t = /^\p{L}$/u.test(ch) ? 'letter' : 'char';
        return { t, v: ch, start: i, end: i + ch.length };
    }

    isChar(tok, v) { return tok.t === 'char' && tok.v === v; }
    isCmd(tok, v) { return tok.t === 'cmd' && tok.v === v; }

    // ---- structure -------------------------------------------------------------------------------------
    parseSequence(stop) {
        const nodes = [];
        for (;;) {
            const tok = this.peek();
            if (tok.t === 'eof' || stop(tok)) break;
            nodes.push(...this.parseScripted());
        }
        return nodes;
    }

    parseGroup() {
        this.next(); // {
        const nodes = this.parseSequence((t) => this.isChar(t, '}'));
        if (this.isChar(this.peek(), '}')) this.next();
        else nodes.push(this.error('missing } to close a group', '{'));
        return row(nodes);
    }

    // An argument of ^, _, \frac, \sqrt, accents, font commands: a group or a single token.
    parseArg(what) {
        const t = this.peek();
        if (t.t === 'eof') return this.error(`missing argument for ${what}`, what);
        if (this.isChar(t, '{')) return this.parseGroup();
        if (this.isChar(t, '}') || this.isChar(t, '&') || this.isCmd(t, 'right') || this.isCmd(t, 'end') || this.isCmd(t, '\\')) {
            return this.error(`missing argument for ${what}`, what);
        }
        if (this.isChar(t, '^') || this.isChar(t, '_')) {
            this.next();
            return this.error(`missing argument for ${what}`, what);
        }
        return this.parseAtom(true).xml;
    }

    parseScripted() {
        const first = this.peek();
        const base = this.isChar(first, '^') || this.isChar(first, '_') ? { xml: '<mrow></mrow>', kind: 'ord' } : this.parseAtom(false);
        let sub = null, sup = null, primes = 0, limits = null;
        for (;;) {
            const t = this.peek();
            if (this.isChar(t, "'")) { this.next(); primes++; continue; }
            if (this.isCmd(t, 'limits') || this.isCmd(t, 'nolimits')) { this.next(); limits = t.v === 'limits'; continue; }
            if (this.isChar(t, '^') || this.isChar(t, '_')) {
                this.next();
                const isSup = t.v === '^';
                const arg = this.parseArg(isSup ? 'superscript' : 'subscript');
                if (isSup) sup = sup === null ? arg : `<mrow>${sup}${this.error('double superscript', '^')}${arg}</mrow>`;
                else sub = sub === null ? arg : `<mrow>${sub}${this.error('double subscript', '_')}${arg}</mrow>`;
                continue;
            }
            break;
        }
        if (primes) {
            const p = `<mo>${'′'.repeat(primes)}</mo>`;
            sup = sup === null ? p : `<mrow>${p}${sup}</mrow>`;
        }
        if (base.xml === '' && sub === null && sup === null) return [];
        let xml = base.xml;
        if (sub !== null || sup !== null) {
            const under = (base.kind === 'big' || base.kind === 'limits') && !primes && (limits ?? this.display);
            const tag = under
                ? (sub !== null && sup !== null ? 'munderover' : sub !== null ? 'munder' : 'mover')
                : (sub !== null && sup !== null ? 'msubsup' : sub !== null ? 'msub' : 'msup');
            xml = `<${tag}>${base.xml || '<mrow></mrow>'}${sub ?? ''}${sup ?? ''}</${tag}>`;
        }
        return base.apply ? [xml, APPLY] : [xml];
    }

    // ---- atoms -----------------------------------------------------------------------------------------
    mi(text, upright = false) {
        let attr = '';
        if (this.variant && this.variant !== 'italic') attr = ` mathvariant="${this.variant}"`;
        else if (upright && !this.variant) attr = ' mathvariant="normal"';
        const styled = this.variant ? [...text].map((c) => styleChar(c, this.variant)).join('') : text;
        return `<mi${attr}>${escapeXml(styled)}</mi>`;
    }

    parseAtom(single) {
        const peeked = this.peek();
        if (this.isChar(peeked, '{')) return { xml: this.parseGroup(), kind: 'ord' };
        const t = this.next();
        switch (t.t) {
            case 'num': {
                let v = t.v;
                if (single && v.length > 1) { v = v[0]; this.pos = t.start + 1; }
                const attr = this.variant === 'bold' || this.variant === 'double-struck' ? ` mathvariant="${this.variant}"` : '';
                const styled = this.variant ? [...v].map((c) => styleChar(c, this.variant)).join('') : v;
                return { xml: `<mn${attr}>${escapeXml(styled)}</mn>`, kind: 'ord' };
            }
            case 'letter': {
                let text = t.v;
                if (this.variant === 'normal' && !single) {
                    for (let n = this.peek(); n.t === 'letter'; n = this.peek()) { this.next(); text += n.v; }
                }
                return { xml: this.mi(text), kind: 'ord' };
            }
            case 'char':
                return { xml: this.charAtom(t), kind: 'ord' };
            default:
                return this.command(t);
        }
    }

    charAtom(t) {
        const c = t.v;
        if (c === '}') return this.error('unmatched }', '}');
        if (c === '&') return this.error('& outside an aligned/cases/matrix environment', '&');
        if (c === '~') return '<mspace width="0.3333em"></mspace>';
        if (c === "'") return '<mo>′</mo>';
        if (c === '-') return '<mo>−</mo>';
        if (c === '*') return '<mo>∗</mo>';
        if (OPERATOR_CHARS.includes(c)) return `<mo>${escapeXml(c)}</mo>`;
        if (c.codePointAt(0) > 127) return `<mo>${escapeXml(c)}</mo>`;
        return this.error(`unexpected character ${c}`, c);
    }

    withVariant(variant, fn) {
        const saved = this.variant;
        this.variant = variant;
        try { return fn(); } finally { this.variant = saved; }
    }

    command(t) {
        const name = t.v;
        if (name in GREEK) return { xml: this.mi(GREEK[name], /^[A-Z]/.test(name)), kind: 'ord' };
        if (name in SYMBOL_MI) return { xml: this.mi(...SYMBOL_MI[name]), kind: 'ord' };
        if (name in SYMBOL_MO) return { xml: `<mo>${escapeXml(SYMBOL_MO[name])}</mo>`, kind: 'ord' };
        if (name in BIG_OPS) return { xml: `<mo>${BIG_OPS[name]}</mo>`, kind: 'big' };
        if (FUNCS.has(name)) return { xml: `<mi>${name}</mi>`, kind: 'func', apply: true };
        if (LIMIT_FUNCS.has(name)) return { xml: `<mi>${LIMIT_FUNC_TEXT[name] ?? name}</mi>`, kind: 'limits', apply: true };
        if (name in SPACES) return { xml: `<mspace width="${SPACES[name]}"></mspace>`, kind: 'ord' };
        if (name in ESCAPED) return { xml: `<mo>${escapeXml(ESCAPED[name])}</mo>`, kind: 'ord' };
        if (name in VARIANT_OF) {
            const v = VARIANT_OF[name];
            return { xml: this.withVariant(v, () => this.parseArg(`\\${name}`)), kind: 'ord' };
        }
        if (name in ACCENTS) {
            const [mark, stretchy] = ACCENTS[name];
            const arg = this.parseArg(`\\${name}`);
            return { xml: `<mover accent="true">${arg}<mo stretchy="${stretchy}">${escapeXml(mark)}</mo></mover>`, kind: 'ord' };
        }
        if (name in BIG_SIZES) return { xml: this.delimiter(`\\${name}`, BIG_SIZES[name]), kind: 'ord' };
        if (/^(big|Big|bigg|Bigg)[lrm]$/.test(name)) return { xml: this.delimiter(`\\${name}`, BIG_SIZES[name.slice(0, -1)]), kind: 'ord' };
        switch (name) {
            case 'frac': case 'dfrac': case 'tfrac': case 'cfrac': {
                const num = this.parseArg(`\\${name}`), den = this.parseArg(`\\${name}`);
                const frac = `<mfrac>${num}${den}</mfrac>`;
                if (name === 'dfrac' || name === 'cfrac') return { xml: `<mstyle displaystyle="true">${frac}</mstyle>`, kind: 'ord' };
                if (name === 'tfrac') return { xml: `<mstyle displaystyle="false">${frac}</mstyle>`, kind: 'ord' };
                return { xml: frac, kind: 'ord' };
            }
            case 'binom': {
                const n = this.parseArg('\\binom'), k = this.parseArg('\\binom');
                return { xml: `<mrow><mo>(</mo><mfrac linethickness="0">${n}${k}</mfrac><mo>)</mo></mrow>`, kind: 'ord' };
            }
            case 'sqrt': {
                let index = null;
                if (this.isChar(this.peek(), '[')) {
                    this.next();
                    const nodes = this.parseSequence((tok) => this.isChar(tok, ']'));
                    if (this.isChar(this.peek(), ']')) this.next();
                    else nodes.push(this.error('missing ] in \\sqrt[…]', '['));
                    index = row(nodes);
                }
                const arg = this.parseArg('\\sqrt');
                return { xml: index === null ? `<msqrt>${arg}</msqrt>` : `<mroot>${arg}${index}</mroot>`, kind: 'ord' };
            }
            case 'left': {
                const open = this.delimiter('\\left');
                const nodes = this.parseSequence((tok) => this.isCmd(tok, 'right'));
                let close;
                if (this.isCmd(this.peek(), 'right')) { this.next(); close = this.delimiter('\\right'); }
                else close = this.error('missing \\right to match \\left', '\\left');
                return { xml: `<mrow>${open}${nodes.join('')}${close}</mrow>`, kind: 'ord' };
            }
            case 'middle':
                return { xml: this.delimiter('\\middle'), kind: 'ord' };
            case 'right': {
                const err = this.error('\\right without matching \\left', '\\right');
                if (this.peek().t !== 'eof') this.next();
                return { xml: err, kind: 'ord' };
            }
            case 'operatorname': {
                let kind = 'func';
                if (this.isChar(this.peek(), '*')) { this.next(); kind = 'limits'; }
                const arg = this.withVariant('normal', () => this.parseArg('\\operatorname'));
                return { xml: arg, kind, apply: true };
            }
            case 'text': case 'textrm': case 'mbox': case 'textbf': case 'textit': case 'textnormal': {
                const text = this.readRawGroup(`\\${name}`);
                if (text === null) return { xml: this.error(`missing argument for \\${name}`, `\\${name}`), kind: 'ord' };
                const attr = name === 'textbf' ? ' mathvariant="bold"' : name === 'textit' ? ' mathvariant="italic"' : '';
                const shown = text.replace(/^ +/, (m) => '\u00a0'.repeat(m.length)).replace(/ +$/, (m) => '\u00a0'.repeat(m.length));
                return { xml: `<mtext${attr}>${escapeXml(shown)}</mtext>`, kind: 'ord' };
            }
            case 'underline': {
                const arg = this.parseArg('\\underline');
                return { xml: `<munder accentunder="true">${arg}<mo stretchy="true">_</mo></munder>`, kind: 'ord' };
            }
            case 'displaystyle': case 'textstyle':
                return { xml: '', kind: 'ord' };
            case 'begin':
                return { xml: this.environment(), kind: 'ord' };
            case 'end': {
                const env = this.readEnvName();
                return { xml: this.error(`\\end{${env}} without matching \\begin`, `\\end{${env}}`), kind: 'ord' };
            }
            case '\\':
                return { xml: this.error('\\\\ line break outside an aligned/cases/matrix environment', '\\\\'), kind: 'ord' };
            default:
                return { xml: this.error(`unknown command \\${name}`, `\\${name}`), kind: 'ord' };
        }
    }

    // A delimiter after \left, \right, \middle or \big…; "." is the empty delimiter.
    delimiter(after, size = null) {
        const t = this.next();
        let d;
        if (t.t === 'char' && t.v in DELIMS_CHAR) d = DELIMS_CHAR[t.v];
        else if (t.t === 'cmd' && t.v in DELIMS_CMD) d = DELIMS_CMD[t.v];
        else {
            if (t.t === 'eof') return this.error(`missing delimiter after ${after}`, after);
            return this.error(`bad delimiter after ${after}: ${this.src.slice(t.start, t.end)}`, `${after}${this.src.slice(t.start, t.end)}`);
        }
        if (d === '') return '';
        if (size) return `<mo fence="true" stretchy="true" minsize="${size}" maxsize="${size}">${escapeXml(d)}</mo>`;
        return `<mo fence="true" stretchy="true">${escapeXml(d)}</mo>`;
    }

    // Raw text argument of \text{…}: balanced braces, with \{ \} \$ \% \& \# \_ and "\ " unescaped.
    readRawGroup() {
        this.skipWs();
        const s = this.src;
        if (this.pos >= s.length) return null;
        if (s[this.pos] !== '{') {
            const ch = String.fromCodePoint(s.codePointAt(this.pos));
            if (ch === '}' || ch === '\\') return null;
            this.pos += ch.length;
            return ch;
        }
        let depth = 0, out = '';
        for (let i = this.pos; i < s.length; i++) {
            const c = s[i];
            if (c === '\\' && i + 1 < s.length) {
                const n = s[i + 1];
                out += '{}$%&#_ '.includes(n) ? n : c + n;
                i++;
                continue;
            }
            if (c === '{') { if (depth++ > 0) out += c; continue; }
            if (c === '}') {
                if (--depth === 0) { this.pos = i + 1; return out; }
                out += c;
                continue;
            }
            out += c;
        }
        this.pos = s.length;
        this.errors.push('missing } to close \\text{…}');
        return out;
    }

    readEnvName() {
        this.skipWs();
        const m = /^\{([A-Za-z*]*)\}/.exec(this.src.slice(this.pos, this.pos + 40));
        if (!m) return '';
        this.pos += m[0].length;
        return m[1];
    }

    environment() {
        const name = this.readEnvName();
        const env = ENVS[name];
        if (!env) return this.error(`unknown environment ${name ? `{${name}}` : '(missing name)'}`, `\\begin{${name}}`);
        const rows = [[]];
        const extra = [];
        for (;;) {
            const nodes = this.parseSequence((t) => this.isChar(t, '&') || this.isCmd(t, '\\') || this.isCmd(t, 'end'));
            rows[rows.length - 1].push(nodes);
            const t = this.peek();
            if (this.isChar(t, '&')) { this.next(); continue; }
            if (this.isCmd(t, '\\')) {
                this.next();
                this.skipWs();
                const opt = /^\[[^\]]*\]/.exec(this.src.slice(this.pos, this.pos + 40)); // \\[2pt]
                if (opt) this.pos += opt[0].length;
                rows.push([]);
                continue;
            }
            if (this.isCmd(t, 'end')) {
                this.next();
                const end = this.readEnvName();
                if (end !== name) extra.push(this.error(`\\begin{${name}} ended by \\end{${end}}`, `\\end{${end}}`));
                break;
            }
            extra.push(this.error(`missing \\end{${name}}`, `\\begin{${name}}`));
            break;
        }
        const last = rows[rows.length - 1];
        if (rows.length > 1 && last.length === 1 && last[0].length === 0) rows.pop();
        const cols = Math.max(...rows.map((r) => r.length));
        const alignOf = (j) => env.align === 'rl' ? (j % 2 === 0 ? 'right' : 'left') : env.align === 'l' ? 'left' : 'center';
        const columnalign = Array.from({ length: cols }, (_, j) => alignOf(j)).join(' ');
        const body = rows.map((r) => `<mtr>${r.map((cell, j) => `<mtd style="text-align:${alignOf(j)}">${cell.length ? row(cell) : ''}</mtd>`).join('')}</mtr>`).join('');
        const display = env.align === 'rl' || env.align === 'c' && name === 'gathered' ? ' displaystyle="true"' : '';
        let xml = `<mtable columnalign="${columnalign}"${display}>${body}</mtable>`;
        if (env.open !== undefined) {
            const fence = (d) => d ? `<mo fence="true" stretchy="true">${escapeXml(d)}</mo>` : '';
            xml = `<mrow>${fence(env.open)}${xml}${fence(env.close)}</mrow>`;
        }
        return extra.length ? `<mrow>${xml}${extra.join('')}</mrow>` : xml;
    }
}

function parse(tex, display) {
    const p = new TexParser(tex, display);
    let body;
    try {
        body = p.parseSequence(() => false).join('');
    } catch (e) {
        // Defensive: pathological input (e.g. nesting deep enough to overflow the stack).
        p.errors.push(`could not parse: ${e && e.message ? e.message : e}`);
        body = `<merror><mtext>${escapeXml(tex)}</mtext></merror>`;
    }
    return { body, errors: p.errors };
}

export function texToMathML(tex, { display = false } = {}) {
    const { body } = parse(tex, display);
    return `<math xmlns="${MATHML_NS}" display="${display ? 'block' : 'inline'}">${body}</math>`;
}

export function texErrors(tex, { display = false } = {}) {
    return parse(tex, display).errors;
}
