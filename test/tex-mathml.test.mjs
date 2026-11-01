// TeX subset → MathML (js/teach/tex-mathml.js, #54).
//
// Each case is TeX → a MathML fragment that must appear in the output (whitespace-normalized), with no
// parse errors. Error cases must render <merror> in place and be reported by texErrors().

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { texToMathML, texErrors } from '../js/teach/tex-mathml.js';

const norm = (s) => s.replace(/\s+/g, ' ').replace(/> </g, '><').trim();
const inner = (tex, display = false) => texToMathML(tex, { display }).replace(/^<math[^>]*>|<\/math>$/g, '');

function check(tex, expected, { display = false, exact = false } = {}) {
    const out = norm(inner(tex, display));
    const want = norm(expected);
    if (exact) assert.equal(out, want, `\n  TeX: ${tex}`);
    else assert.ok(out.includes(want), `\n  TeX: ${tex}\n  got:  ${out}\n  want: ${want}`);
    assert.deepEqual(texErrors(tex, { display }), [], `unexpected errors for ${tex}`);
}

const CASES = [
    // tokens
    ['identifiers split into letters', 'xy', '<mi>x</mi><mi>y</mi>', { exact: true }],
    ['integer', '42', '<mn>42</mn>', { exact: true }],
    ['decimal', '3.14', '<mn>3.14</mn>', { exact: true }],
    ['leading-dot decimal', '.5', '<mn>.5</mn>', { exact: true }],
    ['minus is U+2212', 'a-b', '<mi>a</mi><mo>−</mo><mi>b</mi>', { exact: true }],
    ['operators', '+=,;:!|/()[]', '<mo>+</mo><mo>=</mo><mo>,</mo><mo>;</mo><mo>:</mo><mo>!</mo><mo>|</mo><mo>/</mo><mo>(</mo><mo>)</mo><mo>[</mo><mo>]</mo>', { exact: true }],
    ['< and > escaped', 'a<b>c', '<mi>a</mi><mo>&lt;</mo><mi>b</mi><mo>&gt;</mo><mi>c</mi>', { exact: true }],
    ['escaped braces', '\\{x\\}', '<mo>{</mo><mi>x</mi><mo>}</mo>', { exact: true }],
    ['whitespace ignored', ' a   +  b ', '<mi>a</mi><mo>+</mo><mi>b</mi>', { exact: true }],
    // scripts
    ['superscript', 'x^2', '<msup><mi>x</mi><mn>2</mn></msup>', { exact: true }],
    ['subscript', 'x_i', '<msub><mi>x</mi><mi>i</mi></msub>', { exact: true }],
    ['sub+sup', 'x_i^2', '<msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup>', { exact: true }],
    ['sup+sub order-independent', 'x^2_i', '<msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup>', { exact: true }],
    ['grouped superscript', 'x^{n+1}', '<msup><mi>x</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msup>', { exact: true }],
    ['single-digit script argument', 'x^23', '<msup><mi>x</mi><mn>2</mn></msup><mn>3</mn>', { exact: true }],
    ['nested scripts', 'e^{x_i^2}', '<msup><mi>e</mi><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup></msup>', { exact: true }],
    ['prime', "f'", '<msup><mi>f</mi><mo>′</mo></msup>', { exact: true }],
    ['double prime', "f''(x)", '<msup><mi>f</mi><mo>′′</mo></msup><mo>(</mo>'],
    ['prime with superscript', "f'^2", '<msup><mi>f</mi><mrow><mo>′</mo><mn>2</mn></mrow></msup>', { exact: true }],
    ['group', '{a+b}^2', '<msup><mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow><mn>2</mn></msup>', { exact: true }],
    // fractions and roots
    ['fraction', '\\frac{a}{b}', '<mfrac><mi>a</mi><mi>b</mi></mfrac>', { exact: true }],
    ['fraction with single-token args', '\\frac12', '<mfrac><mn>1</mn><mn>2</mn></mfrac>', { exact: true }],
    ['nested fraction', '\\frac{1}{1+\\frac{1}{x}}', '<mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><mfrac><mn>1</mn><mi>x</mi></mfrac></mrow></mfrac>', { exact: true }],
    ['fraction with sub/sup', '\\frac{x_i^2}{\\sigma^2}', '<mfrac><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup><msup><mi>σ</mi><mn>2</mn></msup></mfrac>', { exact: true }],
    ['dfrac', '\\dfrac{a}{b}', '<mstyle displaystyle="true"><mfrac><mi>a</mi><mi>b</mi></mfrac></mstyle>', { exact: true }],
    ['sqrt', '\\sqrt{x}', '<msqrt><mi>x</mi></msqrt>', { exact: true }],
    ['nth root', '\\sqrt[3]{x+1}', '<mroot><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mn>3</mn></mroot>', { exact: true }],
    ['binom', '\\binom{n}{k}', '<mfrac linethickness="0"><mi>n</mi><mi>k</mi></mfrac>'],
    // fences
    ['left/right parens', '\\left( \\frac{a}{b} \\right)', '<mrow><mo fence="true" stretchy="true">(</mo><mfrac><mi>a</mi><mi>b</mi></mfrac><mo fence="true" stretchy="true">)</mo></mrow>', { exact: true }],
    ['left brace, right dot', '\\left\\{ x \\right.', '<mrow><mo fence="true" stretchy="true">{</mo><mi>x</mi></mrow>', { exact: true }],
    ['left/right bars', '\\left| x \\right|', '<mo fence="true" stretchy="true">|</mo><mi>x</mi><mo fence="true" stretchy="true">|</mo>'],
    ['left/right norm', '\\left\\| v \\right\\|', '<mo fence="true" stretchy="true">‖</mo><mi>v</mi><mo fence="true" stretchy="true">‖</mo>'],
    ['left/right brackets', '\\left[ x \\right]', '<mo fence="true" stretchy="true">[</mo><mi>x</mi><mo fence="true" stretchy="true">]</mo>'],
    ['angle brackets', '\\left\\langle x \\right\\rangle', '<mo fence="true" stretchy="true">⟨</mo><mi>x</mi><mo fence="true" stretchy="true">⟩</mo>'],
    ['nested left/right', '\\left( a \\left[ b \\right] \\right)', '<mrow><mo fence="true" stretchy="true">(</mo><mi>a</mi><mrow><mo fence="true" stretchy="true">[</mo><mi>b</mi><mo fence="true" stretchy="true">]</mo></mrow><mo fence="true" stretchy="true">)</mo></mrow>', { exact: true }],
    ['left/right with script', '\\left( x \\right)^2', '<msup><mrow><mo fence="true" stretchy="true">(</mo><mi>x</mi><mo fence="true" stretchy="true">)</mo></mrow><mn>2</mn></msup>', { exact: true }],
    ['big delimiter', '\\big( x \\big)', '<mo fence="true" stretchy="true" minsize="1.2em" maxsize="1.2em">(</mo>'],
    // symbols
    ['greek lower', '\\alpha\\beta\\omega', '<mi>α</mi><mi>β</mi><mi>ω</mi>', { exact: true }],
    ['greek variants', '\\varepsilon\\vartheta\\varphi', '<mi>ε</mi><mi>ϑ</mi><mi>φ</mi>', { exact: true }],
    ['greek upper is upright', '\\Delta\\Omega', '<mi mathvariant="normal">Δ</mi><mi mathvariant="normal">Ω</mi>', { exact: true }],
    ['nabla and partial', '\\nabla\\partial', '<mi mathvariant="normal">∇</mi><mi>∂</mi>', { exact: true }],
    ['infty', '\\infty', '<mi>∞</mi>', { exact: true }],
    ['relations', '\\le\\ge\\ne\\approx\\equiv\\sim\\propto', '<mo>≤</mo><mo>≥</mo><mo>≠</mo><mo>≈</mo><mo>≡</mo><mo>∼</mo><mo>∝</mo>', { exact: true }],
    ['binary operators', '\\cdot\\times\\div\\pm\\mp\\circ\\ast\\star', '<mo>⋅</mo><mo>×</mo><mo>÷</mo><mo>±</mo><mo>∓</mo><mo>∘</mo><mo>∗</mo><mo>⋆</mo>', { exact: true }],
    ['arrows', '\\to\\rightarrow\\leftarrow\\Rightarrow', '<mo>→</mo><mo>→</mo><mo>←</mo><mo>⇒</mo>', { exact: true }],
    ['sets and logic', '\\in\\notin\\subset\\cup\\cap\\forall\\exists', '<mo>∈</mo><mo>∉</mo><mo>⊂</mo><mo>∪</mo><mo>∩</mo><mo>∀</mo><mo>∃</mo>', { exact: true }],
    ['dots', '\\ldots\\cdots\\dots', '<mo>…</mo><mo>⋯</mo><mo>…</mo>', { exact: true }],
    ['floor/ceil/angle', '\\langle\\rangle\\lfloor\\rfloor\\lceil\\rceil', '<mo>⟨</mo><mo>⟩</mo><mo>⌊</mo><mo>⌋</mo><mo>⌈</mo><mo>⌉</mo>', { exact: true }],
    // functions and big operators
    ['function application', '\\sin x', '<mi>sin</mi><mo>⁡</mo><mi>x</mi>', { exact: true }],
    ['all functions upright', '\\log\\ln\\exp\\cos\\tan\\arg\\det\\deg', '<mi>log</mi><mo>⁡</mo><mi>ln</mi><mo>⁡</mo><mi>exp</mi><mo>⁡</mo><mi>cos</mi><mo>⁡</mo><mi>tan</mi><mo>⁡</mo><mi>arg</mi><mo>⁡</mo><mi>det</mi><mo>⁡</mo><mi>deg</mi><mo>⁡</mo>', { exact: true }],
    ['function with subscript', '\\log_2 n', '<msub><mi>log</mi><mn>2</mn></msub><mo>⁡</mo><mi>n</mi>', { exact: true }],
    ['function power', '\\sin^2\\theta', '<msup><mi>sin</mi><mn>2</mn></msup><mo>⁡</mo><mi>θ</mi>', { exact: true }],
    ['lim inline → msub', '\\lim_{h\\to 0} f', '<msub><mi>lim</mi><mrow><mi>h</mi><mo>→</mo><mn>0</mn></mrow></msub><mo>⁡</mo>'],
    ['lim display → munder', '\\lim_{h\\to 0} f', '<munder><mi>lim</mi><mrow><mi>h</mi><mo>→</mo><mn>0</mn></mrow></munder><mo>⁡</mo>', { display: true }],
    ['max display → munder', '\\max_i x_i', '<munder><mi>max</mi><mi>i</mi></munder>', { display: true }],
    ['min inline', '\\min_i x_i', '<msub><mi>min</mi><mi>i</mi></msub>'],
    ['sum inline → msubsup', '\\sum_{i=1}^{N} x_i', '<msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></msubsup>'],
    ['sum display → munderover', '\\sum_{i=1}^{N} x_i', '<munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover>', { display: true }],
    ['prod display', '\\prod_{k} a_k', '<munder><mo>∏</mo><mi>k</mi></munder>', { display: true }],
    ['int inline', '\\int_0^1 f', '<msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup>'],
    ['int display', '\\int_0^\\infty f', '<munderover><mo>∫</mo><mn>0</mn><mi>∞</mi></munderover>', { display: true }],
    ['oint', '\\oint_C', '<msub><mo>∮</mo><mi>C</mi></msub>'],
    ['\\nolimits in display', '\\sum\\nolimits_i', '<msub><mo>∑</mo><mi>i</mi></msub>', { display: true }],
    ['\\limits inline', '\\sum\\limits_i', '<munder><mo>∑</mo><mi>i</mi></munder>'],
    ['operatorname', '\\operatorname{erf}(x)', '<mi mathvariant="normal">erf</mi><mo>⁡</mo><mo>(</mo>'],
    // fonts and text
    ['mathbf', '\\mathbf{F}', '<mi mathvariant="bold">𝐅</mi>', { exact: true }],
    ['mathbf multiple letters', '\\mathbf{ab}', '<mrow><mi mathvariant="bold">𝐚</mi><mi mathvariant="bold">𝐛</mi></mrow>', { exact: true }],
    ['mathrm keeps runs', '\\mathrm{max}', '<mi mathvariant="normal">max</mi>', { exact: true }],
    ['mathrm single letter', '\\mathrm{d}x', '<mi mathvariant="normal">d</mi><mi>x</mi>', { exact: true }],
    ['mathcal', '\\mathcal{L}', '<mi mathvariant="script">ℒ</mi>', { exact: true }],
    ['mathbb', '\\mathbb{R}^n', '<msup><mi mathvariant="double-struck">ℝ</mi><mi>n</mi></msup>', { exact: true }],
    ['text', '\\text{if } x > 0', '<mtext>if\u00a0</mtext><mi>x</mi>'],
    ['text keeps spaces and braces', '\\text{a {b} c}', '<mtext>a {b} c</mtext>', { exact: true }],
    // accents
    ['hat', '\\hat{x}', '<mover accent="true"><mi>x</mi><mo stretchy="false">^</mo></mover>', { exact: true }],
    ['bar', '\\bar{x}', '<mover accent="true"><mi>x</mi><mo stretchy="false">¯</mo></mover>', { exact: true }],
    ['vec', '\\vec{v}', '<mover accent="true"><mi>v</mi><mo stretchy="false">→</mo></mover>', { exact: true }],
    ['dot', '\\dot{x}', '<mover accent="true"><mi>x</mi><mo stretchy="false">˙</mo></mover>', { exact: true }],
    ['ddot', '\\ddot{x}', '<mover accent="true"><mi>x</mi><mo stretchy="false">¨</mo></mover>', { exact: true }],
    ['tilde', '\\tilde{x}', '<mover accent="true"><mi>x</mi><mo stretchy="false">˜</mo></mover>', { exact: true }],
    ['accent with script', '\\hat{x}_i', '<msub><mover accent="true"><mi>x</mi><mo stretchy="false">^</mo></mover><mi>i</mi></msub>', { exact: true }],
    ['bold vector accent', '\\vec{\\mathbf{v}}', '<mover accent="true"><mi mathvariant="bold">𝐯</mi>'],
    // spaces
    ['thin space', 'a\\,b', '<mspace width="0.1667em"></mspace>'],
    ['thick space', 'a\\;b', '<mspace width="0.2778em"></mspace>'],
    ['quad', 'a\\quad b', '<mspace width="1em"></mspace>'],
    ['qquad', 'a\\qquad b', '<mspace width="2em"></mspace>'],
    // environments
    ['aligned', '\\begin{aligned} a &= b + c \\\\ d &= e \\end{aligned}',
        '<mtable columnalign="right left" displaystyle="true"><mtr><mtd style="text-align:right"><mi>a</mi></mtd><mtd style="text-align:left"><mrow><mo>=</mo><mi>b</mi><mo>+</mo><mi>c</mi></mrow></mtd></mtr><mtr><mtd style="text-align:right"><mi>d</mi></mtd><mtd style="text-align:left"><mrow><mo>=</mo><mi>e</mi></mrow></mtd></mtr></mtable>', { display: true, exact: true }],
    ['aligned trailing \\\\ dropped', '\\begin{aligned} a &= b \\\\ \\end{aligned}', '<mtable columnalign="right left" displaystyle="true"><mtr><mtd style="text-align:right"><mi>a</mi></mtd><mtd style="text-align:left"><mrow><mo>=</mo><mi>b</mi></mrow></mtd></mtr></mtable>', { exact: true }],
    ['cases', 'f(x) = \\begin{cases} 1 & x > 0 \\\\ 0 & \\text{otherwise} \\end{cases}',
        '<mrow><mo fence="true" stretchy="true">{</mo><mtable columnalign="left left"><mtr><mtd style="text-align:left"><mn>1</mn></mtd><mtd style="text-align:left"><mrow><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow></mtd></mtr><mtr><mtd style="text-align:left"><mn>0</mn></mtd><mtd style="text-align:left"><mtext>otherwise</mtext></mtd></mtr></mtable></mrow>'],
    ['pmatrix', '\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}', '<mo fence="true" stretchy="true">(</mo><mtable columnalign="center center"><mtr><mtd style="text-align:center"><mi>a</mi></mtd>'],
    // the issue's sample equation
    ['wave equation', '\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u',
        '<mfrac><mrow><msup><mi>∂</mi><mn>2</mn></msup><mi>u</mi></mrow><mrow><mi>∂</mi><msup><mi>t</mi><mn>2</mn></msup></mrow></mfrac><mo>=</mo><msup><mi>c</mi><mn>2</mn></msup><msup><mi mathvariant="normal">∇</mi><mn>2</mn></msup><mi>u</mi>', { display: true, exact: true }],
];

for (const [name, tex, expected, opts] of CASES) {
    test(`tex: ${name}`, () => check(tex, expected, opts));
}

test('math element: namespace and display attribute', () => {
    assert.equal(texToMathML('x'), '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline"><mi>x</mi></math>');
    assert.equal(texToMathML('x', { display: true }), '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>x</mi></math>');
    assert.equal(texToMathML(''), '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline"></math>');
});

test('text content is escaped (< & " \' inside \\text)', () => {
    const out = texToMathML('\\text{a < b & "c" \'d\'}');
    assert.ok(out.includes('<mtext>a &lt; b &amp; &quot;c&quot; &#39;d&#39;</mtext>'), out);
    assert.ok(!/<b|<script/i.test(out.replace(/<\/?(math|mtext)[^>]*>/g, '')));
});

test('no markup injection through any input', () => {
    for (const tex of ['\\text{</mtext><script>alert(1)</script>}', '\\foo<script>', '<img src=x onerror=alert(1)>', '\\begin{<x>}', '\\left<\\right>']) {
        const out = texToMathML(tex);
        assert.ok(!out.includes('<script') && !out.includes('<img') && !out.includes('<x>'), `${tex} → ${out}`);
    }
});

const ERRORS = [
    ['unknown command', '\\foo x', '<merror><mtext>\\foo</mtext></merror>', /unknown command \\foo/],
    ['unknown command among valid ones', 'a + \\badcmd{b}', '<merror><mtext>\\badcmd</mtext></merror>', /unknown command/],
    ['unclosed group', 'x^{2', '<merror>', /missing }/],
    ['unmatched }', 'a}b', '<merror><mtext>}</mtext></merror>', /unmatched }/],
    ['missing \\right', '\\left( x', '<merror><mtext>\\left</mtext></merror>', /missing \\right/],
    ['\\right without \\left', 'x \\right)', '<merror><mtext>\\right</mtext></merror>', /without matching \\left/],
    ['bad delimiter', '\\left x \\right)', '<merror>', /bad delimiter/],
    ['missing fraction argument', '\\frac{a}', '<merror><mtext>\\frac</mtext></merror>', /missing argument for \\frac/],
    ['missing superscript', 'x^', '<merror><mtext>superscript</mtext></merror>', /missing argument for superscript/],
    ['double superscript', 'x^2^3', '<merror><mtext>^</mtext></merror>', /double superscript/],
    ['& outside environment', 'a & b', '<merror><mtext>&amp;</mtext></merror>', /& outside/],
    ['\\\\ outside environment', 'a \\\\ b', '<merror><mtext>\\\\</mtext></merror>', /line break/],
    ['unknown environment', '\\begin{foo} x \\end{foo}', '<merror><mtext>\\begin{foo}</mtext></merror>', /unknown environment/],
    ['missing \\end', '\\begin{aligned} a &= b', '<merror><mtext>\\begin{aligned}</mtext></merror>', /missing \\end\{aligned\}/],
    ['mismatched \\end', '\\begin{aligned} a \\end{cases}', '<merror><mtext>\\end{cases}</mtext></merror>', /ended by/],
    ['stray \\end', 'x \\end{aligned}', '<merror>', /without matching \\begin/],
    ['unexpected character', 'a # b', '<merror><mtext>#</mtext></merror>', /unexpected character/],
    ['missing \\sqrt bracket', '\\sqrt[3 x', '<merror>', /missing \]/],
    ['unclosed \\text', '\\text{abc', '<mtext>abc</mtext>', /missing } to close \\text/],
];

for (const [name, tex, fragment, re] of ERRORS) {
    test(`tex error: ${name}`, () => {
        let out;
        assert.doesNotThrow(() => { out = texToMathML(tex); });
        assert.ok(norm(out).includes(fragment), `${tex} → ${out}`);
        const errs = texErrors(tex);
        assert.ok(errs.length > 0, `expected errors for ${tex}`);
        assert.ok(errs.some((e) => re.test(e)), `${tex}: ${JSON.stringify(errs)}`);
    });
}

test('never throws on hostile or pathological input', () => {
    const inputs = ['{'.repeat(20000), '\\', '^', '_', '\\left', '\\sqrt[', '\\frac', '\\begin', '\\text', null, undefined, 42, '\\left(\\right', '$', '\u0000'];
    for (const tex of inputs) {
        assert.doesNotThrow(() => texToMathML(tex), String(tex).slice(0, 20));
        assert.doesNotThrow(() => texErrors(tex));
    }
    assert.ok(texErrors('{'.repeat(20000)).length > 0);
});

test('texErrors is empty for valid input', () => {
    assert.deepEqual(texErrors('E = mc^2'), []);
    assert.deepEqual(texErrors('\\Delta t \\le \\frac{h}{c\\sqrt{2}}'), []);
});
