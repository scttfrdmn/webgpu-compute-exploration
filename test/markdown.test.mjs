// Markdown renderer for the Understand tab (js/teach/markdown.js, #54).
//
// Covers every block and inline construct, callouts, math, heading slugs, link resolution, the lint
// pass, and XSS cases: the renderer must never emit markup that did not come from the renderer itself.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderMarkdown, lintMarkdown, headings, safeUrl } from '../js/teach/markdown.js';

const md = (src, opts) => renderMarkdown(src, opts);
const has = (html, fragment) => assert.ok(html.includes(fragment), `\n  got:  ${html}\n  want: ${fragment}`);
const MATH_INLINE = '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">';
const MATH_BLOCK = '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">';

// Every tag in the output must be one the renderer produces (MathML included).
const ALLOWED_TAGS = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr', 'strong', 'em', 'code', 'pre', 'a', 'img',
    'ul', 'ol', 'li', 'blockquote', 'aside', 'div', 'span', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'math', 'mi', 'mn', 'mo', 'mrow', 'msup', 'msub', 'msubsup', 'munder', 'mover', 'munderover', 'mfrac',
    'msqrt', 'mroot', 'mtext', 'mspace', 'mstyle', 'merror', 'mtable', 'mtr', 'mtd']);
const ALLOWED_ATTRS = new Set(['id', 'class', 'href', 'title', 'target', 'rel', 'src', 'alt', 'loading', 'style',
    'start', 'xmlns', 'display', 'mathvariant', 'stretchy', 'fence', 'accent', 'accentunder', 'width',
    'displaystyle', 'columnalign', 'linethickness', 'minsize', 'maxsize']);

function assertInert(html) {
    for (const m of html.matchAll(/<\/?([A-Za-z][\w-]*)([^>]*)>/g)) {
        assert.ok(ALLOWED_TAGS.has(m[1].toLowerCase()), `unexpected tag <${m[1]}> in ${html}`);
        for (const a of m[2].matchAll(/\s([^\s=]+)(?:="([^"]*)")?/g)) {
            assert.ok(ALLOWED_ATTRS.has(a[1].toLowerCase()), `unexpected attribute ${a[1]} in ${m[0]}`);
            assert.ok(!/["<>]/.test(a[2] ?? ''), `unescaped attribute value in ${m[0]}`);
        }
    }
    // No raw scripts; event handlers can only appear as escaped text (attributes are allowlisted above).
    assert.ok(!/<script/i.test(html), html);
    for (const m of html.matchAll(/\s(?:href|src)="([^"]*)"/g)) {
        assert.ok(!/^\s*(javascript|data|vbscript):/i.test(m[1].replace(/&[^;]+;|[\s\u0000-\u001f]/g, '')), m[0]);
    }
}

// ---- blocks --------------------------------------------------------------------------------------------
test('headings: levels 1–4 with ids', () => {
    const html = md('# One\n## Two\n### Three\n#### Four');
    for (const [n, t] of [[1, 'one'], [2, 'two'], [3, 'three'], [4, 'four']]) has(html, `<h${n} id="${t}">`);
    assert.equal(md('#nospace'), '<p>#nospace</p>');
    assert.equal(md('## Closing hashes ##'), '<h2 id="closing-hashes">Closing hashes</h2>');
});

test('heading slugs are unique and markup-free; headings() lists them', () => {
    const src = '## Mapping to the GPU\n## Mapping to the GPU\n## Mapping to the GPU\n### Why `O(N^2)` *hurts*\n## Ünïcode café\n## !!!';
    const list = headings(src);
    assert.deepEqual(list.map((h) => h.id), ['mapping-to-the-gpu', 'mapping-to-the-gpu-2', 'mapping-to-the-gpu-3', 'why-on2-hurts', 'unicode-cafe', 'section']);
    assert.deepEqual(list.map((h) => h.level), [2, 2, 2, 3, 2, 2]);
    assert.equal(list[3].text, 'Why O(N^2) hurts');
    const html = md(src);
    for (const h of list) has(html, `id="${h.id}"`);
    assert.deepEqual(headings('# A', { idPrefix: 'nbody-' }).map((h) => h.id), ['nbody-a']);
    has(md('# A', { idPrefix: 'nbody-' }), '<h1 id="nbody-a">');
});

test('paragraphs: blank lines separate, single newlines join', () => {
    assert.equal(md('one\ntwo\n\nthree'), '<p>one\ntwo</p>\n<p>three</p>');
});

test('hard line breaks: two trailing spaces or a backslash', () => {
    assert.equal(md('a  \nb\\\nc'), '<p>a<br>\nb<br>\nc</p>');
});

test('fenced code: escaped, language class and label, $ and * literal', () => {
    const html = md('```wgsl\nlet y = a * b; // $x$ **not bold** <tag> & "q"\n```');
    assert.equal(html, '<div class="code-block"><span class="code-lang">wgsl</span><pre><code class="language-wgsl">let y = a * b; // $x$ **not bold** &lt;tag&gt; &amp; &quot;q&quot;</code></pre></div>');
    assert.equal(md('```\nplain\n```'), '<pre><code>plain</code></pre>');
    assert.equal(md('~~~js\nx\n~~~'), '<div class="code-block"><span class="code-lang">js</span><pre><code class="language-js">x</code></pre></div>');
    // a shorter inner fence does not close a longer one
    has(md('````\n```\ninner\n```\n````'), '<pre><code>```\ninner\n```</code></pre>');
    // language names cannot inject attributes
    const evil = md('```js"onmouseover="alert(1)\nx\n```');
    assertInert(evil);
    assert.ok(!evil.includes('onmouseover="'), evil);
});

test('unordered, ordered and nested lists', () => {
    assert.equal(md('- a\n- b\n* c'), '<ul>\n<li>a</li>\n<li>b</li>\n<li>c</li>\n</ul>');
    assert.equal(md('1. a\n2. b'), '<ol>\n<li>a</li>\n<li>b</li>\n</ol>');
    assert.equal(md('3. a\n4. b'), '<ol start="3">\n<li>a</li>\n<li>b</li>\n</ol>');
    assert.equal(md('- a\n  - b\n    - c\n  - d\n- e'),
        '<ul>\n<li>a\n<ul>\n<li>b\n<ul>\n<li>c</li>\n</ul></li>\n<li>d</li>\n</ul></li>\n<li>e</li>\n</ul>');
    assert.equal(md('1. step\n   - detail\n2. next'),
        '<ol>\n<li>step\n<ul>\n<li>detail</li>\n</ul></li>\n<li>next</li>\n</ol>');
    // lazy continuation and inline markup inside items
    assert.equal(md('- item *one*\ncontinues'), '<ul>\n<li>item <em>one</em>\ncontinues</li>\n</ul>');
});

test('loose lists wrap items in <p>; code fences inside items', () => {
    assert.equal(md('- a\n\n- b'), '<ul>\n<li><p>a</p></li>\n<li><p>b</p></li>\n</ul>');
    has(md('- run:\n\n  ```sh\n  npm test\n  ```'), '<code class="language-sh">npm test</code>');
});

test('a list ends at a non-indented paragraph after a blank line', () => {
    assert.equal(md('- a\n\ntext'), '<ul>\n<li>a</li>\n</ul>\n<p>text</p>');
});

test('tables with alignment, inline content and | inside math', () => {
    const html = md('| N | ms | note |\n|:--|--:|:-:|\n| 1024 | 0.5 | `a|b` |\n| 2048 | 1.1 | $|x|$ |\n| 4096 |');
    has(html, '<thead><tr><th style="text-align:left">N</th><th style="text-align:right">ms</th><th style="text-align:center">note</th></tr></thead>');
    has(html, '<td style="text-align:left">1024</td><td style="text-align:right">0.5</td><td style="text-align:center"><code>a|b</code></td>');
    has(html, `<td style="text-align:center">${MATH_INLINE}<mo>|</mo><mi>x</mi><mo>|</mo></math></td>`);
    has(html, '<tr><td style="text-align:left">4096</td><td style="text-align:right"></td><td style="text-align:center"></td></tr>');
    assert.ok(html.startsWith('<div class="table-wrap"><table>'));
    has(md('a | b\n--- | ---\n1 | 2'), '<th>a</th><th>b</th>');
    has(md('| a |\n|---|\n| x \\| y |'), '<td>x | y</td>');
});

test('blockquotes and horizontal rules', () => {
    assert.equal(md('> quoted\n> more'), '<blockquote>\n<p>quoted\nmore</p>\n</blockquote>');
    assert.equal(md('> outer\n>\n> > inner'), '<blockquote>\n<p>outer</p>\n<blockquote>\n<p>inner</p>\n</blockquote>\n</blockquote>');
    for (const hr of ['---', '***', '___', '- - -']) assert.equal(md(hr), '<hr>', hr);
});

test('callouts: all five types', () => {
    const types = { 'GPU': 'gpu', 'Numerics': 'numerics', 'Try this': 'try', 'Note': 'note', 'Warning': 'warning' };
    for (const [label, cls] of Object.entries(types)) {
        assert.equal(md(`> **${label}:** body *text*`),
            `<aside class="callout callout-${cls}">\n<p><strong class="callout-label">${label}:</strong> body <em>text</em></p>\n</aside>`);
    }
});

test('callouts: multi-paragraph body, label alone on its line, lists inside', () => {
    assert.equal(md('> **GPU:** one\n>\n> two'),
        '<aside class="callout callout-gpu">\n<p><strong class="callout-label">GPU:</strong> one</p>\n<p>two</p>\n</aside>');
    assert.equal(md('> **Try this:**\n> - lower θ'),
        '<aside class="callout callout-try">\n<p><strong class="callout-label">Try this:</strong></p>\n<ul>\n<li>lower θ</li>\n</ul>\n</aside>');
});

test('unknown callout labels render as a plain blockquote and are linted', () => {
    const src = '> **Tip:** hello';
    assert.equal(md(src), '<blockquote>\n<p><strong>Tip:</strong> hello</p>\n</blockquote>');
    const problems = lintMarkdown(src);
    assert.equal(problems.length, 1);
    assert.match(problems[0].message, /unknown callout type "Tip"/);
    assert.equal(problems[0].line, 1);
});

test('display math: single-line and multi-line $$ paragraphs', () => {
    const wave = md('$$\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u$$');
    assert.ok(wave.startsWith(MATH_BLOCK), wave);
    has(wave, '<mfrac><mrow><msup><mi>∂</mi><mn>2</mn></msup><mi>u</mi></mrow>');
    const multi = md('Text before.\n\n$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$\n\nAfter.');
    has(multi, `<p>Text before.</p>\n${MATH_BLOCK}<mtable`);
    has(multi, '</math>\n<p>After.</p>');
    // $$ directly after a paragraph line still starts display math
    has(md('Energy:\n$$E = mc^2$$'), `<p>Energy:</p>\n${MATH_BLOCK}`);
});

// ---- inline --------------------------------------------------------------------------------------------
test('inline code, bold, emphasis', () => {
    assert.equal(md('`a < b` **bold** __also__ *em* _em_'), '<p><code>a &lt; b</code> <strong>bold</strong> <strong>also</strong> <em>em</em> <em>em</em></p>');
    assert.equal(md('**bold *nested em* here**'), '<p><strong>bold <em>nested em</em> here</strong></p>');
    assert.equal(md('*em **nested bold** here*'), '<p><em>em <strong>nested bold</strong> here</em></p>');
    assert.equal(md('***both***'), '<p><strong><em>both</em></strong></p>');
    assert.equal(md('``code with ` tick``'), '<p><code>code with ` tick</code></p>');
});

test('emphasis edge cases stay literal', () => {
    assert.equal(md('snake_case_name and a_b'), '<p>snake_case_name and a_b</p>');
    assert.equal(md('2 * 3 * 4'), '<p>2 * 3 * 4</p>');
    assert.equal(md('**unclosed'), '<p>**unclosed</p>');
    assert.equal(md('`unclosed code'), '<p>`unclosed code</p>');
    assert.equal(md('\\*not em\\*'), '<p>*not em*</p>');
});

test('inline math $…$ and the price heuristic', () => {
    assert.equal(md('Energy $E = mc^2$ here.'), `<p>Energy ${MATH_INLINE}<mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></math> here.</p>`);
    assert.equal(md('It costs $5 each, or $10 total.'), '<p>It costs $5 each, or $10 total.</p>');
    assert.equal(md('$5 now and $x$ later'), `<p>$5 now and ${MATH_INLINE}<mi>x</mi></math> later</p>`);
    assert.equal(md('from $1,000.'), '<p>from $1,000.</p>');
    assert.equal(md('a $ b $ c'), '<p>a $ b $ c</p>'); // space after opening / before closing
    assert.equal(md('\\$x\\$'), '<p>$x$</p>');
    assert.equal(md('$2$ items'), `<p>${MATH_INLINE}<mn>2</mn></math> items</p>`);
    has(md('cost $\\$5$ ok'), `${MATH_INLINE}<mo>$</mo><mn>5</mn></math>`);
    assert.equal(md('`$x$`'), '<p><code>$x$</code></p>');
    // emphasis characters inside math are not markdown
    has(md('$a*b*c$'), '<mo>∗</mo><mi>b</mi><mo>∗</mo>');
    has(md('$x_i y_j$'), '<msub><mi>x</mi><mi>i</mi></msub><msub><mi>y</mi><mi>j</mi></msub>');
    has(md('*see $x*y$*'), `<em>see ${MATH_INLINE}`);
});

test('links: allowed schemes, titles, external target, resolveLink', () => {
    assert.equal(md('[a](https://example.com/x?y=1&z=2)'), '<p><a href="https://example.com/x?y=1&amp;z=2" target="_blank" rel="noopener noreferrer">a</a></p>');
    assert.equal(md('[a](#nbody/measure)'), '<p><a href="#nbody/measure">a</a></p>');
    assert.equal(md('[a](./learn.html "Path")'), '<p><a href="./learn.html" title="Path">a</a></p>');
    assert.equal(md('[a](js/examples/nbody.js)'), '<p><a href="js/examples/nbody.js">a</a></p>');
    assert.equal(md('[a](<with space.html>)'), '<p><a href="with space.html">a</a></p>');
    assert.equal(md('[f(x)](https://en.wikipedia.org/wiki/F_(x))'), '<p><a href="https://en.wikipedia.org/wiki/F_(x)" target="_blank" rel="noopener noreferrer">f(x)</a></p>');
    assert.equal(md('[**bold** `code`](#x)'), '<p><a href="#x"><strong>bold</strong> <code>code</code></a></p>');
    const resolveLink = (h) => h.startsWith('explore:') ? `#${h.slice(8)}` : h.startsWith('src:') ? `https://github.com/o/r/blob/main/${h.slice(4)}` : h;
    assert.equal(md('[try](explore:nbody)', { resolveLink }), '<p><a href="#nbody">try</a></p>');
    has(md('[code](src:js/examples/nbody.js)', { resolveLink }), 'href="https://github.com/o/r/blob/main/js/examples/nbody.js"');
    // resolver output is still validated
    assert.equal(md('[x](#a)', { resolveLink: () => 'javascript:alert(1)' }), '<p>x</p>');
    assert.equal(md('[x](#a)', { resolveLink: () => { throw new Error('boom'); } }), '<p>x</p>');
    assert.equal(md('[not a link] (x)'), '<p>[not a link] (x)</p>');
});

test('images: lazy, alt escaped, same URL rule', () => {
    assert.equal(md('![A *plot* "q"](img/plot.png)'), '<p><img src="img/plot.png" alt="A plot &quot;q&quot;" loading="lazy"></p>');
    assert.equal(md('![alt](https://x.org/a.png "T")'), '<p><img src="https://x.org/a.png" alt="alt" title="T" loading="lazy"></p>');
    assert.equal(md('![alt](data:image/svg+xml,%3Csvg%20onload=alert(1)%3E)'), '<p>alt</p>');
    assert.equal(md('![alt](data:image/svg+xml,<svg onload=alert(1)>)'), '<p>![alt](data:image/svg+xml,&lt;svg onload=alert(1)&gt;)</p>');
    has(md('[![badge](b.svg)](https://ci)'), '<a href="https://ci" target="_blank" rel="noopener noreferrer"><img src="b.svg" alt="badge" loading="lazy"></a>');
});

test('safeUrl policy', () => {
    for (const ok of ['https://a.b', 'http://a.b', 'HTTPS://A.B', '#frag', 'rel/path.md', './x', '../y', '/abs', '?q=1', 'a:b/../c'.replace('a:', 'a'), 'file.html#x'])
        assert.notEqual(safeUrl(ok), null, ok);
    for (const bad of ['javascript:alert(1)', 'JaVaScRiPt:alert(1)', ' javascript:x', 'java\tscript:x', 'java\nscript:x', '\u0001javascript:x',
        'data:text/html,x', 'vbscript:x', 'mailto:a@b', 'file:///etc/passwd', '//evil.com', '\\\\evil.com', '/\\evil.com', '', '   '])
        assert.equal(safeUrl(bad), null, JSON.stringify(bad));
});

// ---- security ------------------------------------------------------------------------------------------
const XSS = [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<a href="javascript:alert(1)">x</a>',
    '[click](javascript:alert(1))',
    '[click](JAVASCRIPT:alert(1))',
    '[click]( javascript:alert(1) )',
    '[click](java&#115;cript:alert(1))',
    '[click](&#106;avascript:alert(1))',
    '[click](data:text/html;base64,PHNjcmlwdD4=)',
    '[click](vbscript:msgbox)',
    '[x](https://ok.com" onmouseover="alert(1))',
    '[x](https://ok.com "a\\" onmouseover=\\"alert(1)")',
    '[x](#a "t\' onmouseover=\'alert(1)")',
    '![x" onerror="alert(1)](a.png)',
    '![x](a.png" onerror="alert(1))',
    '`<script>alert(1)</script>`',
    '```html\n<script>alert(1)</script>\n```',
    '# <img src=x onerror=alert(1)>',
    '> **GPU:** <iframe src="javascript:alert(1)">',
    '| <b>h</b> |\n|---|\n| <script>x</script> |',
    '- <svg onload=alert(1)>',
    '$\\text{<script>alert(1)</script>}$',
    '$$\\text{</math><script>alert(1)</script>}$$',
    '<![CDATA[ x ]]> ]]> <!-- comment -->',
    '&lt;script&gt; &amp; &#60;script&#62; &quot;',
    '<style>body{display:none}</style>',
    '[a](#x)<a onclick=1>',
    '**<b onmouseover=alert(1)>**',
];

for (const src of XSS) {
    test(`XSS inert: ${JSON.stringify(src).slice(0, 60)}`, () => {
        const html = md(src);
        assertInert(html);
        // Raw < from the source never survives as a tag start.
        const stripped = html.replace(/<\/?([A-Za-z][\w-]*)(\s[^<>]*)?>/g, (m, t) => ALLOWED_TAGS.has(t.toLowerCase()) ? '' : m);
        assert.ok(!stripped.includes('<'), `${src} → ${html}`);
    });
}

test('HTML entities in source are shown literally (double-escaped)', () => {
    assert.equal(md('&amp; &lt;b&gt; &#60;'), '<p>&amp;amp; &amp;lt;b&amp;gt; &amp;#60;</p>');
    assert.equal(md(']]>'), '<p>]]&gt;</p>');
});

test('dangerous links render as text only', () => {
    assert.equal(md('[click me](javascript:alert(1))'), '<p>click me</p>');
    assert.equal(md('![pic](javascript:alert(1))'), '<p>pic</p>');
});

// ---- lint ----------------------------------------------------------------------------------------------
test('lint: clean document has no problems', () => {
    const src = '# Title\n\nText $x^2$ and [link](#a).\n\n> **Numerics:** fine\n\n$$\\sum_i x_i$$\n\n```js\nx\n```';
    assert.deepEqual(lintMarkdown(src), []);
});

test('lint: bad inline and display math with line numbers', () => {
    const problems = lintMarkdown('# T\n\nok $\\frac{a}$ here\n\n$$\n\\foo{x}\n$$');
    assert.equal(problems.length, 2, JSON.stringify(problems));
    assert.equal(problems[0].line, 3);
    assert.match(problems[0].message, /math: missing argument for \\frac/);
    assert.equal(problems[1].line, 5);
    assert.match(problems[1].message, /math: unknown command \\foo/);
    // and the rendered output shows the error in place
    has(md('$\\foo$'), '<merror><mtext>\\foo</mtext></merror>');
});

test('lint: inline problems report the line they are on within a paragraph', () => {
    const problems = lintMarkdown('line one\nline two $\\bad$\nline three [x](javascript:y)');
    assert.deepEqual(problems.map((p) => p.line), [2, 3]);
});

test('lint: unclosed fence and unclosed $$', () => {
    const fence = lintMarkdown('text\n\n```js\nnever closed');
    assert.deepEqual(fence, [{ line: 3, message: 'unclosed code fence' }]);
    has(md('```js\nnever closed'), '<code class="language-js">never closed</code>');
    const dd = lintMarkdown('$$\nx = 1\n\nnext');
    assert.equal(dd.length, 1);
    assert.match(dd[0].message, /unclosed \$\$/);
});

test('lint: empty links and unsafe URLs', () => {
    const problems = lintMarkdown('[](#a) [text]() ![alt]() [bad](javascript:x) [mail](mailto:a@b)');
    const msgs = problems.map((p) => p.message);
    assert.ok(msgs.some((m) => /empty link text/.test(m)), msgs.join('\n'));
    assert.ok(msgs.some((m) => /empty link URL/.test(m)), msgs.join('\n'));
    assert.ok(msgs.some((m) => /empty image URL/.test(m)), msgs.join('\n'));
    assert.ok(msgs.some((m) => /unsafe link URL: javascript:x/.test(m)), msgs.join('\n'));
    assert.ok(msgs.some((m) => /unsafe link URL: mailto:a@b/.test(m)), msgs.join('\n'));
    // empty link text falls back to showing the URL
    assert.equal(md('[](#a)'), '<p><a href="#a">#a</a></p>');
});

test('robust on odd input (CRLF, tabs, empty, non-string)', () => {
    assert.equal(md('a\r\nb'), '<p>a\nb</p>');
    assert.equal(md('-\titem'), '<ul>\n<li>item</li>\n</ul>');
    assert.equal(md(''), '');
    assert.equal(md(undefined), '');
    assert.doesNotThrow(() => md('['.repeat(5000) + '*'.repeat(5000) + '$'.repeat(5000) + '`'.repeat(5000)));
});

test('pathological inline input stays fast (no cubic rescans)', () => {
    const inputs = ['['.repeat(4000) + '`'.repeat(4000), '$a '.repeat(4000), '`x '.repeat(4000) + '*', '*a '.repeat(4000), '_a '.repeat(4000)];
    for (const src of inputs) {
        const t0 = performance.now();
        md(src);
        const ms = performance.now() - t0;
        assert.ok(ms < 2000, `${JSON.stringify(src.slice(0, 12))}… took ${ms.toFixed(0)} ms`);
    }
});

test('sample: an Understand-style document', () => {
    const src = [
        '## The model',
        '',
        'The displacement $u(x, t)$ obeys',
        '',
        '$$\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u$$',
        '',
        '> **GPU:** one invocation per grid cell; neighbours read from the previous buffer.',
        '',
        '| Grid | Step (ms) |',
        '|:-----|----------:|',
        '| 256² | 0.12 |',
    ].join('\n');
    const html = md(src);
    has(html, '<h2 id="the-model">The model</h2>');
    has(html, `<p>The displacement ${MATH_INLINE}`);
    has(html, `${MATH_BLOCK}<mfrac>`);
    has(html, '<aside class="callout callout-gpu">\n<p><strong class="callout-label">GPU:</strong> one invocation per grid cell;');
    has(html, '<th style="text-align:right">Step (ms)</th>');
    assert.deepEqual(lintMarkdown(src), []);
    assertInert(html);
});

// Regression: math that starts with a number used to be taken for a price and left as raw TeX
// (seen on the N-body Understand page: "$2.7 \times 10^{8}$", "$0 \dots N-2$").
test('inline math starting with a number renders when its content is TeX', () => {
    for (const src of ['$2.7 \\times 10^{8}$', '$0 \\dots N-2$', '$8.8 \\times 10^{-7}$', '$2 = x$', '$3 x_i$']) {
        const html = renderMarkdown(`about ${src} here`);
        assert.match(html, /<math /, src);
        assert.doesNotMatch(html.replace(/<math[\s\S]*?<\/math>/g, ''), /\$/, `${src}: no literal $ left`);
    }
    // prices still stay literal
    for (const src of ['It costs $5 each.', 'Pay $1,000. Then $20 more.', 'between $3 and $4 ']) {
        assert.doesNotMatch(renderMarkdown(src), /<math /, src);
    }
});
