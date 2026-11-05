// Reading-level checks for the Understand pages at each visitor level (#90), used by
// test/teach-levels.test.mjs. Pure functions, no DOM.
//
// fleschKincaidGrade is the US school grade of a text (Kincaid et al., 1975):
//   0.39 · (words / sentences) + 11.8 · (syllables / words) − 15.59
// Syllables are estimated by counting vowel groups (a silent final "e" does not count, every word has at
// least one), which is within about one syllable per long word and good enough for a grade bound.
// Flesch–Kincaid only sees sentence and word length: the Advanced pages, written in short sentences, score
// grade 5–7 on it although their ideas are college level. The share of long words (three syllables or more)
// tracks vocabulary better: 8–15% on the Advanced pages, under ~7% in middle-school text. Both are bounds,
// not proof: the real check of a level is reading it.

// Rules per level: the highest Flesch–Kincaid grade and long-word share allowed, whether TeX math ($…$) may
// appear, and words to avoid (matched as whole words, case-insensitive). Advanced pages are not checked.
export const LEVEL_RULES = {
    beginner: {
        maxGrade: 6,
        maxLongWords: 0.07,
        math: false,
        avoid: ['asymptotic', 'asymptotically', 'derivative', 'integrator', 'eigenvalue', 'stochastic', 'heuristic',
            'logarithm', 'logarithmic', 'complexity', 'arithmetic intensity', 'synchronization', 'latency', 'O(N'],
    },
    intermediate: {
        maxGrade: 8,
        maxLongWords: 0.10,
        math: true,
        avoid: ['asymptotically', 'eigenvalue', 'Jacobian', 'stochastic', 'Higham'],
    },
};

// The prose of a Markdown page: no code blocks, inline code, math, link targets, headings markers,
// table rules or callout labels
export function plainText(markdown) {
    return String(markdown)
        .replace(/```[\s\S]*?```/g, ' ')
        .replace(/\$\$[\s\S]*?\$\$/g, ' ')
        .replace(/\$[^$\n]+\$/g, ' ')
        .replace(/`[^`]*`/g, ' ')
        .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
        .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
        .replace(/^\s*\|?\s*:?-{3,}.*$/gm, ' ')
        .replace(/^#{1,6}\s+(.*)$/gm, '$1.')
        // list items and table rows are sentences of their own
        .replace(/^\s*(?:[-*]|\d+\.)\s+(.*?)([.!?])?\s*$/gm, (_, item, end) => `${item}${end ?? '.'}`)
        .replace(/^\s*\|(.*)\|\s*$/gm, (_, row) => `${row.split('|').map(c => c.trim()).filter(Boolean).join('. ')}.`)
        .replace(/^>\s*\*\*[^*]+:\*\*/gm, ' ')
        .replace(/[*_>|#]/g, ' ')
        // full stops that do not end a sentence: decimals, abbreviations, ellipses
        .replace(/(\d)\.(?=\d)/g, '$1')
        .replace(/\b(e\.g|i\.e|etc|vs|cf|approx|Fig|et al)\./gi, (_, a) => a.replace(/\./g, ''))
        .replace(/\.{2,}|…/g, ' ');
}

export function countSyllables(word) {
    const w = word.toLowerCase().replace(/[^a-z]/g, '');
    if (!w) return 0;
    const groups = w.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '').replace(/^y/, '').match(/[aeiouy]{1,2}/g);
    return Math.max(1, groups ? groups.length : 0);
}

// { words, sentences, syllables, grade, longWords } of a text: longWords is the share of words with three or
// more syllables (grade and longWords null when it has no words)
export function readability(text) {
    const words = String(text).match(/[A-Za-z][A-Za-z'’-]*/g) ?? [];
    const sentences = Math.max(1, (String(text).match(/[^.!?]+[.!?]+/g) ?? []).filter(s => /[A-Za-z]/.test(s)).length);
    const syllables = words.reduce((sum, w) => sum + countSyllables(w), 0);
    const grade = words.length ? 0.39 * (words.length / sentences) + 11.8 * (syllables / words.length) - 15.59 : null;
    const longWords = words.length ? words.filter(w => countSyllables(w) >= 3).length / words.length : null;
    return { words: words.length, sentences, syllables, grade, longWords };
}

export const fleschKincaidGrade = (text) => readability(text).grade;

// Terms from rules.avoid that appear in the text (whole words; "O(N" as written)
export function avoidedTerms(text, avoid) {
    return avoid.filter(term => {
        const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        return new RegExp(/^\w/.test(term) ? `\\b${escaped}\\b` : escaped, 'i').test(text);
    });
}
