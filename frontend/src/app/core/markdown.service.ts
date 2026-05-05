import { Injectable } from '@angular/core';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

const CITE_RE = /\[(\d+)\]/g;

// Default DOMPurify allowed attribute set is broad and safe; we extend it only
// with `data-cite` (used by our citation chip rendering). `<sup>` is already
// in the default tag allowlist, so no ADD_TAGS is needed.
const SANITIZE_CONFIG = {
  ADD_ATTR: ['data-cite'],
  // Defence-in-depth: forbid script and event handlers explicitly even though
  // the default config strips them.
  FORBID_TAGS: ['script', 'style'],
  FORBID_ATTR: ['onerror', 'onload', 'onclick'],
};

@Injectable({ providedIn: 'root' })
export class MarkdownService {
  render(text: string, citationCount: number): string {
    const out = marked.parse(text || '', { async: false });
    if (typeof out !== 'string') {
      // marked is configured for sync mode; if a future config change makes it
      // async, fail loudly rather than silently rendering "[object Promise]".
      throw new Error('MarkdownService: expected sync string from marked');
    }
    const withChips = out.replace(CITE_RE, (_match, n: string) => {
      const idx = Math.max(1, Math.min(citationCount || 1, +n));
      return `<sup class="cite-chip" data-cite="${idx}">[${idx}]</sup>`;
    });
    return DOMPurify.sanitize(withChips, SANITIZE_CONFIG);
  }
}
