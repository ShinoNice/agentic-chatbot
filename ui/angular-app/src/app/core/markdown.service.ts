import { Injectable } from '@angular/core';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

const CITE_RE = /\[(\d+)\]/g;

@Injectable({ providedIn: 'root' })
export class MarkdownService {
  render(text: string, citationCount: number): string {
    const html = marked.parse(text || '', { async: false }) as string;
    const withChips = html.replace(CITE_RE, (_match, n: string) => {
      const idx = Math.max(1, Math.min(citationCount || 1, +n));
      return `<sup class="cite-chip" data-cite="${idx}">[${idx}]</sup>`;
    });
    return DOMPurify.sanitize(withChips, {
      ADD_TAGS: ['sup'],
      ADD_ATTR: ['data-cite', 'class'],
    });
  }
}
