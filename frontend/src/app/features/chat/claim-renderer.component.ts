import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { Claim, ClaimStatus } from '../../core/claim.types';

const BADGE: Record<ClaimStatus, { glyph: string; cls: string; title: string }> = {
  verified:     { glyph: '✓', cls: 'badge-ok',   title: 'verified by source' },
  unsupported:  { glyph: '⚠', cls: 'badge-warn', title: 'unsupported by source' },
  contradicted: { glyph: '✕', cls: 'badge-err',  title: 'contradicted by source' },
  pending:      { glyph: '…', cls: 'badge-mute', title: 'verifying…' },
};

@Component({
  selector: 'app-claim-renderer',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="claim-stream">
      <p class="claim-prose">
        @for (c of claims(); track c.id; let last = $last) {<span
          class="claim"
          [attr.data-status]="c.status"
        ><span class="claim-text">{{ c.text }}</span><sup
          class="claim-badge"
          [class]="badge(c.status).cls"
          [title]="hoverText(c)"
          [attr.aria-label]="hoverText(c)"
          tabindex="0"
        >({{ badge(c.status).glyph }})</sup></span>@if (!last) {{{ ' ' }}}}
      </p>
      @if (anyUnsupported()) {
        <p class="claim-banner" role="status">
          ⚠ Some statements above were not supported by the retrieved sources.
        </p>
      }
    </div>
  `,
  styles: [`
    .claim-stream { line-height: 1.7; }
    .claim-prose {
      margin: 0;
      text-align: left;
      hyphens: auto;
    }
    .claim { display: inline; }
    .claim-text {
      /* Subtle underline-on-hover signals the badge that follows is a citation. */
    }
    .claim:hover .claim-text {
      background: rgba(255, 255, 255, 0.025);
      border-radius: 2px;
    }
    /* Footnote-style superscript: small parens around a status glyph,
       sitting just above baseline without stretching line-height. */
    .claim-badge {
      font-size: 0.62em;
      line-height: 0;
      margin-left: 0.15em;
      cursor: help;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0.02em;
      user-select: none;
    }
    .claim-badge:focus { outline: 1px dotted currentColor; outline-offset: 2px; }
    .badge-ok   { color: #4ade80; }   /* green-400 — readable on dark */
    .badge-warn { color: #fbbf24; }   /* amber-400 */
    .badge-err  { color: #f87171; }   /* red-400 */
    .badge-mute { color: #9ca3af; }   /* gray-400 */
    .claim-banner {
      margin: 0.75rem 0 0;
      font-size: 0.85em;
      color: #fbbf24;
      border-top: 1px solid rgba(251, 191, 36, 0.2);
      padding-top: 0.5rem;
    }
  `],
})
export class ClaimRendererComponent {
  claims = input.required<Claim[]>();

  badge(s: ClaimStatus) { return BADGE[s]; }

  anyUnsupported = computed(() =>
    this.claims().some(c => c.status === 'unsupported' || c.status === 'contradicted')
  );

  hoverText(c: Claim): string {
    const head = BADGE[c.status].title;
    if (!c.citations?.length) return head;
    const cites = c.citations.map(ci => `${ci.chunk_id}: "${ci.quote}"`).join('\n');
    return `${head}\n\n${cites}`;
  }
}
