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
      <ul class="claim-list">
        @for (c of claims(); track c.id) {
          <li
            class="claim-item"
            [attr.data-status]="c.status"
            [title]="hoverText(c)"
          >
            <span
              class="claim-marker"
              [class]="badge(c.status).cls"
              [attr.aria-label]="badge(c.status).title"
            >{{ badge(c.status).glyph }}</span>
            <span class="claim-text">{{ c.text }}</span>
          </li>
        }
      </ul>
      @if (anyUnsupported()) {
        <p class="claim-banner" role="status">
          ⚠ Some statements above were not supported by the retrieved sources.
        </p>
      }
    </div>
  `,
  styles: [`
    .claim-stream { line-height: 1.5; }
    .claim-list {
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .claim-item {
      display: grid;
      grid-template-columns: 1.1rem 1fr;
      gap: 0.55rem;
      align-items: baseline;
      cursor: help;
      padding: 0.15rem 0;
      border-radius: 4px;
      transition: background 120ms ease;
    }
    .claim-item:hover { background: rgba(255, 255, 255, 0.03); }
    .claim-marker {
      font-size: 0.85em;
      line-height: 1.4;
      text-align: center;
      user-select: none;
      font-weight: 600;
    }
    .claim-text { color: inherit; }
    .badge-ok   { color: #4ade80; }
    .badge-warn { color: #fbbf24; }
    .badge-err  { color: #f87171; }
    .badge-mute { color: #9ca3af; }
    /* Subtle left-edge tint reinforces status without dominating the row. */
    .claim-item[data-status="unsupported"] {
      box-shadow: inset 2px 0 0 rgba(251, 191, 36, 0.5);
      padding-left: 0.4rem;
    }
    .claim-item[data-status="contradicted"] {
      box-shadow: inset 2px 0 0 rgba(248, 113, 113, 0.6);
      padding-left: 0.4rem;
    }
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
