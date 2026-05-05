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
      @for (c of claims(); track c.id) {
        <span class="claim" [attr.data-status]="c.status">
          <span class="claim-text">{{ c.text }}</span>
          <button
            type="button"
            class="claim-badge"
            [class]="badge(c.status).cls"
            [title]="hoverText(c)"
            [attr.aria-label]="hoverText(c)"
          >{{ badge(c.status).glyph }}</button>
        </span>
      }
      @if (anyUnsupported()) {
        <p class="claim-banner" role="status">
          ⚠ Some statements above were not supported by the corpus.
        </p>
      }
    </div>
  `,
  styles: [`
    .claim-stream { line-height: 1.6; }
    .claim { display: inline; }
    .claim-badge {
      font-size: 0.7em; vertical-align: super; margin: 0 0.15em;
      border: none; background: transparent; cursor: help;
    }
    .badge-ok   { color: #2e7d32; }
    .badge-warn { color: #c98300; }
    .badge-err  { color: #c62828; }
    .badge-mute { color: #888; }
    .claim-banner {
      margin-top: 0.75rem; font-size: 0.85em; color: #c98300;
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
