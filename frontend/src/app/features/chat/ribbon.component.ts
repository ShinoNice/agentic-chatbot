import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  GuardrailsReport,
  VerificationDetail,
} from '../../models/chat.model';

@Component({
  selector: 'app-ribbon',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (verification && !verification.supported) {
      <div
        class="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
      >
        <span class="mt-px">⚠</span>
        <div class="flex-1 min-w-0">
          <div class="font-medium">Verifier flagged this answer</div>
          <div class="text-amber-300/80">{{ verifierBlurb() }}</div>
        </div>
      </div>
    }
    @if (guardrailsActive()) {
      <div
        class="flex items-start gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-xs text-violet-200"
      >
        <span class="mt-px">🛡</span>
        <div class="flex-1 min-w-0">
          <div class="font-medium">PII redacted</div>
          <div class="text-violet-300/80">
            @if (guardrails!.input_redactions) {
              <span>{{ guardrails!.input_redactions }} in question</span>
            }
            @if (guardrails!.input_redactions && guardrails!.output_redactions) {
              <span> · </span>
            }
            @if (guardrails!.output_redactions) {
              <span>{{ guardrails!.output_redactions }} in answer</span>
            }
          </div>
        </div>
      </div>
    }
    @if (relevanceStatus === 'NO_MATCH') {
      <div
        class="flex items-start gap-2 rounded-lg border border-zinc-700 bg-zinc-800/60 px-3 py-2 text-xs text-zinc-300"
      >
        <span class="mt-px">ℹ</span>
        <div>No relevant documents found for this question.</div>
      </div>
    }
  `,
})
export class RibbonComponent {
  @Input() verification: VerificationDetail | null = null;
  @Input() guardrails: GuardrailsReport | null = null;
  @Input() relevanceStatus: string | undefined;

  guardrailsActive(): boolean {
    return !!(
      this.guardrails &&
      (this.guardrails.input_pii_found || this.guardrails.output_pii_found)
    );
  }

  verifierBlurb(): string {
    if (!this.verification) return '';
    const v = this.verification;
    const u = v.unsupported_claims.length;
    const c = v.contradictions.length;
    const parts: string[] = [];
    if (u) parts.push(`${u} unsupported claim${u === 1 ? '' : 's'}`);
    if (c) parts.push(`${c} contradiction${c === 1 ? '' : 's'}`);
    return parts.join(' · ') || 'Open the inspector for details.';
  }
}
