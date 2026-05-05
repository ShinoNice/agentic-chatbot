import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SessionStore } from '../../core/session-store.service';

interface Stage {
  key: string;
  label: string;
  nodes: string[];
}

const STAGES: Stage[] = [
  {
    key: 'retrieve',
    label: 'Retrieve',
    nodes: ['guardrails_input', 'retrieve', 'rerank', 'check_relevance'],
  },
  // 'research' is the legacy single-shot drafter; 'draft_claims' is the new
  // claim-grounded drafter. Both map to the Draft stage during rollout.
  { key: 'draft', label: 'Draft', nodes: ['research', 'draft_claims'] },
  {
    key: 'verify',
    label: 'Verify',
    // Legacy: 'guardrails_output' + 'verify'.
    // New claim pipeline: 'verify_claims', 'repair_claims', 'finalize'.
    nodes: [
      'guardrails_output',
      'verify',
      'verify_claims',
      'repair_claims',
      'finalize',
    ],
  },
];

type StageState = 'pending' | 'active' | 'done';

@Component({
  selector: 'app-pipeline-stepper',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (visible()) {
      <div
        class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900/70 border border-zinc-800 backdrop-blur text-[11px] font-mono shadow-lg"
      >
        @for (stage of stages; track stage.key; let last = $last) {
          @let st = state(stage.key);
          <div
            class="flex items-center gap-1.5"
            [class.text-violet-300]="st === 'active'"
            [class.text-zinc-500]="st === 'pending'"
            [class.text-zinc-300]="st === 'done'"
          >
            <span
              class="relative inline-flex h-1.5 w-1.5 rounded-full"
              [class.bg-violet-500]="st === 'active'"
              [class.bg-zinc-600]="st === 'pending'"
              [class.bg-emerald-500]="st === 'done'"
              [class.pulse-ring]="st === 'active'"
            ></span>
            <span class="uppercase tracking-wider">{{ stage.label }}</span>
            @if (stage.key === 'verify' && iteration() > 1) {
              <span class="text-violet-300">×{{ iteration() }}</span>
            }
          </div>
          @if (!last) {
            <span class="text-zinc-700">→</span>
          }
        }
      </div>
    }
  `,
})
export class PipelineStepperComponent {
  private store = inject(SessionStore);
  readonly stages = STAGES;
  private warnedNodes = new Set<string>();

  readonly visible = computed(() => this.store.streamingStep() !== null);
  readonly iteration = computed(
    () => this.store.streamingStep()?.iteration ?? 1,
  );

  state(stageKey: string): StageState {
    const step = this.store.streamingStep();
    if (!step) return 'pending';
    const idx = STAGES.findIndex((s) => s.nodes.includes(step.node));
    const here = STAGES.findIndex((s) => s.key === stageKey);
    if (idx === -1) {
      if (!this.warnedNodes.has(step.node)) {
        this.warnedNodes.add(step.node);
        console.warn(
          '[PipelineStepper] unknown pipeline node, falling back to pending:',
          step.node,
        );
      }
      return 'pending';
    }
    if (here < idx) return 'done';
    if (here === idx) return 'active';
    return 'pending';
  }
}
