import {
  Component,
  Input,
  Output,
  EventEmitter,
  inject,
  signal,
  OnChanges,
  SimpleChanges,
  DestroyRef,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { ChatMessage, AuditEvent } from '../../models/chat.model';
import { ApiService } from '../../core/api.service';
import { SessionStore } from '../../core/session-store.service';

type Tab = 'sources' | 'verification' | 'guardrails' | 'audit';

@Component({
  selector: 'app-inspector',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (msg) {
      <aside
        class="flex h-full w-full max-w-md flex-col border-l border-zinc-800 bg-zinc-950/95 backdrop-blur"
      >
        <div
          class="flex items-start justify-between gap-3 px-4 py-3 border-b border-zinc-800"
        >
          <div class="min-w-0">
            <div
              class="text-[10px] uppercase tracking-wider text-zinc-500 font-mono"
            >
              Message inspector
            </div>
            <div class="text-sm text-zinc-200 truncate">{{ snippet() }}</div>
          </div>
          <button
            (click)="close.emit()"
            class="text-zinc-500 hover:text-zinc-200 text-xl leading-none -mt-1"
            aria-label="Close inspector"
          >
            ×
          </button>
        </div>

        <nav class="flex border-b border-zinc-800 text-[10px] font-mono">
          @for (t of tabs; track t.key) {
            <button
              (click)="select(t.key)"
              class="flex-1 px-2 py-2 uppercase tracking-wider transition-colors"
              [class.text-violet-300]="active() === t.key"
              [class.border-b-2]="active() === t.key"
              [class.border-violet-500]="active() === t.key"
              [class.text-zinc-500]="active() !== t.key"
              [class.hover:text-zinc-300]="active() !== t.key"
            >
              {{ t.label }}
              @if (t.count > 0) {
                <span class="text-[9px] opacity-70">{{ t.count }}</span>
              }
            </button>
          }
        </nav>

        <div class="flex-1 overflow-y-auto p-4 text-sm text-zinc-300 space-y-3">
          @switch (active()) {
            @case ('sources') {
              @if (!msg.sources?.length) {
                <div class="text-zinc-500 text-xs">No sources cited.</div>
              }
              @for (s of msg.sources ?? []; track $index) {
                <div class="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                  <div class="font-mono text-[10px] text-violet-300 mb-1">
                    [{{ $index + 1 }}] {{ s.source }}
                    @if (s.page_number) {
                      <span> · p.{{ s.page_number }}</span>
                    }
                  </div>
                  <div class="text-xs text-zinc-300 leading-relaxed">
                    {{ s.snippet }}…
                  </div>
                </div>
              }
            }
            @case ('verification') {
              @if (!msg.verification) {
                <div class="text-zinc-500 text-xs">No verification report.</div>
              } @else {
                <div class="text-xs">
                  Status:
                  <span
                    class="font-medium"
                    [class.text-emerald-400]="msg.verification.supported"
                    [class.text-amber-400]="!msg.verification.supported"
                  >
                    {{ msg.verification.supported ? 'Supported' : 'Issues found' }}
                  </span>
                </div>
                @if (msg.verification.unsupported_claims.length) {
                  <div class="space-y-1">
                    <div class="text-[10px] font-mono uppercase text-zinc-500">
                      Unsupported claims
                    </div>
                    @for (c of msg.verification.unsupported_claims; track c) {
                      <div
                        class="text-xs rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-200 p-2"
                      >
                        {{ c }}
                      </div>
                    }
                  </div>
                }
                @if (msg.verification.contradictions.length) {
                  <div class="space-y-1">
                    <div class="text-[10px] font-mono uppercase text-zinc-500">
                      Contradictions
                    </div>
                    @for (c of msg.verification.contradictions; track c) {
                      <div
                        class="text-xs rounded-md bg-rose-500/10 border border-rose-500/30 text-rose-200 p-2"
                      >
                        {{ c }}
                      </div>
                    }
                  </div>
                }
                @if (msg.verification.additional_details) {
                  <div class="text-xs text-zinc-400 italic">
                    {{ msg.verification.additional_details }}
                  </div>
                }
              }
            }
            @case ('guardrails') {
              @if (
                !msg.guardrails ||
                (!msg.guardrails.input_pii_found &&
                  !msg.guardrails.output_pii_found)
              ) {
                <div class="text-zinc-500 text-xs">No PII detected.</div>
              } @else {
                <div class="text-xs">
                  Input redactions: {{ msg.guardrails.input_redactions }}
                </div>
                <div class="text-xs">
                  Output redactions: {{ msg.guardrails.output_redactions }}
                </div>
                @if (msg.guardrails.detections.length) {
                  <div>
                    <div
                      class="text-[10px] font-mono uppercase text-zinc-500 mt-2 mb-1"
                    >
                      Detected types
                    </div>
                    <div class="flex flex-wrap gap-1">
                      @for (
                        d of msg.guardrails.detections;
                        track $index
                      ) {
                        <span
                          class="rounded-md bg-violet-500/10 border border-violet-500/30 text-violet-200 px-2 py-0.5 text-[10px] font-mono"
                          >{{ d.pii_type }}</span
                        >
                      }
                    </div>
                  </div>
                }
              }
            }
            @case ('audit') {
              @if (loadingAudit()) {
                <div class="text-zinc-500 text-xs">Loading…</div>
              } @else if (!audit().length) {
                <div class="text-zinc-500 text-xs">No audit events.</div>
              }
              @for (e of audit(); track e.event_id) {
                <div
                  class="text-xs flex flex-wrap gap-2 font-mono items-baseline border-b border-zinc-900 pb-1"
                >
                  <span class="text-zinc-500">{{ time(e.timestamp) }}</span>
                  <span class="text-violet-300">{{ e.node_name }}</span>
                  <span class="text-zinc-300">→ {{ e.event_type }}</span>
                </div>
              }
            }
          }
        </div>
      </aside>
    }
  `,
})
export class InspectorComponent implements OnChanges {
  private api = inject(ApiService);
  private store = inject(SessionStore);
  private destroyRef = inject(DestroyRef);
  private auditSub: Subscription | null = null;

  @Input() msg: ChatMessage | null = null;
  @Output() close = new EventEmitter<void>();

  readonly active = signal<Tab>('sources');
  readonly audit = signal<AuditEvent[]>([]);
  readonly loadingAudit = signal(false);

  readonly tabs = [
    { key: 'sources' as Tab, label: 'Sources', count: 0 },
    { key: 'verification' as Tab, label: 'Verify', count: 0 },
    { key: 'guardrails' as Tab, label: 'PII', count: 0 },
    { key: 'audit' as Tab, label: 'Audit', count: 0 },
  ];

  ngOnChanges(ch: SimpleChanges): void {
    if (ch['msg']) {
      this.tabs[0].count = this.msg?.sources?.length ?? 0;
      this.tabs[1].count = this.msg?.verification
        ? this.msg.verification.supported
          ? 0
          : (this.msg.verification.unsupported_claims.length || 0) +
            (this.msg.verification.contradictions.length || 0)
        : 0;
      const g = this.msg?.guardrails;
      this.tabs[2].count = g
        ? (g.input_redactions || 0) + (g.output_redactions || 0)
        : 0;
      this.audit.set([]);
      this.tabs[3].count = 0;
      if (this.msg && this.active() === 'audit') this.loadAudit();
    }
  }

  select(tab: Tab): void {
    this.active.set(tab);
    if (tab === 'audit' && !this.audit().length && !this.loadingAudit()) {
      this.loadAudit();
    }
  }

  snippet(): string {
    const t = (this.msg?.content || '').replace(/\s+/g, ' ').trim();
    return t.slice(0, 80) + (t.length > 80 ? '…' : '');
  }

  time(ts: string): string {
    if (!ts) return '';
    const t = ts.includes('T') ? ts.split('T')[1] : ts;
    return t.slice(0, 8);
  }

  loadAudit(): void {
    const sid = this.store.activeId();
    if (!sid) return;
    // Cancel any in-flight audit request to prevent overlapping responses
    // from racing each other and to free the connection.
    if (this.auditSub) {
      this.auditSub.unsubscribe();
      this.auditSub = null;
    }
    this.loadingAudit.set(true);
    this.auditSub = this.api
      .audit(sid)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (events) => {
          this.audit.set(events);
          this.tabs[3].count = events.length;
          this.loadingAudit.set(false);
        },
        error: (e) => {
          this.loadingAudit.set(false);
          console.error('inspector: audit load failed', e);
        },
      });
  }
}
