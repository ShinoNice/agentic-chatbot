import { Component, Input, Output, EventEmitter, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatMessage } from '../../models/chat.model';
import { MarkdownService } from '../../core/markdown.service';
import { RibbonComponent } from './ribbon.component';

@Component({
  selector: 'app-message',
  standalone: true,
  imports: [CommonModule, RibbonComponent],
  template: `
    <div
      class="animate-msg-in flex gap-3"
      [class.flex-row-reverse]="msg.role === 'user'"
    >
      <div class="shrink-0">
        @if (msg.role === 'assistant') {
          <div
            class="h-8 w-8 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 grid place-items-center text-white text-xs font-semibold shadow-lg shadow-violet-500/20"
            [class.pulse-ring]="msg.isLoading"
          >
            ⌬
          </div>
        } @else {
          <div
            class="h-8 w-8 rounded-full bg-zinc-800 border border-zinc-700 grid place-items-center text-zinc-300 text-xs font-semibold"
          >
            U
          </div>
        }
      </div>

      <div class="min-w-0 flex-1 max-w-3xl">
        @if (msg.role === 'assistant') {
          @if (msg.isLoading) {
            <div
              class="rounded-2xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-sm text-zinc-400 inline-flex items-center gap-1"
            >
              <span class="thinking-dot h-1.5 w-1.5 rounded-full bg-violet-400"></span>
              <span class="thinking-dot h-1.5 w-1.5 rounded-full bg-violet-400"></span>
              <span class="thinking-dot h-1.5 w-1.5 rounded-full bg-violet-400"></span>
            </div>
          } @else {
            <div class="space-y-2">
              <app-ribbon
                [verification]="msg.verification ?? null"
                [guardrails]="msg.guardrails ?? null"
                [relevanceStatus]="msg.relevance_status"
              />
              <div
                class="md-body rounded-2xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-zinc-100"
                [style.borderColor]="msg.isError ? 'rgba(244,63,94,0.4)' : ''"
                [innerHTML]="rendered"
              ></div>
              <div
                class="flex items-center gap-3 text-[10px] text-zinc-500 font-mono"
              >
                @if (msg.iterations && msg.iterations > 1) {
                  <span>↻ {{ msg.iterations }} iterations</span>
                }
                @if (msg.sources?.length) {
                  <button
                    class="hover:text-violet-300 transition-colors uppercase tracking-wider"
                    (click)="inspectorRequested.emit(msg)"
                  >
                    {{ msg.sources!.length }} source{{
                      msg.sources!.length === 1 ? '' : 's'
                    }}
                    →
                  </button>
                } @else {
                  <button
                    class="hover:text-violet-300 transition-colors uppercase tracking-wider"
                    (click)="inspectorRequested.emit(msg)"
                  >
                    inspect →
                  </button>
                }
              </div>
            </div>
          }
        } @else {
          <div class="flex justify-end">
            <div
              class="inline-block rounded-2xl bg-violet-500/15 border border-violet-500/30 px-4 py-2 text-sm text-zinc-100 whitespace-pre-wrap text-left"
            >
              {{ msg.content }}
            </div>
          </div>
        }
      </div>
    </div>
  `,
})
export class MessageComponent {
  private md = inject(MarkdownService);

  @Input({ required: true }) msg!: ChatMessage;
  @Output() inspectorRequested = new EventEmitter<ChatMessage>();

  get rendered(): string {
    return this.md.render(this.msg.content || '', this.msg.sources?.length ?? 0);
  }
}
