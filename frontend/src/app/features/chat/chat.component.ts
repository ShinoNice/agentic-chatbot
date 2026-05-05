import {
  Component,
  ElementRef,
  ViewChild,
  AfterViewChecked,
  EventEmitter,
  Output,
  inject,
  signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { SessionStore } from '../../core/session-store.service';
import { SseService } from '../../core/sse.service';
import { MessageComponent } from './message.component';
import { ComposerComponent } from './composer.component';
import { PipelineStepperComponent } from './pipeline-stepper.component';
import { ChatMessage } from '../../models/chat.model';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    MessageComponent,
    ComposerComponent,
    PipelineStepperComponent,
  ],
  template: `
    <div class="relative flex h-full flex-col">
      <header
        class="flex items-center justify-between border-b border-zinc-800 bg-zinc-950/60 backdrop-blur px-5 py-3 shrink-0"
      >
        <div class="min-w-0">
          <div class="text-sm text-zinc-200 truncate">
            {{ store.active()?.title || 'New chat' }}
          </div>
          <div
            class="text-[10px] font-mono uppercase tracking-wider text-zinc-500 truncate"
          >
            @if (store.hasUpload()) {
              📄 {{ store.uploadedFilename() }} ({{
                store.uploadedChunks()
              }}
              chunks)
            } @else {
              📚 default corpus
            }
          </div>
        </div>
      </header>

      <div #scroll class="flex-1 overflow-y-auto px-5 py-6 space-y-5 min-h-0">
        @if (!store.messages().length) {
          <div class="grid h-full place-items-center text-center">
            <div class="space-y-3 max-w-md">
              <div
                class="mx-auto h-14 w-14 rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 grid place-items-center text-white text-2xl font-bold pulse-ring"
              >
                ⌬
              </div>
              <h2 class="text-lg font-semibold text-zinc-100">
                Ask the agent
              </h2>
              <p class="text-sm text-zinc-400 text-pretty leading-relaxed">
                It will retrieve relevant documents, draft an answer, and
                verify it. Watch the pipeline run live above the composer.
              </p>
              <div
                class="flex flex-wrap justify-center gap-1.5 pt-2 text-[10px] font-mono uppercase tracking-wider text-zinc-500"
              >
                <span class="rounded-full border border-zinc-800 px-2 py-0.5"
                  >Hybrid retrieval</span
                >
                <span class="rounded-full border border-zinc-800 px-2 py-0.5"
                  >PII guardrails</span
                >
                <span class="rounded-full border border-zinc-800 px-2 py-0.5"
                  >Self-verification</span
                >
                <span class="rounded-full border border-zinc-800 px-2 py-0.5"
                  >Audit trail</span
                >
              </div>
            </div>
          </div>
        }
        @for (msg of store.messages(); track msg.id) {
          <app-message
            [msg]="msg"
            (inspectorRequested)="inspect.emit($event)"
          />
        }
      </div>

      <div
        class="px-5 pt-2 pb-4 space-y-2 bg-gradient-to-t from-zinc-950 via-zinc-950/95 to-transparent shrink-0"
      >
        <div class="flex justify-center min-h-[1.5rem]">
          <app-pipeline-stepper />
        </div>
        <app-composer [disabled]="sending()" (send)="send($event)" />
      </div>
    </div>
  `,
})
export class ChatComponent implements AfterViewChecked {
  protected store = inject(SessionStore);
  private sse = inject(SseService);

  @ViewChild('scroll') scrollEl!: ElementRef<HTMLDivElement>;
  @Output() inspect = new EventEmitter<ChatMessage>();

  sending = signal(false);

  ngAfterViewChecked(): void {
    const el = this.scrollEl?.nativeElement;
    if (el) el.scrollTop = el.scrollHeight;
  }

  send(text: string): void {
    if (this.sending()) return;
    const sessionId = this.store.activeId() || this.store.newSession();
    this.sending.set(true);

    this.store.addMessage({
      id: this.uid(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    });
    this.store.addMessage({
      id: this.uid(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isLoading: true,
    });
    this.store.setStreamingStep({ node: 'guardrails_input' });

    this.sse.streamChat(text, sessionId).subscribe({
      next: (ev) => {
        if (ev.type === 'step') {
          this.store.setStreamingStep({
            node: ev.node,
            iteration: ev.iteration,
          });
        } else if (ev.type === 'result') {
          this.store.updateLastMessage({
            content: ev.payload.answer,
            sources: ev.payload.sources,
            verification: ev.payload.verification,
            guardrails: ev.payload.guardrails,
            relevance_status: ev.payload.relevance_status,
            iterations: ev.payload.iterations,
            isLoading: false,
          });
        } else if (ev.type === 'error') {
          this.store.updateLastMessage({
            content: `⚠️ ${ev.detail}`,
            isLoading: false,
            isError: true,
          });
        }
      },
      complete: () => {
        this.sending.set(false);
        this.store.setStreamingStep(null);
      },
      error: (err: Error) => {
        this.store.updateLastMessage({
          content: `⚠️ ${err.message}`,
          isLoading: false,
          isError: true,
        });
        this.sending.set(false);
        this.store.setStreamingStep(null);
      },
    });
  }

  private uid(): string {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID();
    }
    return 'm-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
}
