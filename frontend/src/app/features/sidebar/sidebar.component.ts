import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { SessionStore } from '../../core/session-store.service';
import { ThemeService } from '../../core/theme.service';

type HealthStatus = 'unknown' | 'ok' | 'partial' | 'down';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <aside
      class="flex h-full flex-col border-r border-zinc-800 bg-zinc-950/80 backdrop-blur transition-all duration-200 shrink-0"
      [class.w-64]="!collapsed()"
      [class.w-14]="collapsed()"
    >
      <div
        class="flex items-center justify-between px-3 py-3 border-b border-zinc-800"
      >
        @if (!collapsed()) {
          <div class="flex items-center gap-2 min-w-0">
            <div
              class="h-7 w-7 shrink-0 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 grid place-items-center text-white font-bold text-xs"
            >
              ⌬
            </div>
            <div class="text-sm font-semibold text-zinc-100 truncate">
              Agentic RAG
            </div>
          </div>
        }
        <button
          (click)="collapsed.set(!collapsed())"
          class="text-zinc-500 hover:text-zinc-200 text-sm font-mono px-1 leading-none"
          [attr.aria-label]="collapsed() ? 'Expand sidebar' : 'Collapse sidebar'"
        >
          {{ collapsed() ? '›' : '‹' }}
        </button>
      </div>

      <button
        (click)="store.newSession()"
        class="m-3 rounded-xl border border-zinc-800 hover:border-violet-500/50 bg-zinc-900/50 hover:bg-violet-500/10 px-3 py-2 text-xs text-zinc-200 hover:text-violet-200 font-mono uppercase tracking-wider transition-colors flex items-center gap-2 shrink-0"
        [class.justify-center]="collapsed()"
      >
        <span class="text-base leading-none">＋</span>
        @if (!collapsed()) {
          <span>New chat</span>
        }
      </button>

      @if (!collapsed()) {
        <div class="flex-1 overflow-y-auto px-2 space-y-0.5 min-h-0">
          @for (s of store.sessions(); track s.id) {
            <div
              class="group flex items-center gap-1 rounded-lg px-2 py-1.5 cursor-pointer text-xs transition-colors"
              [class.text-violet-200]="s.id === store.activeId()"
              [class.text-zinc-400]="s.id !== store.activeId()"
              [style.background]="
                s.id === store.activeId() ? 'rgba(139,92,246,0.12)' : ''
              "
              (click)="store.selectSession(s.id)"
            >
              <span class="flex-1 truncate">{{ s.title }}</span>
              <button
                class="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-rose-400 transition-opacity text-sm leading-none"
                (click)="$event.stopPropagation(); store.deleteSession(s.id)"
                aria-label="Delete chat"
              >
                ×
              </button>
            </div>
          }
        </div>

        <div class="border-t border-zinc-800 p-3 space-y-3 text-xs shrink-0">
          <div>
            <label
              class="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 mb-1"
            >
              Upload PDF
            </label>
            <label
              class="block rounded-lg border border-dashed border-zinc-700 hover:border-violet-500/60 px-3 py-3 text-center text-zinc-400 hover:text-violet-300 cursor-pointer transition-colors"
              (drop)="onDrop($event)"
              (dragover)="$event.preventDefault()"
            >
              @if (uploading()) {
                <span class="text-violet-300">Uploading…</span>
              } @else if (store.uploadedFilename()) {
                <span class="text-violet-300 font-mono text-[10px] block truncate">
                  {{ store.uploadedFilename() }}
                </span>
              } @else {
                <span class="text-[11px]">Drop a PDF or click</span>
              }
              <input
                type="file"
                accept="application/pdf"
                class="hidden"
                #fi
                (change)="onPicked(fi.files); fi.value = ''"
              />
            </label>
            @if (store.uploadedFilename()) {
              <button
                class="mt-1 text-[10px] text-zinc-500 hover:text-zinc-300 font-mono uppercase tracking-wider"
                (click)="store.clearUpload()"
              >
                ↩ default corpus
              </button>
            }
            @if (uploadError()) {
              <div class="mt-1 text-[10px] text-rose-400">
                {{ uploadError() }}
              </div>
            }
          </div>

          <div>
            <label
              class="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 mb-1"
            >
              API URL
            </label>
            <input
              [ngModel]="api.baseUrl()"
              (ngModelChange)="api.setBaseUrl($event); checkHealth()"
              class="w-full rounded-md bg-zinc-900 border border-zinc-800 px-2 py-1 text-[11px] font-mono text-zinc-300 outline-none focus:border-violet-500/60"
            />
          </div>

          <div class="flex items-center justify-between">
            <button
              (click)="theme.toggle()"
              class="text-[10px] font-mono uppercase tracking-wider text-zinc-400 hover:text-violet-300"
            >
              {{ theme.theme() === 'dark' ? '☾ dark' : '☀ light' }}
            </button>
            <div class="flex items-center gap-1.5 text-[10px] font-mono">
              <span
                class="h-1.5 w-1.5 rounded-full"
                [class.bg-emerald-500]="health() === 'ok'"
                [class.bg-amber-500]="health() === 'partial'"
                [class.bg-rose-500]="health() === 'down'"
                [class.bg-zinc-600]="health() === 'unknown'"
              ></span>
              <span class="text-zinc-500">{{ healthLabel() }}</span>
              <button
                class="text-zinc-500 hover:text-zinc-300"
                (click)="checkHealth()"
                aria-label="Recheck health"
              >
                ↻
              </button>
            </div>
          </div>
        </div>
      }
    </aside>
  `,
})
export class SidebarComponent {
  protected store = inject(SessionStore);
  protected api = inject(ApiService);
  protected theme = inject(ThemeService);
  private destroyRef = inject(DestroyRef);
  private healthSub: Subscription | null = null;
  private uploadSub: Subscription | null = null;

  collapsed = signal(false);
  uploading = signal(false);
  uploadError = signal<string | null>(null);
  health = signal<HealthStatus>('unknown');

  constructor() {
    this.checkHealth();
  }

  healthLabel(): string {
    return ({ unknown: '—', ok: 'ready', partial: 'kb empty', down: 'offline' } as const)[
      this.health()
    ];
  }

  checkHealth(): void {
    // Cancel any in-flight health check before starting a new one.
    if (this.healthSub) {
      this.healthSub.unsubscribe();
      this.healthSub = null;
    }
    this.healthSub = this.api
      .health()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((res) => {
        if (!res) this.health.set('down');
        else if (res.knowledge_base_ready) this.health.set('ok');
        else this.health.set('partial');
      });
  }

  onPicked(files: FileList | null): void {
    if (!files?.length) return;
    this.upload(files[0]);
  }

  onDrop(ev: DragEvent): void {
    ev.preventDefault();
    const f = ev.dataTransfer?.files?.[0];
    if (f) this.upload(f);
  }

  private upload(file: File): void {
    const isPdf =
      file.type === 'application/pdf' ||
      file.name.toLowerCase().endsWith('.pdf');
    if (!isPdf) {
      this.uploadError.set('PDF only');
      return;
    }
    this.uploadError.set(null);
    this.uploading.set(true);
    const sid = this.store.activeId() || this.store.newSession();
    if (this.uploadSub) {
      this.uploadSub.unsubscribe();
    }
    this.uploadSub = this.api
      .upload(file, sid)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.store.setUpload(res.filename, res.total_chunks);
          this.uploading.set(false);
        },
        error: (err) => {
          this.uploadError.set(err?.error?.detail || 'Upload failed');
          this.uploading.set(false);
        },
      });
  }
}
