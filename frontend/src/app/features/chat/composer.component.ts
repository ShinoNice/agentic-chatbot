import {
  Component,
  ElementRef,
  EventEmitter,
  Input,
  Output,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-composer',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <form
      (ngSubmit)="submit()"
      class="flex items-end gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-2 backdrop-blur focus-within:border-violet-500/60 transition-colors shadow-xl"
    >
      <textarea
        #ta
        [(ngModel)]="text"
        name="q"
        rows="1"
        maxlength="4000"
        [disabled]="disabled"
        (input)="autosize()"
        (keydown)="onKey($event)"
        placeholder="Ask about your documents… (Shift+Enter for newline)"
        class="flex-1 resize-none bg-transparent px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 outline-none max-h-48 leading-relaxed font-sans"
      ></textarea>
      <button
        type="submit"
        [disabled]="disabled || !text.trim()"
        class="shrink-0 grid place-items-center h-9 w-9 rounded-xl bg-violet-500 text-white shadow-lg shadow-violet-500/30 hover:bg-violet-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        aria-label="Send"
      >
        @if (disabled) {
          <span
            class="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white"
          ></span>
        } @else {
          <svg
            viewBox="0 0 24 24"
            class="h-4 w-4"
            fill="none"
            stroke="currentColor"
            stroke-width="2.4"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        }
      </button>
    </form>
    @if (text.length > 600) {
      <div class="mt-1 text-right text-[10px] text-zinc-500 font-mono">
        {{ text.length }} chars
      </div>
    }
  `,
})
export class ComposerComponent {
  @Input() disabled = false;
  @Output() send = new EventEmitter<string>();
  @ViewChild('ta', { static: true }) ta!: ElementRef<HTMLTextAreaElement>;
  text = '';

  submit(): void {
    const t = this.text.trim();
    if (!t || this.disabled) return;
    this.send.emit(t);
    this.text = '';
    queueMicrotask(() => this.autosize());
  }

  onKey(ev: KeyboardEvent): void {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      this.submit();
    }
  }

  autosize(): void {
    const el = this.ta.nativeElement;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 192) + 'px';
  }
}
