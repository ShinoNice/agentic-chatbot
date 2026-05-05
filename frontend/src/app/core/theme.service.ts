import { Injectable, signal, effect } from '@angular/core';

type Theme = 'dark' | 'light';
const KEY = 'agentic-theme';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly theme = signal<Theme>(
    (localStorage.getItem(KEY) as Theme | null) || 'dark',
  );

  constructor() {
    effect(() => {
      const t = this.theme();
      document.documentElement.classList.toggle('light', t === 'light');
      localStorage.setItem(KEY, t);
    });
  }

  toggle(): void {
    this.theme.update((t) => (t === 'dark' ? 'light' : 'dark'));
  }
}
