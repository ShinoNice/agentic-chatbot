import { Injectable, computed, signal, effect } from '@angular/core';
import { ChatMessage, ChatSession } from '../models/chat.model';

const STORAGE_KEY = 'agentic-chat-sessions-v1';
const MAX_SESSIONS = 50;

interface PersistedState {
  sessions: ChatSession[];
  activeId: string | null;
}

@Injectable({ providedIn: 'root' })
export class SessionStore {
  private _sessions = signal<ChatSession[]>([]);
  private _activeId = signal<string | null>(null);
  private _streamingStep = signal<{ node: string; iteration?: number } | null>(
    null,
  );

  readonly sessions = this._sessions.asReadonly();
  readonly activeId = this._activeId.asReadonly();
  readonly streamingStep = this._streamingStep.asReadonly();

  readonly active = computed<ChatSession | null>(() => {
    const id = this._activeId();
    return this._sessions().find((s) => s.id === id) || null;
  });

  readonly messages = computed<ChatMessage[]>(
    () => this.active()?.messages ?? [],
  );
  readonly hasUpload = computed<boolean>(() => !!this.active()?.uploadedFilename);
  readonly uploadedFilename = computed<string | null>(
    () => this.active()?.uploadedFilename ?? null,
  );
  readonly uploadedChunks = computed<number>(
    () => this.active()?.uploadedChunks ?? 0,
  );

  constructor() {
    this.hydrate();
    effect(() => {
      this._sessions();
      this._activeId();
      this.persist();
    });
  }

  newSession(): string {
    const id = this.uid();
    const session: ChatSession = {
      id,
      title: 'New chat',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
      uploadedFilename: null,
      uploadedChunks: 0,
    };
    this._sessions.update((arr) => [session, ...arr].slice(0, MAX_SESSIONS));
    this._activeId.set(id);
    return id;
  }

  selectSession(id: string): void {
    this._activeId.set(id);
  }

  deleteSession(id: string): void {
    this._sessions.update((arr) => arr.filter((s) => s.id !== id));
    if (this._activeId() === id) {
      const next = this._sessions()[0];
      if (next) this._activeId.set(next.id);
      else this.newSession();
    }
  }

  addMessage(msg: ChatMessage): void {
    this.mutateActive((s) => {
      s.messages = [...s.messages, msg];
      if (s.title === 'New chat' && msg.role === 'user') {
        s.title = msg.content.slice(0, 60);
      }
      s.updatedAt = Date.now();
    });
  }

  updateLastMessage(patch: Partial<ChatMessage>): void {
    this.mutateActive((s) => {
      const last = s.messages.at(-1);
      if (!last) return;
      s.messages = [...s.messages.slice(0, -1), { ...last, ...patch }];
      s.updatedAt = Date.now();
    });
  }

  setUpload(filename: string, chunks: number): void {
    this.mutateActive((s) => {
      s.uploadedFilename = filename;
      s.uploadedChunks = chunks;
      s.messages = [];
      s.updatedAt = Date.now();
    });
  }

  clearUpload(): void {
    this.mutateActive((s) => {
      s.uploadedFilename = null;
      s.uploadedChunks = 0;
    });
  }

  setStreamingStep(step: { node: string; iteration?: number } | null): void {
    this._streamingStep.set(step);
  }

  private mutateActive(mutator: (s: ChatSession) => void): void {
    const id = this._activeId();
    if (!id) return;
    this._sessions.update((arr) =>
      arr.map((s) => {
        if (s.id !== id) return s;
        const copy: ChatSession = { ...s, messages: [...s.messages] };
        mutator(copy);
        return copy;
      }),
    );
  }

  private uid(): string {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID();
    }
    return 'sess-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  }

  private hydrate(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        this.newSession();
        return;
      }
      const parsed = JSON.parse(raw) as PersistedState;
      this._sessions.set(parsed.sessions || []);
      this._activeId.set(parsed.activeId);
      if (!this._sessions().length) this.newSession();
      else if (!this._sessions().find((s) => s.id === this._activeId())) {
        this._activeId.set(this._sessions()[0].id);
      }
    } catch {
      this.newSession();
    }
  }

  private persist(): void {
    const state: PersistedState = {
      sessions: this._sessions(),
      activeId: this._activeId(),
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      /* quota exceeded; ignore */
    }
  }
}
