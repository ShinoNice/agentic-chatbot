import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { ApiService } from './api.service';
import { ChatResponse, StreamEvent } from '../models/chat.model';

@Injectable({ providedIn: 'root' })
export class SseService {
  private api = inject(ApiService);

  streamChat(question: string, sessionId: string): Observable<StreamEvent> {
    const url = `${this.api.baseUrl()}/api/chat/stream`;
    return new Observable<StreamEvent>((sub) => {
      const ac = new AbortController();

      (async () => {
        try {
          const res = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Accept: 'text/event-stream',
            },
            body: JSON.stringify({ question, session_id: sessionId }),
            signal: ac.signal,
          });

          if (!res.ok || !res.body) {
            sub.next({ type: 'error', detail: `HTTP ${res.status}` });
            sub.complete();
            return;
          }

          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buf = '';

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            let idx: number;
            while ((idx = buf.indexOf('\n\n')) !== -1) {
              const raw = buf.slice(0, idx);
              buf = buf.slice(idx + 2);
              const evt = parseFrame(raw);
              if (!evt) continue;
              sub.next(toStreamEvent(evt));
            }
          }
          sub.complete();
        } catch (err) {
          if ((err as Error).name !== 'AbortError') {
            sub.next({ type: 'error', detail: (err as Error).message });
          }
          sub.complete();
        }
      })();

      return () => ac.abort();
    });
  }
}

function parseFrame(raw: string): { event: string; data: string } | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  return { event, data: dataLines.join('\n') };
}

function toStreamEvent(frame: { event: string; data: string }): StreamEvent {
  try {
    const data = frame.data ? JSON.parse(frame.data) : {};
    if (frame.event === 'step') return { type: 'step', ...data };
    if (frame.event === 'result')
      return { type: 'result', payload: data as ChatResponse };
    if (frame.event === 'done') return { type: 'done' };
    if (frame.event === 'error')
      return { type: 'error', detail: data.detail || 'Unknown error' };
  } catch {
    /* fallthrough */
  }
  return { type: 'error', detail: `Bad frame: ${frame.event}` };
}
