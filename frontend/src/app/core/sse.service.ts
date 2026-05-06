import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { ApiService } from './api.service';
import { ChatResponse, StreamEvent } from '../models/chat.model';
import type { Claim, ClaimStatus } from './claim.types';

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
            let bodyText = '';
            try {
              bodyText = await res.text();
            } catch {
              /* ignore body read errors */
            }
            sub.error({
              status: res.status,
              detail: bodyText || `HTTP ${res.status}`,
            });
            return;
          }

          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buf = '';

          const splitFrame = (b: string): [string, string] | null => {
            const candidates = ['\r\n\r\n', '\n\n', '\r\r'];
            let bestIdx = -1;
            let bestLen = 0;
            for (const sep of candidates) {
              const i = b.indexOf(sep);
              if (i !== -1 && (bestIdx === -1 || i < bestIdx)) {
                bestIdx = i;
                bestLen = sep.length;
              }
            }
            if (bestIdx === -1) return null;
            return [b.slice(0, bestIdx), b.slice(bestIdx + bestLen)];
          };

          const drainBuffer = () => {
            let split = splitFrame(buf);
            while (split) {
              const [raw, rest] = split;
              buf = rest;
              const evt = parseFrame(raw);
              if (evt) sub.next(toStreamEvent(evt));
              split = splitFrame(buf);
            }
          };

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            drainBuffer();
          }
          // Flush trailing decoder bytes and process any remaining buffer.
          buf += decoder.decode();
          drainBuffer();
          if (buf.trim().length) {
            const evt = parseFrame(buf);
            if (evt) sub.next(toStreamEvent(evt));
            buf = '';
          }
          sub.complete();
        } catch (err) {
          if ((err as Error).name !== 'AbortError') {
            sub.error(err);
          } else {
            sub.complete();
          }
        }
      })().catch((err) => {
        // Belt-and-suspenders: anything that escapes the inner try.
        sub.error(err);
      });

      return () => ac.abort();
    });
  }
}

function parseFrame(raw: string): { event: string; data: string } | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of raw.split(/\r\n|\n|\r/)) {
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
    if (frame.event === 'claim_drafted')
      return { type: 'claim_drafted', claim: data.claim as Claim };
    if (frame.event === 'claim_verified')
      return {
        type: 'claim_verified',
        claim_id: data.claim_id as string,
        status: data.status as ClaimStatus,
        note: (data.note ?? null) as string | null,
      };
    if (frame.event === 'claim_repaired')
      return {
        type: 'claim_repaired',
        claim_id: data.claim_id as string,
        status: data.status as ClaimStatus,
      };
  } catch (err) {
    console.warn('SSE: bad frame', err, frame);
  }
  return { type: 'error', detail: `Bad frame: ${frame.event}` };
}
