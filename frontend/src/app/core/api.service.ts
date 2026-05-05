import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, of } from 'rxjs';
import {
  AuditEvent,
  ChatResponse,
  HealthResponse,
  UploadResponse,
} from '../models/chat.model';

const DEFAULT_BASE = 'http://localhost:8001';
const STORAGE_KEY = 'agentic-api-base-url';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  readonly baseUrl = signal<string>(
    localStorage.getItem(STORAGE_KEY) || DEFAULT_BASE,
  );

  setBaseUrl(url: string): void {
    const cleaned = (url || '').replace(/\/$/, '');
    this.baseUrl.set(cleaned);
    localStorage.setItem(STORAGE_KEY, cleaned);
  }

  private url(path: string): string {
    return `${this.baseUrl()}/api${path}`;
  }

  chat(question: string, sessionId: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(this.url('/chat'), {
      question,
      session_id: sessionId,
    });
  }

  upload(file: File, sessionId: string): Observable<UploadResponse> {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('session_id', sessionId);
    return this.http.post<UploadResponse>(this.url('/upload'), fd);
  }

  health(): Observable<HealthResponse | null> {
    return this.http
      .get<HealthResponse>(this.url('/health'))
      .pipe(catchError(() => of(null)));
  }

  audit(sessionId: string): Observable<AuditEvent[]> {
    return this.http
      .get<AuditEvent[]>(this.url(`/audit/${sessionId}`))
      .pipe(catchError(() => of([])));
  }
}
