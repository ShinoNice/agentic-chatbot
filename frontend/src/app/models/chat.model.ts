import type { Claim, ClaimStatus } from '../core/claim.types';

export interface SourceDocument {
  source: string;
  page_number: number | null;
  snippet: string;
}

export interface VerificationDetail {
  supported: boolean;
  unsupported_claims: string[];
  contradictions: string[];
  relevant: boolean;
  additional_details: string | null;
}

export interface PiiDetection {
  pii_type: string;
  [k: string]: unknown;
}

export interface GuardrailsReport {
  input_pii_found: boolean;
  output_pii_found: boolean;
  input_redactions: number;
  output_redactions: number;
  detections: PiiDetection[];
}

export interface ChatResponse {
  answer: string;
  session_id: string | null;
  relevance_status: string;
  verification: VerificationDetail | null;
  sources: SourceDocument[];
  iterations: number;
  guardrails: GuardrailsReport | null;
  claims?: Claim[];
  truncated?: boolean;
}

export interface UploadResponse {
  filename: string;
  total_chunks: number;
  namespace: string;
  session_id: string;
}

export interface HealthResponse {
  status: string;
  knowledge_base_ready: boolean;
  vector_store_type: string;
  documents_indexed: number | null;
}

export interface AuditEvent {
  event_id: string;
  session_id: string;
  timestamp: string;
  event_type: string;
  node_name: string;
  details: Record<string, unknown>;
}

export type AgentNode =
  | 'guardrails_input'
  | 'retrieve'
  | 'rerank'
  | 'check_relevance'
  | 'research'
  | 'guardrails_output'
  | 'verify';

export type StreamEvent =
  | {
      type: 'step';
      node: AgentNode | string;
      status: 'started' | 'completed';
      iteration?: number;
      doc_count?: number;
      supported?: boolean;
    }
  | { type: 'result'; payload: ChatResponse }
  | { type: 'done' }
  | { type: 'error'; detail: string }
  | { type: 'claim_drafted'; claim: Claim }
  | {
      type: 'claim_verified';
      claim_id: string;
      status: ClaimStatus;
      note: string | null;
    }
  | { type: 'claim_repaired'; claim_id: string; status: ClaimStatus };

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  sources?: SourceDocument[];
  verification?: VerificationDetail | null;
  guardrails?: GuardrailsReport | null;
  relevance_status?: string;
  iterations?: number;
  isLoading?: boolean;
  isError?: boolean;
  claims?: Claim[];
  truncated?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  uploadedFilename: string | null;
  uploadedChunks: number;
}
