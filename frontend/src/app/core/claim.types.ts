export type ClaimStatus = 'pending' | 'verified' | 'unsupported' | 'contradicted';

export interface Citation {
  chunk_id: string;
  quote: string;
}

export interface Claim {
  id: string;
  text: string;
  citations: Citation[];
  status: ClaimStatus;
  verifier_note: string | null;
  attempts: number;
}
