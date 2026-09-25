// src/types/civicSync.ts
// CivicSync API type definitions — mirrors backend/schemas.py

export interface EvidenceSource {
  chunk_id?: number | string | null;
  domain?: string | null;
  source_file?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  similarity?: number | null;
}

export interface SituationInfo {
  domain?: string | null;
  issue?: string | null;
  intent?: string | null;
  confidence?: number | null;
  entities?: string[] | null;
  situation?: string | null;
}

export interface EvidenceItem {
  chunk_id?: number | string | null;
  source_file?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  similarity?: number | null;
}

export interface QualificationItem {
  text: string;
  evidence?: EvidenceItem[];
}

export interface QualificationInfo {
  status?: string | null;
  domain?: string | null;
  issue?: string | null;
  applicable_information?: QualificationItem[] | null;
  rights_or_protections?: QualificationItem[] | null;
  possible_actions?: QualificationItem[] | null;
  authorities_or_channels?: QualificationItem[] | null;
  documents_or_evidence?: QualificationItem[] | null;
  conditions?: QualificationItem[] | null;
  missing_information?: string[] | null;
}

export interface ActionInfo {
  immediate_steps?: QualificationItem[] | null;
  formal_remedies?: QualificationItem[] | null;
  authority_pathway?: QualificationItem[] | null;
  escalation?: QualificationItem[] | null;
  information_to_collect?: string[] | null;
}

export type QueryStatus = 'success' | 'clarification_required' | 'error';

export interface CivicSyncResponse {
  status: QueryStatus;
  situation?: SituationInfo | null;
  qualification?: QualificationInfo | null;
  actions?: ActionInfo | null;
  answer?: string | null;
  sources?: EvidenceSource[];
  limitations?: string[];
  message?: string; // clarification or error message
  conversation_id?: string;
}

export interface QueryRequest {
  message: string;
  language: string;
  conversation_id?: string;
}

export type Language = 'auto' | 'en' | 'hi' | 'gu';

export interface Category {
  id: string;
  label: string;
  icon: string;
  color: string;
  suggestions: string[];
}
