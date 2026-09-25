// src/api/civicSync.ts
// Centralized API client — the ONLY place fetch calls are made.
// All components use this module. Never scatter fetch() across the app.

import type { CivicSyncResponse, QueryRequest } from '../types/civicSync';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

export class ApiError extends Error {
  statusCode: number;
  constructor(
    statusCode: number,
    message: string,
  ) {
    super(message);
    this.statusCode = statusCode;
    this.name = 'ApiError';
  }
}

/**
 * POST /api/query
 * Submits the citizen's situation to the CivicSync backend.
 * Returns a structured guidance response.
 */
export async function queryCivicSync(
  message: string,
  language: string = 'auto',
  conversationId?: string,
  signal?: AbortSignal,
): Promise<CivicSyncResponse> {
  const payload: QueryRequest = { 
    message: message.trim(), 
    language,
    conversation_id: conversationId,
  };

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError(0, 'Could not reach the CivicSync service. Please check your connection.');
  }

  if (!response.ok) {
    let detail = `Server returned ${response.status}.`;
    try {
      const body = await response.json();
      if (body?.message) detail = body.message;
    } catch {
      // ignore
    }
    throw new ApiError(response.status, detail);
  }

  const data: CivicSyncResponse = await response.json();
  return data;
}

/**
 * GET /api/health
 * Used for connectivity checks only.
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}
