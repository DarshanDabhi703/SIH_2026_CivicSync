// src/pages/Guidance.tsx
import { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { queryCivicSync, ApiError } from '../api/civicSync';
import type { CivicSyncResponse, Language } from '../types/civicSync';
import QueryInput from '../components/QueryInput';
import LoadingState from '../components/LoadingState';
import GuidanceResult from '../components/GuidanceResult';
import Disclaimer from '../components/Disclaimer';
import './Guidance.css';

type UIState = 'idle' | 'loading' | 'success' | 'clarification' | 'error';

export default function Guidance() {
  const location  = useLocation();
  const navigate  = useNavigate();
  const abortRef  = useRef<AbortController | null>(null);

  const [uiState, setUiState]   = useState<UIState>('idle');
  const [result,  setResult]    = useState<CivicSyncResponse | null>(null);
  const [query,   setQuery]     = useState<string>('');
  const [errorMsg, setErrorMsg] = useState('');

  const runQuery = useCallback(async (message: string, language: Language) => {
    // Cancel any in-flight request
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setQuery(message);
    setUiState('loading');
    setResult(null);
    setErrorMsg('');

    try {
      const data = await queryCivicSync(message, language, undefined, ctrl.signal);

      if (data.status === 'clarification_required') {
        setResult(data);
        setUiState('clarification');
      } else if (data.status === 'error') {
        setErrorMsg(data.message ?? 'Something went wrong. Please try again.');
        setUiState('error');
      } else {
        setResult(data);
        setUiState('success');
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      const msg = err instanceof ApiError
        ? err.message
        : 'Something went wrong while retrieving your guidance. Please try again.';
      setErrorMsg(msg);
      setUiState('error');
    }
  }, []);

  // Auto-run if navigated here with a message in state
  useEffect(() => {
    const state = location.state as { message?: string; language?: Language } | null;
    if (state?.message) {
      runQuery(state.message, state.language ?? 'auto');
      // Clear state so refresh doesn't re-run
      window.history.replaceState({}, '');
    }
    return () => abortRef.current?.abort();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleReset = () => {
    setUiState('idle');
    setResult(null);
    setErrorMsg('');
  };

  return (
    <main className="guidance" id="main-content" aria-label="Guidance page">
      <div className="container">

        {/* ── Idle / form ─── */}
        {uiState === 'idle' && (
          <div className="guidance__form-wrap animate-slide-up">
            <h1 className="guidance__form-title">Describe your situation</h1>
            <p className="guidance__form-sub">
              Tell us what happened in your own words. We'll help you understand your rights and options.
            </p>
            <QueryInput onSubmit={runQuery} />
            <Disclaimer compact />
          </div>
        )}

        {/* ── Loading ─── */}
        {uiState === 'loading' && <LoadingState />}

        {/* ── Success ─── */}
        {uiState === 'success' && result && (
          <div className="animate-slide-up">
            <GuidanceResult result={result} query={query} onNewQuery={handleReset} />
          </div>
        )}

        {/* ── Clarification required ─── */}
        {uiState === 'clarification' && result && (
          <div className="guidance__clarification animate-slide-up" role="alert">
            <div className="guidance__clarification-box">
              <span className="guidance__clarification-icon" aria-hidden="true">🤔</span>
              <h2>We need a bit more detail</h2>
              <p>{result.message ?? 'Please provide more specific details about your situation.'}</p>
            </div>
            <div className="guidance__form-wrap">
              <QueryInput onSubmit={runQuery} initialMessage={query} />
            </div>
          </div>
        )}

        {/* ── Error ─── */}
        {uiState === 'error' && (
          <div className="guidance__error animate-slide-up" role="alert">
            <div className="guidance__error-box">
              <span className="guidance__error-icon" aria-hidden="true">⚠</span>
              <h2>Something went wrong</h2>
              <p>{errorMsg}</p>
              <div className="guidance__error-actions">
                <button className="btn-primary" onClick={() => runQuery(query, 'auto')}>
                  Try Again
                </button>
                <button className="btn-ghost" onClick={() => navigate('/')}>
                  Back to Home
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </main>
  );
}
