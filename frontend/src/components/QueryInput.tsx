// src/components/QueryInput.tsx
import { useState, useRef, useCallback } from 'react';
import type { Language } from '../types/civicSync';
import './QueryInput.css';

interface QueryInputProps {
  onSubmit: (message: string, language: Language) => void;
  loading?: boolean;
  initialMessage?: string;
}

const LANGUAGES: { value: Language; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'en',   label: 'English' },
  { value: 'hi',   label: 'हिन्दी' },
  { value: 'gu',   label: 'ગુજરાતી' },
];

const MAX_LENGTH = 2000;

export default function QueryInput({ onSubmit, loading = false, initialMessage = '' }: QueryInputProps) {
  const [message, setMessage]   = useState(initialMessage);
  const [language, setLanguage] = useState<Language>('auto');
  const [error, setError]       = useState('');
  const textareaRef             = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = message.trim();
    if (!trimmed) {
      setError('Please describe your situation before submitting.');
      textareaRef.current?.focus();
      return;
    }
    if (trimmed.length > MAX_LENGTH) {
      setError(`Please keep your description under ${MAX_LENGTH} characters.`);
      return;
    }
    setError('');
    onSubmit(trimmed, language);
  }, [message, language, onSubmit]);

  const remaining = MAX_LENGTH - message.length;

  return (
    <form className="query-input" onSubmit={handleSubmit} noValidate aria-label="Legal query form">
      <div className="query-input__field-wrap">
        <label htmlFor="qi-textarea" className="sr-only">
          Describe your situation
        </label>
        <textarea
          id="qi-textarea"
          ref={textareaRef}
          className={`query-input__textarea${error ? ' query-input__textarea--error' : ''}`}
          placeholder="Tell us what happened — in your own words…"
          value={message}
          onChange={e => { setMessage(e.target.value); if (error) setError(''); }}
          rows={4}
          maxLength={MAX_LENGTH + 1}
          disabled={loading}
          aria-required="true"
          aria-describedby={error ? 'qi-error' : 'qi-hint'}
        />
        <span className={`query-input__count${remaining < 100 ? ' query-input__count--warn' : ''}`} aria-live="polite">
          {remaining < MAX_LENGTH ? `${remaining} characters remaining` : ''}
        </span>
      </div>

      {error && (
        <p id="qi-error" className="query-input__error" role="alert" aria-live="assertive">
          {error}
        </p>
      )}

      <p id="qi-hint" className="query-input__hint">
        Example: "My employer has not paid my salary for two months."
      </p>

      <div className="query-input__controls">
        {/* Language selector */}
        <div className="query-input__lang-wrap">
          <label htmlFor="qi-lang" className="query-input__lang-label">Language</label>
          <select
            id="qi-lang"
            className="query-input__lang"
            value={language}
            onChange={e => setLanguage(e.target.value as Language)}
            disabled={loading}
            aria-label="Select language"
          >
            {LANGUAGES.map(l => (
              <option key={l.value} value={l.value}>{l.label}</option>
            ))}
          </select>
        </div>

        <div className="query-input__btn-group">
          {/* Voice placeholder — Phase 7 */}
          <button
            type="button"
            className="query-input__voice-btn"
            disabled
            aria-label="Voice input — coming soon"
            title="Voice assistance — coming soon"
          >
            <span aria-hidden="true">🎙</span>
            <span>Voice</span>
            <span className="query-input__badge">Soon</span>
          </button>

          <button
            type="submit"
            className="query-input__submit"
            disabled={loading || !message.trim()}
            aria-busy={loading}
          >
            {loading ? (
              <span className="query-input__spinner" aria-hidden="true" />
            ) : null}
            {loading ? 'Getting Guidance…' : 'Get Guidance'}
          </button>
        </div>
      </div>
    </form>
  );
}
