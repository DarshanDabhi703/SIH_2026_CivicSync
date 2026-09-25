// src/pages/Home.tsx
import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Send, Mic, RefreshCw, BookOpen, ChevronDown, ChevronUp, AlertCircle, Sparkles } from 'lucide-react';
import { queryCivicSync, ApiError } from '../api/civicSync';
import type { CivicSyncResponse, Language } from '../types/civicSync';
import { DOMAIN_LABELS } from '../data/categories';
import Disclaimer from '../components/Disclaimer';
import './Home.css';

interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: Date;
  result?: CivicSyncResponse | null;
  loading?: boolean;
}

const LANGUAGES: { value: Language; label: string }[] = [
  { value: 'auto', label: 'Auto Detect' },
  { value: 'en',   label: 'English' },
  { value: 'hi',   label: 'हिन्दी (Hindi)' },
  { value: 'gu',   label: 'ગુજરાતી (Gujarati)' },
];

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: "Hello! I am CivicSync, your conversational legal guidance assistant.\n\nDescribe what happened in your own words (e.g. \"My employer hasn't paid my salary for two months\"). I will help you understand your rights, next steps, and where to seek help.",
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState<Language>('auto');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const runQuery = useCallback(async (messageText: string, language: Language) => {
    const trimmed = messageText.trim();
    if (!trimmed) return;

    setLoading(true);
    setErrorMsg('');

    // 1. Add user message
    const userMsgId = uuid();
    const newUserMsg: Message = {
      id: userMsgId,
      sender: 'user',
      text: trimmed,
      timestamp: new Date(),
    };

    // 2. Add temporary loading assistant message
    const assistantMsgId = uuid();
    const loadingMsg: Message = {
      id: assistantMsgId,
      sender: 'assistant',
      text: '',
      timestamp: new Date(),
      loading: true,
    };

    setMessages(prev => [...prev, newUserMsg, loadingMsg]);

    try {
      // 3. Query API (passing latest conversationId if any)
      // Note: We use a functional update pattern or read latest conversationId state.
      // Since conversationId state might change, we read it directly from state closure or state ref.
      // To keep it clean, we can capture the state directly.
      const response = await queryCivicSync(
        trimmed,
        language,
        conversationId || undefined
      );

      // Persist conversation id
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }

      // Update assistant message with response
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMsgId
            ? {
                ...m,
                text: response.answer || response.message || 'No answer generated.',
                result: response,
                loading: false,
              }
            : m
        )
      );
    } catch (err) {
      const msg = err instanceof ApiError
        ? err.message
        : 'Could not connect to the CivicSync assistant. Please try again.';
      
      setErrorMsg(msg);
      // Remove the loading message if failed
      setMessages(prev => prev.filter(m => m.id !== assistantMsgId));
    } finally {
      setLoading(false);
    }
  }, [conversationId]);

  // Execute query received via Router state on mount
  useEffect(() => {
    const state = location.state as { message?: string; language?: Language } | null;
    if (state?.message) {
      runQuery(state.message, state.language ?? 'auto');
      // Clear state so refresh/back doesn't trigger the query again
      window.history.replaceState({}, '');
    }
  }, [location.state, runQuery]);

  const handleSend = () => {
    const trimmed = inputValue.trim();
    if (!trimmed || loading) return;
    setInputValue('');
    runQuery(trimmed, selectedLanguage);
  };

  const handleClear = () => {
    setMessages([
      {
        id: 'welcome',
        sender: 'assistant',
        text: "Hello! I am CivicSync, your conversational legal guidance assistant.\n\nDescribe what happened in your own words (e.g. \"My employer hasn't paid my salary for two months\"). I will help you understand your rights, next steps, and where to seek help.",
        timestamp: new Date(),
      }
    ]);
    setConversationId(null);
    setInputValue('');
    setErrorMsg('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <main className="chat-container" id="main-content">
      {/* ─── Header Panel ─── */}
      <section className="chat-header">
        <div className="container chat-header__inner">
          <div className="chat-header__brand">
            <Sparkles className="chat-header__brand-icon" />
            <h1>Conversational Assistant</h1>
          </div>
          <p className="chat-header__sub">
            Explain your situation in plain text. Get grounded, evidence-backed legal guidance dynamically.
          </p>
          <div className="chat-header__actions">
            <button
              className="chat-header__nav-btn"
              onClick={() => navigate('/rights')}
              aria-label="Explore Categories"
            >
              <BookOpen size={16} />
              <span>Know Your Rights Catalog</span>
            </button>
            {(messages.length > 1 || conversationId) && (
              <button
                className="chat-header__clear-btn"
                onClick={handleClear}
                aria-label="Reset Conversation"
              >
                <RefreshCw size={14} />
                <span>New Session</span>
              </button>
            )}
          </div>
        </div>
      </section>

      {/* ─── Messages View ─── */}
      <div className="container chat-messages-wrap">
        <div className="chat-messages" role="log" aria-label="Conversation history">
          {messages.map(msg => (
            <div
              key={msg.id}
              className={`chat-bubble-wrap chat-bubble-wrap--${msg.sender}`}
            >
              <div className="chat-avatar" aria-hidden="true">
                {msg.sender === 'user' ? '👤' : '⚖️'}
              </div>
              <div className="chat-bubble">
                {msg.loading ? (
                  <div className="chat-loading" aria-label="Thinking">
                    <span className="chat-loading__dot" style={{ animationDelay: '0ms' }}></span>
                    <span className="chat-loading__dot" style={{ animationDelay: '200ms' }}></span>
                    <span className="chat-loading__dot" style={{ animationDelay: '400ms' }}></span>
                  </div>
                ) : (
                  <div className="chat-bubble__content">
                    <p className="chat-bubble__text">{msg.text}</p>
                    {msg.result && <StructuredOutput result={msg.result} />}
                  </div>
                )}
                <span className="chat-bubble__time">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </div>
          ))}
          {errorMsg && (
            <div className="chat-error-card animate-slide-up" role="alert">
              <AlertCircle className="chat-error-card__icon" />
              <div className="chat-error-card__body">
                <p>{errorMsg}</p>
                <button onClick={handleSend} className="chat-error-card__retry">
                  Try Again
                </button>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      </div>

      {/* ─── Persistent Input Dock ─── */}
      <div className="chat-dock">
        <div className="container chat-dock__inner">
          <div className="chat-dock__controls">
            <div className="chat-dock__lang">
              <label htmlFor="lang-select" className="sr-only">Select response language</label>
              <select
                id="lang-select"
                value={selectedLanguage}
                onChange={e => setSelectedLanguage(e.target.value as Language)}
                disabled={loading}
              >
                {LANGUAGES.map(lang => (
                  <option key={lang.value} value={lang.value}>{lang.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="chat-dock__input-row">
            <textarea
              ref={inputRef}
              rows={1}
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask CivicSync a question or describe your situation..."
              disabled={loading}
              aria-label="Message Input"
            />
            
            {/* Future Voice assistance button */}
            <button
              type="button"
              className="chat-dock__voice-btn"
              disabled
              title="Voice assistant coming soon"
              aria-label="Voice input — coming soon"
            >
              <Mic size={18} />
              <span className="chat-dock__voice-badge">Soon</span>
            </button>

            <button
              className="chat-dock__send-btn"
              onClick={handleSend}
              disabled={loading || !inputValue.trim()}
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </div>
          <div className="chat-dock__disclaimer">
            <Disclaimer compact />
          </div>
        </div>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Structured Output Component
// Renders the structured findings visually distinct inside the chat thread.
// ---------------------------------------------------------------------------
function StructuredOutput({ result }: { result: CivicSyncResponse }) {
  const sit = result.situation;
  const qual = result.qualification;
  const act = result.actions;
  const sources = result.sources || [];

  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  // Group all items
  const applicable = qual?.applicable_information || [];
  const rights = qual?.rights_or_protections || [];
  const actions = [
    ...(act?.immediate_steps || []),
    ...(act?.formal_remedies || []),
  ];
  const documents = qual?.documents_or_evidence || act?.information_to_collect || [];
  const authorities = qual?.authorities_or_channels || act?.authority_pathway || [];

  const hasStructuredData = 
    applicable.length > 0 || 
    rights.length > 0 || 
    actions.length > 0 || 
    documents.length > 0 || 
    authorities.length > 0;

  if (!hasStructuredData && sources.length === 0) return null;

  return (
    <div className="structured-output animate-fade-in">
      {sit && (sit.domain !== 'unknown' || sit.issue !== 'unknown') && (
        <div className="so-situation">
          <strong>What I understood:</strong>{' '}
          <span>
            {sit.domain && DOMAIN_LABELS[sit.domain] ? DOMAIN_LABELS[sit.domain] : sit.domain}
            {sit.issue && ` — ${sit.issue}`}
          </span>
        </div>
      )}

      {/* ─── Applicable Info / General Guidance ─── */}
      {applicable.length > 0 && (
        <div className="so-section">
          <p className="so-section__title">What you should know</p>
          <ul className="so-section__list">
            {applicable.map((item, idx) => (
              <li key={idx}>{typeof item === 'string' ? item : item.text}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ─── Rights / Protections ─── */}
      {rights.length > 0 && (
        <div className="so-section so-section--green">
          <p className="so-section__title">Your rights & protections</p>
          <ul className="so-section__list">
            {rights.map((item, idx) => (
              <li key={idx}>{typeof item === 'string' ? item : item.text}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ─── Action steps ─── */}
      {actions.length > 0 && (
        <div className="so-section so-section--amber">
          <p className="so-section__title">What you can do</p>
          <ol className="so-section__list so-section__list--ordered">
            {actions.map((item, idx) => (
              <li key={idx}>{typeof item === 'string' ? item : item.text}</li>
            ))}
          </ol>
        </div>
      )}

      {/* ─── Documents / Keep ready ─── */}
      {documents.length > 0 && (
        <div className="so-section">
          <p className="so-section__title">What you may need</p>
          <ul className="so-section__list">
            {documents.map((item, idx) => (
              <li key={idx}>{typeof item === 'string' ? item : item.text}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ─── Help / Authorities ─── */}
      {authorities.length > 0 && (
        <div className="so-section so-section--purple">
          <p className="so-section__title">Where to get help</p>
          <ul className="so-section__list">
            {authorities.map((item, idx) => (
              <li key={idx}>{typeof item === 'string' ? item : item.text}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ─── Expandable Sources Used ─── */}
      {sources.length > 0 && (
        <div className="so-sources">
          <button
            className="so-sources__toggle"
            onClick={() => setSourcesExpanded(e => !e)}
            aria-expanded={sourcesExpanded}
          >
            <span>Sources used ({sources.length})</span>
            {sourcesExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {sourcesExpanded && (
            <div className="so-sources__panel animate-fade-in" role="list">
              {sources.map((s, idx) => {
                const name = s.source_file
                  ? s.source_file.replace('civicsync_', '').replace(/_/g, ' ').replace(/\bv\d+\b/g, '').trim()
                  : 'Regulatory document';
                const displayName = name.charAt(0).toUpperCase() + name.slice(1);
                return (
                  <div key={idx} className="so-sources__item" role="listitem">
                    <span className="so-sources__item-name">{displayName}</span>
                    {s.page_start != null && (
                      <span className="so-sources__item-page">Page {s.page_start}</span>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Simple unique ID generator
function uuid() {
  return Math.random().toString(36).substring(2, 9);
}
