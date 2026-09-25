// src/components/GuidanceResult.tsx
// Renders the full structured API response into citizen-friendly sections.
// No content is invented — only data returned by the API is displayed.

import type { CivicSyncResponse, QualificationItem } from '../types/civicSync';
import { DOMAIN_LABELS, INTENT_LABELS } from '../data/categories';
import Sources from './Sources';
import Disclaimer from './Disclaimer';
import './GuidanceResult.css';

interface GuidanceResultProps {
  result: CivicSyncResponse;
  query: string;
  onNewQuery: () => void;
}

function Section({ id, title, children, accent }: {
  id: string; title: string; children: React.ReactNode; accent?: string;
}) {
  return (
    <section className="gr-section" aria-labelledby={id} style={{ '--accent': accent } as React.CSSProperties}>
      <h2 id={id} className="gr-section__title">{title}</h2>
      <div className="gr-section__body">{children}</div>
    </section>
  );
}

function ItemCard({ item, index }: { item: QualificationItem | string; index: number }) {
  const text = typeof item === 'string' ? item : item.text;
  if (!text) return null;
  return (
    <div className="gr-item-card">
      <span className="gr-item-card__num" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
      <p className="gr-item-card__text">{text}</p>
    </div>
  );
}

function parseAnswer(raw: string): Record<string, string> {
  const sections: Record<string, string> = {};
  const regex = /\*{0,2}([A-Z][A-Z\s\/]+?)\*{0,2}\s*\n([\s\S]*?)(?=\n\*{0,2}[A-Z][A-Z\s\/]+?\*{0,2}\s*\n|$)/g;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(raw)) !== null) {
    sections[m[1].trim()] = m[2].trim();
  }
  return sections;
}

export default function GuidanceResult({ result, query, onNewQuery }: GuidanceResultProps) {
  const sit  = result.situation;
  const qual = result.qualification;
  const act  = result.actions;

  const allActions = [
    ...(act?.immediate_steps ?? []),
    ...(act?.formal_remedies ?? []),
  ];

  const authorities = qual?.authorities_or_channels ?? act?.authority_pathway ?? [];
  const rights      = qual?.rights_or_protections ?? [];
  const applicable  = qual?.applicable_information ?? [];
  const limitations = result.limitations ?? qual?.missing_information ?? [];

  const parsedAnswer = result.answer ? parseAnswer(result.answer) : {};

  const getSection = (...keys: string[]) => {
    for (const k of keys) {
      const v = Object.entries(parsedAnswer).find(([kk]) => kk.includes(k));
      if (v) return v[1];
    }
    return null;
  };

  const domainLabel = sit?.domain ? DOMAIN_LABELS[sit.domain] ?? sit.domain : null;
  const intentLabel = sit?.intent ? INTENT_LABELS[sit.intent] ?? sit.intent : null;

  return (
    <article className="gr" aria-label="Guidance result">
      {/* ─── Situation badge ─── */}
      <header className="gr-header">
        <div className="gr-header__meta">
          {domainLabel && <span className="gr-badge gr-badge--domain">{domainLabel}</span>}
          {sit?.issue  && <span className="gr-badge gr-badge--issue">{sit.issue}</span>}
          {intentLabel && <span className="gr-badge gr-badge--intent">{intentLabel}</span>}
        </div>
        <blockquote className="gr-header__query">"{query}"</blockquote>
      </header>

      <div className="gr-body">

        {/* ─── What you should know ─── */}
        {(applicable.length > 0 || getSection('SOURCES SAY', 'WHAT THE')) && (
          <Section id="gr-know" title="What You Should Know" accent="#1a3a6b">
            {applicable.length > 0 ? (
              <div className="gr-items">
                {applicable.map((item, i) => <ItemCard key={i} item={item} index={i} />)}
              </div>
            ) : (
              <p className="gr-prose">{getSection('SOURCES SAY', 'WHAT THE')}</p>
            )}
          </Section>
        )}

        {/* ─── Rights ─── */}
        {(rights.length > 0 || getSection('RIGHTS', 'PROTECTIONS')) && (
          <Section id="gr-rights" title="Your Rights & Protections" accent="#065f46">
            {rights.length > 0 ? (
              <div className="gr-items gr-items--green">
                {rights.map((item, i) => <ItemCard key={i} item={item} index={i} />)}
              </div>
            ) : (
              <p className="gr-prose">{getSection('RIGHTS', 'PROTECTIONS')}</p>
            )}
          </Section>
        )}

        {/* ─── What you can do ─── */}
        {(allActions.length > 0 || getSection('WHAT YOU CAN DO', 'CAN DO')) && (
          <Section id="gr-actions" title="What You Can Do" accent="#d97706">
            {allActions.length > 0 ? (
              <div className="gr-steps">
                {allActions.map((item, i) => (
                  <div key={i} className="gr-step">
                    <span className="gr-step__num" aria-label={`Step ${i + 1}`}>{String(i + 1).padStart(2, '0')}</span>
                    <p className="gr-step__text">{typeof item === 'string' ? item : item.text}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="gr-prose">{getSection('WHAT YOU CAN DO', 'CAN DO')}</p>
            )}
          </Section>
        )}

        {/* ─── Where to seek help ─── */}
        {(authorities.length > 0 || getSection('WHERE TO GO', 'WHERE TO SEEK')) && (
          <Section id="gr-auth" title="Where to Seek Help" accent="#7c3aed">
            {authorities.length > 0 ? (
              <div className="gr-auth-list">
                {authorities.map((auth, i) => (
                  <div key={i} className="gr-auth-card">
                    <span className="gr-auth-card__icon" aria-hidden="true">🏛</span>
                    <p>{typeof auth === 'string' ? auth : auth.text}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="gr-prose">{getSection('WHERE TO GO', 'WHERE TO SEEK')}</p>
            )}
          </Section>
        )}

        {/* ─── Legal basis ─── */}
        {getSection('LEGAL BASIS') && (
          <Section id="gr-legal" title="Legal Basis">
            <p className="gr-prose gr-prose--muted">{getSection('LEGAL BASIS')}</p>
          </Section>
        )}

        {/* ─── Sources ─── */}
        {result.sources && result.sources.length > 0 && (
          <Sources sources={result.sources} />
        )}

        {/* ─── Limitations / Disclaimer ─── */}
        <Disclaimer limitations={limitations} />

        {/* ─── New query ─── */}
        <div className="gr-footer">
          <button className="gr-new-query" onClick={onNewQuery} aria-label="Start a new query">
            ← Ask another question
          </button>
        </div>
      </div>
    </article>
  );
}
