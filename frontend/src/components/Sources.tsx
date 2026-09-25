// src/components/Sources.tsx
import { useState } from 'react';
import type { EvidenceSource } from '../types/civicSync';
import './Sources.css';

interface SourcesProps {
  sources: EvidenceSource[];
}

function SourceCard({ source, index }: { source: EvidenceSource; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const name = source.source_file
    ? source.source_file.replace('civicsync_', '').replace(/_/g, ' ').replace(/\bv\d+\b/g, '').trim()
    : 'Regulatory document';
  const displayName = name.charAt(0).toUpperCase() + name.slice(1);
  const sim = source.similarity ? Math.round(source.similarity * 100) : null;

  return (
    <div className="src-card" aria-label={`Source ${index + 1}: ${displayName}`}>
      <div className="src-card__summary">
        <span className="src-card__index" aria-hidden="true">{index + 1}</span>
        <div className="src-card__meta">
          <p className="src-card__name">{displayName}</p>
          {source.page_start != null && (
            <p className="src-card__page">
              Page {source.page_start}{source.page_end && source.page_end !== source.page_start ? `–${source.page_end}` : ''}
            </p>
          )}
        </div>
        {sim !== null && (
          <span className="src-card__sim" title={`Relevance: ${sim}%`} aria-label={`${sim}% relevance`}>
            {sim}%
          </span>
        )}
        <button
          className="src-card__toggle"
          onClick={() => setExpanded(e => !e)}
          aria-expanded={expanded}
          aria-label={expanded ? 'Hide source details' : 'View source details'}
        >
          {expanded ? '▲ Hide' : '▼ Details'}
        </button>
      </div>

      {expanded && (
        <div className="src-card__detail animate-fade-in" aria-live="polite">
          {source.chunk_id != null && (
            <p className="src-card__detail-row">
              <span>Reference</span>
              <span>#{String(source.chunk_id)}</span>
            </p>
          )}
          {source.domain && (
            <p className="src-card__detail-row">
              <span>Domain</span>
              <span>{source.domain}</span>
            </p>
          )}
          {sim !== null && (
            <p className="src-card__detail-row">
              <span>Relevance score</span>
              <span>{sim}%</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function Sources({ sources }: SourcesProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <section className="sources" aria-labelledby="sources-heading">
      <h2 id="sources-heading" className="sources__heading">
        Sources
      </h2>
      <p className="sources__desc">
        This guidance is based on the following legal and regulatory documents.
      </p>
      <div className="sources__list" role="list">
        {sources.map((s, i) => (
          <div key={i} role="listitem">
            <SourceCard source={s} index={i} />
          </div>
        ))}
      </div>
    </section>
  );
}
