// src/pages/KnowYourRights.tsx
import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { CATEGORIES } from '../data/categories';
import type { Category } from '../types/civicSync';
import './KnowYourRights.css';

function CategoryPanel({
  cat,
  active,
  onSelect,
}: {
  cat: Category;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <button
      className={`kyr-tab${active ? ' kyr-tab--active' : ''}`}
      onClick={() => onSelect(cat.id)}
      style={{ '--cat-color': cat.color } as React.CSSProperties}
      aria-pressed={active}
      aria-label={`Select ${cat.label}`}
    >
      <span className="kyr-tab__icon" aria-hidden="true">{cat.icon}</span>
      <span className="kyr-tab__label">{cat.label}</span>
    </button>
  );
}

export default function KnowYourRights() {
  const navigate  = useNavigate();
  const location  = useLocation();
  const initCat   = (location.state as { categoryId?: string } | null)?.categoryId ?? CATEGORIES[0].id;
  const [active, setActive] = useState(initCat);

  const currentCat = CATEGORIES.find(c => c.id === active) ?? CATEGORIES[0];

  const handleSuggestion = (text: string) => {
    navigate('/', { state: { message: text, language: 'auto' } });
  };

  return (
    <main className="kyr" id="main-content" aria-label="Know Your Rights">
      <div className="kyr__hero">
        <div className="container">
          <h1 className="kyr__title">Know Your Rights</h1>
          <p className="kyr__subtitle">
            Explore common situations by category. Click a situation to receive guidance.
          </p>
        </div>
      </div>

      <div className="container kyr__body">
        {/* ── Category tabs ── */}
        <nav className="kyr__tabs" aria-label="Rights categories">
          {CATEGORIES.map(cat => (
            <CategoryPanel
              key={cat.id}
              cat={cat}
              active={active === cat.id}
              onSelect={setActive}
            />
          ))}
        </nav>

        {/* ── Suggestions panel ── */}
        <section
          className="kyr__panel"
          aria-labelledby="kyr-panel-heading"
          key={active}
        >
          <div className="kyr__panel-header">
            <span className="kyr__panel-icon" aria-hidden="true">{currentCat.icon}</span>
            <h2 id="kyr-panel-heading" className="kyr__panel-title">
              {currentCat.label}
            </h2>
          </div>
          <p className="kyr__panel-desc">
            Select a situation below to get guidance from CivicSync.
            These are common scenarios — describe your specific situation for personalised guidance.
          </p>
          <ul className="kyr__suggestions" role="list">
            {currentCat.suggestions.map((s, i) => (
              <li key={i} role="listitem">
                <button
                  className="kyr__suggestion"
                  onClick={() => handleSuggestion(s)}
                  aria-label={`Get guidance: ${s}`}
                >
                  <span className="kyr__suggestion-text">{s}</span>
                  <span className="kyr__suggestion-arrow" aria-hidden="true">→</span>
                </button>
              </li>
            ))}
          </ul>

          <div className="kyr__custom">
            <p>Don't see your situation?</p>
            <button
              className="kyr__custom-btn"
              onClick={() => navigate('/')}
            >
              Describe your own situation →
            </button>
          </div>
        </section>
      </div>
    </main>
  );
}
