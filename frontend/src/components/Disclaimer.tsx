// src/components/Disclaimer.tsx
import './Disclaimer.css';

interface DisclaimerProps {
  limitations?: string[];
  compact?: boolean;
}

export default function Disclaimer({ limitations = [], compact = false }: DisclaimerProps) {
  return (
    <aside className={`disclaimer${compact ? ' disclaimer--compact' : ''}`} aria-label="Legal disclaimer">
      <span className="disclaimer__icon" aria-hidden="true">ℹ</span>
      <div className="disclaimer__body">
        <p className="disclaimer__main">
          Legal information is provided for awareness and guidance only — not professional legal advice.
          Verify current requirements with the relevant official authority or a qualified legal professional.
        </p>
        {limitations.length > 0 && !compact && (
          <ul className="disclaimer__limitations" aria-label="Information limitations">
            {limitations.map((lim, i) => (
              <li key={i}>{lim}</li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
