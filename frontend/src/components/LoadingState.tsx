// src/components/LoadingState.tsx
import { useEffect, useState } from 'react';
import './LoadingState.css';

const STEPS = [
  'Understanding your situation…',
  'Finding relevant legal information…',
  'Preparing your guidance…',
];

export default function LoadingState() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (step >= STEPS.length - 1) return;
    const t = setTimeout(() => setStep(s => s + 1), 2800);
    return () => clearTimeout(t);
  }, [step]);

  return (
    <div className="loading" role="status" aria-live="polite" aria-label="Loading guidance">
      <div className="loading__dots" aria-hidden="true">
        {[0, 1, 2].map(i => (
          <span key={i} className="loading__dot" style={{ animationDelay: `${i * 200}ms` }} />
        ))}
      </div>
      <p className="loading__text">{STEPS[step]}</p>
      <p className="loading__sub">This may take a few moments</p>
    </div>
  );
}
