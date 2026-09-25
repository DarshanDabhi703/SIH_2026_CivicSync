// src/components/CategoryGrid.tsx
import { useNavigate } from 'react-router-dom';
import { CATEGORIES } from '../data/categories';
import type { Category } from '../types/civicSync';
import './CategoryGrid.css';

interface CategoryGridProps {
  onSelect?: (suggestion: string) => void;
}

function CategoryCard({ cat, onSelect }: { cat: Category; onSelect?: (s: string) => void }) {
  const navigate = useNavigate();

  const handleClick = () => {
    if (onSelect) {
      onSelect(cat.suggestions[0]);
    } else {
      navigate('/rights', { state: { categoryId: cat.id } });
    }
  };

  return (
    <button
      className="cat-card"
      onClick={handleClick}
      aria-label={`Explore ${cat.label}`}
      style={{ '--cat-color': cat.color } as React.CSSProperties}
    >
      <span className="cat-card__icon" aria-hidden="true">{cat.icon}</span>
      <span className="cat-card__label">{cat.label}</span>
    </button>
  );
}

export default function CategoryGrid({ onSelect }: CategoryGridProps) {
  return (
    <section className="cat-grid" aria-labelledby="cat-grid-heading">
      <h2 id="cat-grid-heading" className="cat-grid__heading">
        Or explore by topic
      </h2>
      <div className="cat-grid__cards" role="list">
        {CATEGORIES.map(cat => (
          <div key={cat.id} role="listitem">
            <CategoryCard cat={cat} onSelect={onSelect} />
          </div>
        ))}
      </div>
    </section>
  );
}
