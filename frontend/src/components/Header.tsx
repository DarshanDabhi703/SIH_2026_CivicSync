// src/components/Header.tsx
import { Link, useLocation } from 'react-router-dom';
import './Header.css';

export default function Header() {
  const loc = useLocation();

  return (
    <header className="header" role="banner">
      <div className="container header__inner">
        <Link to="/" className="header__brand" aria-label="CivicSync — home">
          <span className="header__logo-mark" aria-hidden="true">⚖</span>
          <span className="header__name">CivicSync</span>
        </Link>

        <nav className="header__nav" aria-label="Main navigation">
          <Link
            to="/"
            className={`header__link${loc.pathname === '/' ? ' header__link--active' : ''}`}
          >
            Home
          </Link>
          <Link
            to="/rights"
            className={`header__link${loc.pathname === '/rights' ? ' header__link--active' : ''}`}
          >
            Know Your Rights
          </Link>
        </nav>
      </div>
    </header>
  );
}
