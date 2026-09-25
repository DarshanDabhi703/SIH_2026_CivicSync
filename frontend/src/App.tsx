// src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Home from './pages/Home';
import Guidance from './pages/Guidance';
import KnowYourRights from './pages/KnowYourRights';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <Header />
      <Routes>
        <Route path="/"         element={<Home />} />
        <Route path="/guidance" element={<Guidance />} />
        <Route path="/rights"   element={<KnowYourRights />} />
        <Route path="*"         element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

function NotFound() {
  return (
    <main style={{ padding: '60px 20px', textAlign: 'center' }}>
      <h1 style={{ fontSize: '2rem', marginBottom: '12px', color: 'var(--color-primary)' }}>
        Page not found
      </h1>
      <p style={{ color: 'var(--color-text-muted)', marginBottom: '24px' }}>
        The page you are looking for does not exist.
      </p>
      <a href="/" style={{ color: 'var(--color-primary)', fontWeight: 600 }}>← Return home</a>
    </main>
  );
}
