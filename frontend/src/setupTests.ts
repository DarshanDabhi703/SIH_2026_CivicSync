import '@testing-library/jest-dom';

// Stub out scrollIntoView since jsdom doesn't implement it
if (!window.HTMLElement.prototype.scrollIntoView) {
  window.HTMLElement.prototype.scrollIntoView = function() {};
}
