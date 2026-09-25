import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import Home from '../pages/Home';
import { queryCivicSync, ApiError } from '../api/civicSync';
import type { CivicSyncResponse } from '../types/civicSync';

vi.mock('../api/civicSync', () => {
  class MockApiError extends Error {
    statusCode: number;

    constructor(statusCode: number, message: string) {
      super(message);
      this.statusCode = statusCode;
      this.name = 'ApiError';
    }
  }

  return {
    queryCivicSync: vi.fn(),
    ApiError: MockApiError,
  };
});

const MOCK_SUCCESS_RESPONSE: CivicSyncResponse = {
  status: 'success',
  conversation_id: 'test_session_999',
  situation: {
    domain: 'insurance',
    issue: 'claim rejection',
  },
  qualification: {
    status: 'supported',
    applicable_information: [{ text: 'Insurers must settle claims within 30 days.' }],
    rights_or_protections: [{ text: 'Policyholder can appeal to Ombudsman.' }],
  },
  actions: {
    immediate_steps: [{ text: 'Gather medical records.' }],
    formal_remedies: [{ text: 'File Ombudsman complaint.' }],
  },
  answer: 'Your insurer rejected your claim. You have a right to challenge this rejection.',
  sources: [
    {
      chunk_id: 2070,
      source_file: 'civicsync_insurance_regulatory_v2',
      page_start: 24,
      similarity: 0.835,
    },
  ],
  limitations: ['Verified against current IRDAI rules.'],
};

function renderHome() {
  return render(
    <BrowserRouter>
      <Home />
    </BrowserRouter>
  );
}

describe('CivicSync Frontend Conversational Assistant', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    expect(ApiError).toBeDefined();
  });

  // 1. Home renders
  it('renders the Home page with branding and assistant welcome message', () => {
    renderHome();
    expect(screen.getByText('Conversational Assistant')).toBeInTheDocument();
    expect(screen.getByText(/Hello! I am CivicSync/)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Ask CivicSync a question/)).toBeInTheDocument();
  });

  // 2. Query input works
  it('allows the user to type in the query input box', () => {
    renderHome();
    const textarea = screen.getByPlaceholderText(/Ask CivicSync a question/) as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'Unpaid salary for two months' } });
    expect(textarea.value).toBe('Unpaid salary for two months');
  });

  // 3. Language selector works
  it('contains a language selector dropdown', () => {
    renderHome();
    const select = screen.getAllByRole('combobox')[0];
    expect(select).toBeInTheDocument();
    
    // Default value is auto
    expect(select).toHaveValue('auto');
    
    // Can select English/Hindi
    fireEvent.change(select, { target: { value: 'hi' } });
    expect(select).toHaveValue('hi');
  });

  // 4. Category selection works
  it('displays category navigation options in the header link', () => {
    renderHome();
    const link = screen.getByText('Know Your Rights Catalog');
    expect(link).toBeInTheDocument();
    expect(link.closest('button')).toBeInTheDocument();
  });

  // 5. API request is generated correctly & 6. Loading state appears
  // 7. Successful response renders & 8. Sources render & 9. Disclaimer renders
  it('submits API request correctly, shows loading, and renders response, sources, and disclaimer', async () => {
    vi.mocked(queryCivicSync).mockResolvedValue(MOCK_SUCCESS_RESPONSE);

    renderHome();
    const textarea = screen.getByPlaceholderText(/Ask CivicSync a question/);
    const sendButton = screen.getByLabelText('Send message');

    // Type and send
    fireEvent.change(textarea, { target: { value: 'My insurer rejected my claim' } });
    fireEvent.click(sendButton);

    // Verify loading indicator is displayed
    const loadingDots = screen.getAllByLabelText('Thinking')[0] || screen.getByLabelText('Thinking');
    expect(loadingDots).toBeInTheDocument();

    // Wait for response to render
    await waitFor(() => {
      expect(screen.getByText('Your insurer rejected your claim. You have a right to challenge this rejection.')).toBeInTheDocument();
    });

    // Check queryCivicSync API call signature
    expect(queryCivicSync).toHaveBeenCalledWith(
      'My insurer rejected my claim',
      'auto',
      undefined
    );

    // 7. Check structured result sections
    expect(screen.getByText('What you should know')).toBeInTheDocument();
    expect(screen.getByText('Insurers must settle claims within 30 days.')).toBeInTheDocument();
    expect(screen.getByText('Your rights & protections')).toBeInTheDocument();
    expect(screen.getByText('Policyholder can appeal to Ombudsman.')).toBeInTheDocument();
    expect(screen.getByText('What you can do')).toBeInTheDocument();
    expect(screen.getByText('Gather medical records.')).toBeInTheDocument();

    // 8. Sources render
    const sourcesToggle = screen.getByText(/Sources used/);
    expect(sourcesToggle).toBeInTheDocument();
    
    // Expand sources
    fireEvent.click(sourcesToggle);
    expect(screen.getByText('Insurance regulatory')).toBeInTheDocument();
    expect(screen.getByText('Page 24')).toBeInTheDocument();

    // 9. Disclaimer renders
    expect(screen.getByText(/Legal information is provided for awareness/)).toBeInTheDocument();
  });

  // 10. API error renders safely
  it('renders API errors gracefully without exposing sensitive internals', async () => {
    vi.mocked(queryCivicSync).mockRejectedValue(new Error('Internal Supabase Secret Credential Error 10065'));

    renderHome();
    const textarea = screen.getByPlaceholderText(/Ask CivicSync a question/);
    const sendButton = screen.getByLabelText('Send message');

    // Type and send
    fireEvent.change(textarea, { target: { value: 'Unpaid salary' } });
    fireEvent.click(sendButton);

    // Verify fallback error message is shown safely
    await waitFor(() => {
      const errorCard = screen.getByRole('alert');
      expect(errorCard).toBeInTheDocument();
      // Should not contain "Supabase" or "Secret" or "Credential" or internal codes
      expect(errorCard).not.toHaveTextContent('Supabase');
      expect(errorCard).not.toHaveTextContent('Secret');
      expect(errorCard).not.toHaveTextContent('Credential');
      expect(errorCard).toHaveTextContent('Could not connect to the CivicSync assistant');
    });
  });

  // 11. Mobile layout check
  it('contains viewport-friendly semantic tags and accessibility controls for mobile layouts', () => {
    renderHome();
    const main = screen.getByRole('main');
    expect(main).toHaveClass('chat-container');
    const inputRow = screen.getByLabelText('Message Input').closest('div');
    expect(inputRow).toHaveClass('chat-dock__input-row');
  });
});
