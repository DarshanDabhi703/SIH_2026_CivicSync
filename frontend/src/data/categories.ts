// src/data/categories.ts
// Static category data — suggestions only, no hardcoded legal content.

import type { Category } from '../types/civicSync';

export const CATEGORIES: Category[] = [
  {
    id: 'labour',
    label: 'Labour Rights',
    icon: '👷',
    color: '#1d4ed8',
    suggestions: [
      'My employer has not paid my salary for two months.',
      'My employer deducted money from my wages without explanation.',
      'I was dismissed without notice or reason.',
      'My employer is not giving me my legal leave entitlement.',
      'I am being forced to work overtime without pay.',
    ],
  },
  {
    id: 'consumer',
    label: 'Consumer Rights',
    icon: '🛒',
    color: '#0d9488',
    suggestions: [
      'I bought a defective product and the seller refuses a refund.',
      'An e-commerce platform is not delivering my order or refund.',
      'I received a product different from what I ordered.',
      'A service provider is not delivering what was promised.',
    ],
  },
  {
    id: 'insurance',
    label: 'Insurance Rights',
    icon: '🏥',
    color: '#7c3aed',
    suggestions: [
      'My insurer rejected my hospital claim.',
      'My insurance claim has been delayed for months.',
      'I want to know my rights as a policyholder.',
      'The insurer is not responding to my grievance.',
    ],
  },
  {
    id: 'land_property',
    label: 'Property & Tenancy',
    icon: '🏠',
    color: '#b45309',
    suggestions: [
      'My landlord is not returning my security deposit.',
      'My landlord is threatening to evict me without notice.',
      'There is a dispute over my property ownership.',
      'My landlord is not carrying out necessary repairs.',
    ],
  },
  {
    id: 'traffic',
    label: 'Traffic Rights',
    icon: '🚦',
    color: '#dc2626',
    suggestions: [
      'A traffic challan was issued to me unfairly.',
      'My vehicle was towed without proper notice.',
      'I want to understand traffic violation penalties.',
      'My driving licence was suspended and I want to know my rights.',
    ],
  },
  {
    id: 'women_safety',
    label: 'Women Safety',
    icon: '🛡️',
    color: '#be185d',
    suggestions: [
      'I am being harassed at my workplace.',
      'I want to know how to file a complaint about harassment.',
      'I need help understanding protection orders.',
      'I am facing domestic violence and need to know my options.',
    ],
  },
];

export const DOMAIN_LABELS: Record<string, string> = {
  labour:        'Labour',
  consumer:      'Consumer',
  insurance:     'Insurance',
  land_property: 'Property & Tenancy',
  traffic:       'Traffic',
  women_safety:  'Women Safety',
  unknown:       'General',
};

export const INTENT_LABELS: Record<string, string> = {
  seek_remedy:          'Seeking remedy',
  file_complaint:       'Filing a complaint',
  understand_rights:    'Understanding rights',
  understand_consequences: 'Understanding consequences',
  obtain_document:      'Obtaining a document',
  understand_procedure: 'Understanding procedure',
  general_information:  'General information',
  unknown:              'General enquiry',
};
