import {
  TextAnalysisResult,
  MemeAnalysisResult,
  BatchAnalysisResponse,
  SystemHealth,
  SocialAnalysisResponse,
} from '../types';

const API_BASE = '/api';

export async function checkHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

export async function analyzeText(
  text: string,
  language?: string,
  enableJev: boolean = true
): Promise<TextAnalysisResult> {
  const res = await fetch(`${API_BASE}/analyze/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, language, enable_jev: enableJev }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(err.detail || 'Text analysis request failed');
  }
  return res.json();
}

export async function analyzeMeme(
  image: File,
  text?: string
): Promise<MemeAnalysisResult> {
  const formData = new FormData();
  formData.append('image', image);
  if (text) {
    formData.append('text', text);
  }

  const res = await fetch(`${API_BASE}/analyze/meme`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Meme analysis failed' }));
    throw new Error(err.detail || 'Multimodal meme analysis request failed');
  }
  return res.json();
}

export async function analyzeMemeBase64(
  imageBase64: string,
  text?: string
): Promise<MemeAnalysisResult> {
  const res = await fetch(`${API_BASE}/analyze/meme-json`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageBase64, text: text || '' }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Meme analysis failed' }));
    throw new Error(err.detail || 'Multimodal analysis request failed');
  }
  return res.json();
}

export async function analyzeBatch(
  items: { id: string; text: string }[]
): Promise<BatchAnalysisResponse> {
  const res = await fetch(`${API_BASE}/analyze/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Batch analysis failed' }));
    throw new Error(err.detail || 'Batch analysis request failed');
  }
  return res.json();
}

export async function fetchResearchMetrics(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/metrics`);
  if (!res.ok) return null;
  return res.json();
}

export async function fetchCaseStudies(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/case-studies`);
  if (!res.ok) return null;
  return res.json();
}

export async function analyzeSocialUrl(
  url: string,
  platform: string = 'auto',
  maxComments: number = 50,
  demoFallback: boolean = true,
  enableJev: boolean = false
): Promise<SocialAnalysisResponse> {
  const res = await fetch(`${API_BASE}/analyze/social`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url,
      platform,
      max_comments: maxComments,
      demo_fallback: demoFallback,
      enable_jev: enableJev,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Social analysis request failed' }));
    throw new Error(err.detail || 'Failed to fetch and analyze social media comments');
  }
  return res.json();
}

// ── Profile Watch Endpoints ────────────────────────────────────

export async function fetchProfiles(): Promise<import('../types').ProfileRecord[]> {
  const res = await fetch(`${API_BASE}/profile-watch`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to list profiles' }));
    throw new Error(err.detail || 'Failed to list profiles');
  }
  return res.json();
}

export async function scanProfile(
  url: string,
  limit: number = 25,
  demoFallback: boolean = true,
  enableJev: boolean = false
): Promise<import('../types').ScanSummaryResponse> {
  const res = await fetch(`${API_BASE}/profile-watch/scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url,
      limit,
      demo_fallback: demoFallback,
      enable_jev: enableJev,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Profile scan failed' }));
    throw new Error(err.detail || 'Profile scan request failed');
  }
  return res.json();
}

export async function fetchProfileSummary(
  profileId: string
): Promise<import('../types').ProfileSummaryResponse> {
  const res = await fetch(`${API_BASE}/profile-watch/${encodeURIComponent(profileId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to fetch profile summary' }));
    throw new Error(err.detail || 'Failed to fetch profile summary');
  }
  return res.json();
}

export async function fetchProfileActivity(
  profileId: string,
  limit: number = 50,
  offset: number = 0,
  riskFilter?: string
): Promise<import('../types').ObservationRecord[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  if (riskFilter && riskFilter !== 'ALL') {
    params.set('risk_filter', riskFilter);
  }
  const res = await fetch(`${API_BASE}/profile-watch/${encodeURIComponent(profileId)}/activity?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to fetch profile activity' }));
    throw new Error(err.detail || 'Failed to fetch profile activity');
  }
  return res.json();
}

export async function fetchProfileEvents(
  profileId: string
): Promise<import('../types').PatternEventRecord[]> {
  const res = await fetch(`${API_BASE}/profile-watch/${encodeURIComponent(profileId)}/events`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to fetch profile events' }));
    throw new Error(err.detail || 'Failed to fetch profile events');
  }
  return res.json();
}

export async function startProfileMonitoring(
  profileId: string
): Promise<import('../types').ProfileRecord> {
  const res = await fetch(`${API_BASE}/profile-watch/${encodeURIComponent(profileId)}/monitor`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start monitoring' }));
    throw new Error(err.detail || 'Failed to activate monitoring');
  }
  return res.json();
}

export async function pauseProfileMonitoring(
  profileId: string
): Promise<import('../types').ProfileRecord> {
  const res = await fetch(`${API_BASE}/profile-watch/${encodeURIComponent(profileId)}/pause`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to pause monitoring' }));
    throw new Error(err.detail || 'Failed to pause monitoring');
  }
  return res.json();
}

export async function compareProfiles(
  profileA: string,
  profileB: string
): Promise<import('../types').ProfileComparisonResponse> {
  const res = await fetch(
    `${API_BASE}/profile-watch/compare?profile_a=${encodeURIComponent(profileA)}&profile_b=${encodeURIComponent(profileB)}`
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to compare profiles' }));
    throw new Error(err.detail || 'Failed to compare profiles');
  }
  return res.json();
}



