import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ProfileRecord,
  ObservationRecord,
  PatternEventRecord,
  AggregateMetrics,
  ProfileSummaryResponse,
  ScanSummaryResponse,
  ProfileComparisonResponse,
  RetrievalAudit,
  ProfileActivityRisk,
} from '../../types';
import {
  fetchProfiles,
  scanProfile,
  fetchProfileSummary,
  compareProfiles,
} from '../../api/client';

interface ProfileWatchOverlayProps {
  onGoToSocial?: () => void;
  onGoToText?: () => void;
  onProfileStateChange?: (
    profile: ProfileRecord | null,
    aggregates: AggregateMetrics | null,
    observations: ObservationRecord[],
    events: PatternEventRecord[],
    isScanning: boolean
  ) => void;
}

const PRESET_PROFILES = [
  {
    platform: 'reddit',
    label: '🤖 Reddit Public Profile',
    handle: 'u/test_analyst',
    url: 'https://www.reddit.com/user/test_analyst',
  },
  {
    platform: 'youtube',
    label: '📺 YouTube Channel',
    handle: '@ToxiscopeLab',
    url: 'https://www.youtube.com/@ToxiscopeLab',
  },
  {
    platform: 'twitter',
    label: '🐦 Twitter / X Account',
    handle: '@researcher_ai',
    url: 'https://twitter.com/researcher_ai',
  },
];

const SCAN_STAGES = [
  'INITIALIZING FORENSIC SCAN',
  'RETRIEVING HISTORICAL PUBLIC ACTIVITY',
  'ANALYZING MULTILINGUAL CONTENT (MuRIL + JEV + CLIP)',
  'BUILDING ACTIVITY PROFILE & AGGREGATIONS',
  'PROFILE INTELLIGENCE DOSSIER READY',
];

export const ProfileWatchOverlay: React.FC<ProfileWatchOverlayProps> = ({
  onGoToSocial,
  onGoToText,
  onProfileStateChange,
}) => {
  // Input and scan state
  const [inputUrl, setInputUrl] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [scanStageIndex, setScanStageIndex] = useState(0);
  const [scanLimit, setScanLimit] = useState(25);
  const [enableJev, setEnableJev] = useState(false);
  const [demoFallback, setDemoFallback] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Active Profile & Detailed Data
  const [profilesList, setProfilesList] = useState<ProfileRecord[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);
  const [profileSummary, setProfileSummary] = useState<ProfileSummaryResponse | null>(null);
  const [audit, setAudit] = useState<RetrievalAudit | null>(null);

  // Evidence Filtering & Inspection
  const [riskFilter, setRiskFilter] = useState<string>('ALL');
  const [typeFilter, setTypeFilter] = useState<string>('ALL');
  const [languageFilter, setLanguageFilter] = useState<string>('ALL');
  const [selectedPeriod, setSelectedPeriod] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedEvidenceItem, setSelectedEvidenceItem] = useState<ObservationRecord | null>(null);

  // Comparison State
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [compareTargetId, setCompareTargetId] = useState<string>('');
  const [comparisonResult, setComparisonResult] = useState<ProfileComparisonResponse | null>(null);
  const [isComparing, setIsComparing] = useState(false);

  // Load existing scanned profiles on mount
  useEffect(() => {
    loadScannedProfiles();
  }, []);

  // Animate scan stages when scanning is active
  useEffect(() => {
    if (!isScanning) return;
    setScanStageIndex(0);
    const interval = setInterval(() => {
      setScanStageIndex((prev) => {
        if (prev < SCAN_STAGES.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 650);
    return () => clearInterval(interval);
  }, [isScanning]);

  const loadScannedProfiles = async () => {
    try {
      const list = await fetchProfiles();
      setProfilesList(list);
      if (list.length > 0 && !selectedProfileId) {
        selectProfile(list[0].id);
      }
    } catch (err: any) {
      console.warn('Could not load scanned profiles list:', err);
    }
  };

  const selectProfile = async (id: string) => {
    setSelectedProfileId(id);
    setError(null);
    setSelectedPeriod(null);
    setSelectedCategory(null);
    try {
      const summary = await fetchProfileSummary(id);
      setProfileSummary(summary);
      setAudit(summary.retrieval_audit || null);

      if (onProfileStateChange) {
        onProfileStateChange(
          summary.profile,
          summary.aggregates,
          summary.recent_observations,
          summary.events,
          false
        );
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to load profile dossier.');
    }
  };

  const handleStartScan = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const clean = inputUrl.trim();
    if (!clean) {
      setError('Please provide a public social profile URL or handle.');
      return;
    }

    setError(null);
    setIsScanning(true);
    setSelectedPeriod(null);
    setSelectedCategory(null);

    if (onProfileStateChange) {
      onProfileStateChange(null, null, [], [], true);
    }

    try {
      const summary = await scanProfile(clean, scanLimit, demoFallback, enableJev);
      await loadScannedProfiles();
      setSelectedProfileId(summary.profile.id);

      // Build consolidated profile summary from scan result
      const fullSummary: ProfileSummaryResponse = {
        profile: summary.profile,
        aggregates: summary.aggregates,
        recent_observations: summary.dossier?.recent_observations || [],
        events: summary.events_detected,
        retrieval_audit: summary.retrieval_audit,
        activity_risk: summary.aggregates.profile_activity_risk,
        representative_evidence: summary.dossier?.representative_evidence || [],
      };
      setProfileSummary(fullSummary);
      setAudit(summary.retrieval_audit || null);

      if (onProfileStateChange) {
        onProfileStateChange(
          summary.profile,
          summary.aggregates,
          fullSummary.recent_observations,
          summary.events_detected,
          false
        );
      }
    } catch (err: any) {
      setError(err?.message || 'Historical scan failed. Check URL format and network access.');
    } finally {
      setIsScanning(false);
    }
  };

  const handleRunComparison = async () => {
    if (!selectedProfileId || !compareTargetId) {
      return;
    }
    setIsComparing(true);
    try {
      const res = await compareProfiles(selectedProfileId, compareTargetId);
      setComparisonResult(res);
    } catch (err: any) {
      setError(err?.message || 'Comparison failed.');
    } finally {
      setIsComparing(false);
    }
  };

  // Filtered evidence observations
  const filteredObservations = useMemo(() => {
    if (!profileSummary?.recent_observations) return [];
    return profileSummary.recent_observations.filter((obs) => {
      // Risk filter
      if (riskFilter !== 'ALL' && obs.final_risk_level !== riskFilter) {
        return false;
      }
      // Type filter
      if (typeFilter !== 'ALL' && obs.activity_type !== typeFilter) {
        return false;
      }
      // Language filter
      if (languageFilter !== 'ALL' && obs.language !== languageFilter) {
        return false;
      }
      // Temporal period filter
      if (selectedPeriod && !obs.timestamp.startsWith(selectedPeriod)) {
        return false;
      }
      // Category filter
      if (selectedCategory) {
        const catScore = obs.scores[selectedCategory] || 0;
        if (catScore < 0.45) return false;
      }
      return true;
    });
  }, [profileSummary, riskFilter, typeFilter, languageFilter, selectedPeriod, selectedCategory]);

  const riskBadgeStyles = (risk?: string) => {
    switch (risk) {
      case 'HIGH':
      case 'High Risk':
        return 'bg-crimson/20 border-crimson/50 text-rose-300 shadow-[0_0_12px_rgba(244,63,94,0.3)]';
      case 'ELEVATED':
        return 'bg-amber-500/20 border-amber-500/50 text-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.3)]';
      case 'MODERATE':
      case 'Review':
        return 'bg-amber-500/15 border-amber-500/40 text-amber-200';
      case 'LOW':
      case 'Safe':
      default:
        return 'bg-cyan-500/15 border-cyan-500/40 text-cyan-200 shadow-[0_0_10px_rgba(6,182,212,0.25)]';
    }
  };

  return (
    <div className="absolute inset-0 z-30 pointer-events-none flex flex-col justify-between p-4 md:p-6 overflow-hidden">
      {/* ── TOP HEADER & FORENSIC INVESTIGATION BANNER ── */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-3 pointer-events-auto bg-[#0a0d14]/90 backdrop-blur-md border border-white/10 p-3.5 md:p-4 rounded-xl shadow-2xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-mono text-lg shadow-[0_0_15px_rgba(6,182,212,0.3)]">
            📁
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono tracking-widest text-cyan-400 bg-cyan-950/70 border border-cyan-500/30 px-2 py-0.5 rounded">
                CHAMBER 08
              </span>
              <h1 className="text-base md:text-lg font-bold font-mono tracking-wider text-slate-100">
                HISTORICAL PROFILE INTELLIGENCE SCANNER
              </h1>
            </div>
            <p className="text-[11px] font-mono text-slate-400">
              Single-shot forensic investigation of publicly available message history. Zero continuous surveillance.
            </p>
          </div>
        </div>

        {/* Action Controls & Navigation Shortcuts */}
        <div className="flex items-center gap-2 self-end md:self-auto">
          {profilesList.length > 1 && (
            <button
              onClick={() => {
                setShowComparisonModal(true);
                const other = profilesList.find((p) => p.id !== selectedProfileId);
                if (other) setCompareTargetId(other.id);
              }}
              className="px-3 py-1.5 rounded-lg bg-indigo-950/70 border border-indigo-500/40 hover:border-indigo-400 text-indigo-300 font-mono text-xs flex items-center gap-1.5 transition-all shadow-[0_0_12px_rgba(99,102,241,0.25)]"
            >
              <span>⚖️</span>
              <span>COMPARE DOSSIERS</span>
            </button>
          )}

          {onGoToSocial && (
            <button
              onClick={onGoToSocial}
              className="px-3 py-1.5 rounded-lg bg-rose-950/50 border border-rose-500/30 hover:border-rose-400 text-rose-300 font-mono text-xs flex items-center gap-1.5 transition-all"
            >
              <span>🌐</span>
              <span>SOCIAL LINKS</span>
            </button>
          )}
        </div>
      </header>

      {/* ── CORE WORKSPACE: INPUT OR DOSSIER VIEW ── */}
      <div className="flex-1 flex flex-col my-3 overflow-hidden pointer-events-auto">
        {/* INPUT DOCK (ALWAYS VISIBLE AT TOP) */}
        <div className="bg-[#0b0e16]/95 border border-white/10 rounded-xl p-3.5 mb-3 backdrop-blur-md shadow-xl">
          <form onSubmit={handleStartScan} className="flex flex-col lg:flex-row gap-3">
            <div className="flex-1 relative flex items-center">
              <span className="absolute left-3.5 text-slate-500 font-mono text-xs pointer-events-none">
                URL / HANDLE:
              </span>
              <input
                type="text"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="reddit.com/user/username, youtube.com/@channel, or twitter.com/handle"
                className="w-full bg-[#05070c] border border-white/15 focus:border-cyan-400 rounded-lg pl-32 pr-4 py-2.5 text-sm font-mono text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition-all"
                disabled={isScanning}
              />
            </div>

            <div className="flex items-center gap-2">
              {/* Presets dropdown / pills */}
              <div className="hidden sm:flex items-center gap-1.5">
                {PRESET_PROFILES.map((preset) => (
                  <button
                    key={preset.url}
                    type="button"
                    onClick={() => setInputUrl(preset.url)}
                    className="px-2.5 py-2 rounded-lg bg-slate-900 border border-white/10 hover:border-cyan-500/40 text-[11px] font-mono text-slate-300 hover:text-cyan-300 transition-all whitespace-nowrap"
                    title={preset.url}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              {/* Limit selector */}
              <div className="flex items-center gap-1 bg-[#05070c] border border-white/15 rounded-lg px-2 py-1 text-xs font-mono text-slate-300">
                <span className="text-slate-500 text-[10px]">ITEMS:</span>
                <select
                  value={scanLimit}
                  onChange={(e) => setScanLimit(Number(e.target.value))}
                  className="bg-transparent text-cyan-300 focus:outline-none cursor-pointer"
                  disabled={isScanning}
                >
                  <option value={10}>10</option>
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                </select>
              </div>

              {/* Submit button */}
              <button
                type="submit"
                disabled={isScanning}
                className="px-5 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-mono font-semibold text-xs tracking-wider uppercase transition-all shadow-[0_0_15px_rgba(6,182,212,0.4)] flex items-center gap-2 whitespace-nowrap"
              >
                {isScanning ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>ANALYZING...</span>
                  </>
                ) : (
                  <>
                    <span>🔍</span>
                    <span>INVESTIGATE PROFILE</span>
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Quick options and past scans list */}
          <div className="mt-2.5 pt-2 border-t border-white/[0.06] flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400 gap-2">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableJev}
                  onChange={(e) => setEnableJev(e.target.checked)}
                  className="accent-cyan-400 rounded"
                />
                <span>Contextual Moderation (JEV)</span>
              </label>

              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={demoFallback}
                  onChange={(e) => setDemoFallback(e.target.checked)}
                  className="accent-cyan-400 rounded"
                />
                <span>Curated Multilingual Demo Fallback</span>
              </label>
            </div>

            {/* Previously Scanned Switcher */}
            {profilesList.length > 0 && (
              <div className="flex items-center gap-1.5">
                <span className="text-slate-500 text-[10px]">PREVIOUS SCANS:</span>
                <div className="flex items-center gap-1 overflow-x-auto max-w-md">
                  {profilesList.slice(0, 5).map((p) => (
                    <button
                      key={p.id}
                      onClick={() => selectProfile(p.id)}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono transition-all ${
                        selectedProfileId === p.id
                          ? 'bg-cyan-500/25 border border-cyan-400/50 text-cyan-200'
                          : 'bg-slate-900 border border-white/10 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {p.platform}: {p.handle}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ERROR NOTIFICATION */}
        {error && (
          <div className="mb-3 p-3 rounded-lg bg-rose-950/80 border border-rose-500/50 text-rose-200 font-mono text-xs flex items-center justify-between shadow-lg">
            <span>⚠️ {error}</span>
            <button onClick={() => setError(null)} className="text-rose-400 hover:text-rose-100 font-bold">
              ✕
            </button>
          </div>
        )}

        {/* SCAN PROGRESS TERMINAL */}
        <AnimatePresence>
          {isScanning && (
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="p-5 rounded-xl bg-[#07090f]/95 border border-cyan-500/40 font-mono shadow-[0_0_25px_rgba(6,182,212,0.25)] my-auto max-w-xl mx-auto w-full text-center"
            >
              <div className="inline-block p-3 rounded-full bg-cyan-950/60 border border-cyan-500/30 mb-3 animate-pulse">
                <span className="text-2xl">⚡</span>
              </div>
              <h3 className="text-sm font-bold text-cyan-300 tracking-wider mb-3">
                {SCAN_STAGES[scanStageIndex]}
              </h3>
              <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-white/10 mb-4">
                <motion.div
                  className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                  initial={{ width: '15%' }}
                  animate={{ width: `${((scanStageIndex + 1) / SCAN_STAGES.length) * 100}%` }}
                  transition={{ duration: 0.4 }}
                />
              </div>
              <p className="text-[11px] text-slate-400">
                Extracting available public interactions • Removing duplicate items • Executing frozen MuRIL pipeline
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── PROFILE INTELLIGENCE DOSSIER REPORT ── */}
        {!isScanning && profileSummary && (
          <div className="flex-1 overflow-y-auto space-y-3 pr-1 pb-4 custom-scrollbar">
            {/* 1. DOSSIER PASSPORT & AUDIT CARD */}
            <div className="p-4 rounded-xl bg-[#0a0d15]/90 border border-white/10 backdrop-blur-md shadow-xl flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
              <div className="flex items-center gap-3.5">
                <div className="w-12 h-12 rounded-xl bg-slate-900 border border-white/15 flex items-center justify-center text-2xl shadow-inner">
                  {profileSummary.profile.platform.toLowerCase() === 'reddit' ? '🤖' :
                   profileSummary.profile.platform.toLowerCase() === 'youtube' ? '📺' : '🐦'}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono uppercase px-2 py-0.5 rounded bg-slate-800 border border-white/10 text-slate-300">
                      {profileSummary.profile.platform}
                    </span>
                    <h2 className="text-base font-bold font-mono text-slate-100">
                      {profileSummary.profile.handle}
                    </h2>
                  </div>
                  <a
                    href={profileSummary.profile.profile_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[11px] font-mono text-cyan-400/80 hover:text-cyan-300 hover:underline flex items-center gap-1 mt-0.5"
                  >
                    <span>{profileSummary.profile.profile_url}</span>
                    <span>↗</span>
                  </a>
                </div>
              </div>

              {/* Risk Level Badge & Real vs Demo Tag */}
              <div className="flex flex-wrap items-center gap-2.5">
                {audit?.is_demo ? (
                  <div className="px-3 py-1 rounded-lg bg-amber-500/20 border border-amber-500/50 text-amber-300 font-mono text-xs font-bold tracking-wider flex items-center gap-1.5 shadow-[0_0_12px_rgba(245,158,11,0.25)]">
                    <span>⚠️</span>
                    <span>DEMO DATASET</span>
                  </div>
                ) : (
                  <div className="px-3 py-1 rounded-lg bg-emerald-500/20 border border-emerald-500/50 text-emerald-300 font-mono text-xs font-bold tracking-wider flex items-center gap-1.5 shadow-[0_0_12px_rgba(16,185,129,0.25)]">
                    <span>✓</span>
                    <span>REAL PLATFORM DATA</span>
                  </div>
                )}

                <div
                  className={`px-3.5 py-1.5 rounded-lg border font-mono text-xs font-bold tracking-wider flex flex-col items-end ${riskBadgeStyles(
                    profileSummary.activity_risk || profileSummary.aggregates.profile_activity_risk
                  )}`}
                >
                  <span className="text-[9px] opacity-75 uppercase tracking-widest">
                    PROFILE ACTIVITY RISK
                  </span>
                  <span className="text-sm">
                    {profileSummary.activity_risk || profileSummary.aggregates.profile_activity_risk || 'LOW'}
                  </span>
                </div>
              </div>
            </div>

            {/* Mandatory Objective Disclaimer */}
            <div className="px-3 py-2 rounded-lg bg-[#07090f]/80 border border-white/[0.08] text-[11px] font-mono text-slate-400 flex items-center gap-2">
              <span className="text-cyan-400 font-bold">ℹ️ DEFINITION:</span>
              <span>
                Risk level reflects classified public activity only. Does not characterize personal morality, intent, psychological state, or future behavior.
              </span>
            </div>

            {/* 2. OBSERVATION WINDOW & INGESTION AUDIT */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <div className="p-3 rounded-lg bg-[#090c13]/80 border border-white/10 font-mono">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block">EARLIEST OBSERVED</span>
                <span className="text-xs font-semibold text-slate-200">
                  {audit?.earliest_timestamp ? audit.earliest_timestamp.slice(0, 10) : 'N/A'}
                </span>
              </div>

              <div className="p-3 rounded-lg bg-[#090c13]/80 border border-white/10 font-mono">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block">LATEST OBSERVED</span>
                <span className="text-xs font-semibold text-slate-200">
                  {audit?.latest_timestamp ? audit.latest_timestamp.slice(0, 10) : 'N/A'}
                </span>
              </div>

              <div className="p-3 rounded-lg bg-[#090c13]/80 border border-white/10 font-mono">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block">INGESTION AUDIT</span>
                <span className="text-xs font-semibold text-cyan-300">
                  {audit ? `${audit.retrieved_count} / ${audit.requested_limit} ITEMS` : `${profileSummary.aggregates.total_observations} ITEMS`}
                </span>
              </div>

              <div className="p-3 rounded-lg bg-[#090c13]/80 border border-white/10 font-mono">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block">LIMITATION AUDIT</span>
                <span className="text-xs font-semibold text-slate-300 truncate block" title={audit?.retrieval_limitations || 'None'}>
                  {audit?.rate_limit_encountered ? 'API RATE-LIMITED' : 'STANDARD QUOTA'}
                </span>
              </div>
            </div>

            {/* 3. ACTIVITY BREAKDOWN & TOXICITY RATES */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {/* Activity Summary by Type */}
              <div className="p-4 rounded-xl bg-[#090c14]/90 border border-white/10 font-mono">
                <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center justify-between">
                  <span>ACTIVITY BY TYPE</span>
                  <span className="text-[10px] text-slate-500">OBSERVED VOLUME</span>
                </h3>
                <div className="space-y-2 text-xs">
                  {['post', 'comment', 'reply'].map((t) => {
                    const data = profileSummary.aggregates.activity_type_breakdown?.[t] || {
                      total: 0,
                      toxicity_rate: 0,
                      high_risk: 0,
                    };
                    return (
                      <div key={t} className="p-2 rounded bg-slate-950/60 border border-white/[0.06] flex items-center justify-between">
                        <span className="uppercase text-slate-400 font-semibold">{t}s</span>
                        <div className="flex items-center gap-3">
                          <span className="text-slate-300">{data.total} items</span>
                          <span className={`text-[11px] font-bold ${data.toxicity_rate > 30 ? 'text-rose-400' : 'text-cyan-400'}`}>
                            {data.toxicity_rate}% TOXIC
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Toxicity Metric Rates */}
              <div className="p-4 rounded-xl bg-[#090c14]/90 border border-white/10 font-mono">
                <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
                  TOXICITY SUMMARY MATRIX
                </h3>
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="p-2 rounded bg-slate-950/60 border border-white/[0.06]">
                    <span className="text-[10px] text-slate-500 block uppercase">AVG TOXICITY</span>
                    <span className="text-base font-bold text-cyan-300">
                      {profileSummary.aggregates.average_toxicity_score !== undefined
                        ? (profileSummary.aggregates.average_toxicity_score * 100).toFixed(1)
                        : (profileSummary.aggregates.average_scores?.toxic * 100 || 0).toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-950/60 border border-white/[0.06]">
                    <span className="text-[10px] text-slate-500 block uppercase">TOXICITY RATE</span>
                    <span className="text-base font-bold text-amber-300">
                      {profileSummary.aggregates.toxicity_rate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-950/60 border border-white/[0.06]">
                    <span className="text-[10px] text-slate-500 block uppercase">HIGH RISK RATE</span>
                    <span className="text-base font-bold text-rose-400">
                      {profileSummary.aggregates.high_risk_rate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-950/60 border border-white/[0.06]">
                    <span className="text-[10px] text-slate-500 block uppercase">SAFE RATE</span>
                    <span className="text-base font-bold text-cyan-400">
                      {profileSummary.aggregates.safe_rate !== undefined
                        ? profileSummary.aggregates.safe_rate.toFixed(1)
                        : (
                            (profileSummary.aggregates.safe_count / Math.max(profileSummary.aggregates.total_observations, 1)) *
                            100
                          ).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Language Distribution Breakdown */}
              <div className="p-4 rounded-xl bg-[#090c14]/90 border border-white/10 font-mono">
                <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center justify-between">
                  <span>LANGUAGE DISTRIBUTION</span>
                  <span className="text-[10px] text-slate-500">LEXICAL / SCRIPT</span>
                </h3>
                <div className="space-y-1.5 text-xs">
                  {profileSummary.aggregates.language_distribution &&
                    Object.entries(profileSummary.aggregates.language_distribution).map(([lang, count]) => {
                      const pct = profileSummary.aggregates.language_percentages?.[lang] ||
                        ((count / Math.max(profileSummary.aggregates.total_observations, 1)) * 100);
                      return (
                        <div key={lang} className="p-1.5 rounded bg-slate-950/60 border border-white/[0.06] flex items-center justify-between">
                          <span className="text-slate-300">{lang}</span>
                          <span className="text-cyan-300 font-semibold">{pct.toFixed(1)}% ({count})</span>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>

            {/* 4. HISTORICAL TREND TIMELINE */}
            {profileSummary.aggregates.historical_timeline && profileSummary.aggregates.historical_timeline.length > 0 && (
              <div className="p-4 rounded-xl bg-[#090c14]/90 border border-white/10 font-mono">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                      HISTORICAL ACTIVITY TIMELINE
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      Period changes in observed activity volume, toxicity rate, and dominant category.
                    </p>
                  </div>
                  {selectedPeriod && (
                    <button
                      onClick={() => setSelectedPeriod(null)}
                      className="text-[10px] text-cyan-400 hover:underline"
                    >
                      RESET PERIOD FILTER
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                  {profileSummary.aggregates.historical_timeline.map((stat) => {
                    const isSelected = selectedPeriod === stat.period;
                    return (
                      <button
                        key={stat.period}
                        onClick={() => setSelectedPeriod(isSelected ? null : stat.period)}
                        className={`p-2.5 rounded-lg border text-left transition-all ${
                          isSelected
                            ? 'bg-cyan-950/60 border-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.3)]'
                            : 'bg-slate-950/60 border-white/[0.08] hover:border-white/25'
                        }`}
                      >
                        <span className="text-xs font-bold text-slate-200 block">{stat.period}</span>
                        <div className="text-[10px] text-slate-400 mt-1 space-y-0.5">
                          <div>Vol: <span className="text-slate-200">{stat.volume}</span></div>
                          <div>Tox: <span className={stat.toxicity_rate > 35 ? 'text-rose-400 font-bold' : 'text-cyan-400'}>{stat.toxicity_rate}%</span></div>
                          <div className="truncate text-slate-500 uppercase">Dom: {stat.dominant_category}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 5. SIX CATEGORIES MATRIX & BEHAVIORAL PATTERNS */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {/* Category Breakdown Matrix */}
              <div className="p-4 rounded-xl bg-[#090c14]/90 border border-white/10 font-mono">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    CATEGORY FREQUENCY MATRIX
                  </h3>
                  {selectedCategory && (
                    <button onClick={() => setSelectedCategory(null)} className="text-[10px] text-cyan-400 hover:underline">
                      CLEAR FILTER
                    </button>
                  )}
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'].map((catKey) => {
                    const avg = profileSummary.aggregates.average_scores[catKey] || 0;
                    const flags = profileSummary.aggregates.category_flag_counts[catKey] || 0;
                    const isSelected = selectedCategory === catKey;
                    const isElevated = flags > 0 || avg >= 0.45;

                    return (
                      <button
                        key={catKey}
                        onClick={() => setSelectedCategory(isSelected ? null : catKey)}
                        className={`p-2.5 rounded-lg border text-left transition-all ${
                          isSelected
                            ? 'bg-rose-950/60 border-rose-400 shadow-[0_0_12px_rgba(244,63,94,0.3)]'
                            : isElevated
                            ? 'bg-slate-950/60 border-rose-500/30'
                            : 'bg-slate-950/60 border-white/[0.08]'
                        }`}
                      >
                        <span className="text-[10px] text-slate-400 uppercase block truncate">
                          {catKey.replace('_', ' ')}
                        </span>
                        <div className="flex items-baseline justify-between mt-1">
                          <span className="text-sm font-bold text-slate-200">
                            {(avg * 100).toFixed(0)}%
                          </span>
                          <span className={`text-[10px] font-semibold ${flags > 0 ? 'text-rose-400' : 'text-slate-500'}`}>
                            {flags} flagged
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Behavioral Pattern Findings */}
              <div className="p-4 rounded-xl bg-[#090c14]/90 border border-white/10 font-mono">
                <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
                  OBSERVABLE BEHAVIORAL PATTERNS
                </h3>
                {profileSummary.events.length === 0 ? (
                  <div className="p-4 rounded-lg bg-slate-950/60 border border-white/[0.06] text-xs text-slate-400">
                    No recurring or concentrated elevated toxicity patterns detected in the observed dataset.
                  </div>
                ) : (
                  <div className="space-y-2 max-h-48 overflow-y-auto pr-1 custom-scrollbar">
                    {profileSummary.events.map((ev) => (
                      <div
                        key={ev.id}
                        className="p-2.5 rounded-lg bg-slate-950/80 border border-white/[0.08] flex items-start gap-2.5"
                      >
                        <span
                          className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                            ev.severity === 'CRITICAL'
                              ? 'bg-crimson/20 text-rose-300 border border-crimson/40'
                              : ev.severity === 'HIGH'
                              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                              : 'bg-cyan-500/15 text-cyan-200 border border-cyan-500/30'
                          }`}
                        >
                          {ev.severity}
                        </span>
                        <div className="flex-1">
                          <h4 className="text-xs font-semibold text-slate-200">{ev.title}</h4>
                          <p className="text-[11px] text-slate-400 mt-0.5">{ev.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* 6. FORENSIC EVIDENCE STREAM */}
            <div className="p-4 rounded-xl bg-[#090c14]/90 border border-white/10 font-mono">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                    FORENSIC EVIDENCE STREAM ({filteredObservations.length} ITEMS)
                  </h3>
                  {(selectedPeriod || selectedCategory || riskFilter !== 'ALL' || typeFilter !== 'ALL' || languageFilter !== 'ALL') && (
                    <span className="text-[10px] text-cyan-400 bg-cyan-950/50 border border-cyan-500/30 px-2 py-0.5 rounded">
                      FILTERS ACTIVE
                    </span>
                  )}
                </div>

                {/* Filter Controls */}
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <select
                    value={riskFilter}
                    onChange={(e) => setRiskFilter(e.target.value)}
                    className="bg-slate-950 border border-white/15 rounded px-2 py-1 text-slate-200"
                  >
                    <option value="ALL">All Risk Levels</option>
                    <option value="High Risk">High Risk Only</option>
                    <option value="Review">Review Only</option>
                    <option value="Safe">Safe Only</option>
                  </select>

                  <select
                    value={typeFilter}
                    onChange={(e) => setTypeFilter(e.target.value)}
                    className="bg-slate-950 border border-white/15 rounded px-2 py-1 text-slate-200"
                  >
                    <option value="ALL">All Activity Types</option>
                    <option value="post">Posts</option>
                    <option value="comment">Comments</option>
                    <option value="reply">Replies</option>
                  </select>

                  <select
                    value={languageFilter}
                    onChange={(e) => setLanguageFilter(e.target.value)}
                    className="bg-slate-950 border border-white/15 rounded px-2 py-1 text-slate-200"
                  >
                    <option value="ALL">All Languages</option>
                    <option value="English">English</option>
                    <option value="Hindi">Hindi</option>
                    <option value="Hinglish">Hinglish</option>
                  </select>
                </div>
              </div>

              {/* Items List */}
              <div className="space-y-2 max-h-96 overflow-y-auto pr-1 custom-scrollbar">
                {filteredObservations.map((obs) => (
                  <div
                    key={obs.id}
                    onClick={() => setSelectedEvidenceItem(obs)}
                    className="p-3 rounded-lg bg-slate-950/70 border border-white/[0.08] hover:border-cyan-500/40 cursor-pointer transition-all flex flex-col md:flex-row md:items-start justify-between gap-2.5"
                  >
                    <div className="flex-1">
                      <div className="flex flex-wrap items-center gap-2 mb-1 text-[10px]">
                        <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 uppercase font-bold">
                          {obs.activity_type || 'COMMENT'}
                        </span>
                        <span className="px-1.5 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-500/20">
                          {obs.language}
                        </span>
                        <span className="text-slate-500">
                          {obs.timestamp.replace('T', ' ').slice(0, 19)}
                        </span>
                        {obs.permalink && (
                          <a
                            href={obs.permalink}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-cyan-400 hover:underline"
                          >
                            LINK ↗
                          </a>
                        )}
                      </div>
                      <p className="text-xs text-slate-200 line-clamp-2">{obs.text}</p>
                    </div>

                    <div className="flex items-center gap-2 self-end md:self-center">
                      <span
                        className={`px-2 py-1 rounded text-xs font-bold ${
                          obs.final_risk_level === 'High Risk'
                            ? 'bg-rose-950/80 text-rose-300 border border-rose-500/40'
                            : obs.final_risk_level === 'Review'
                            ? 'bg-amber-950/80 text-amber-300 border border-amber-500/40'
                            : 'bg-cyan-950/80 text-cyan-300 border border-cyan-500/40'
                        }`}
                      >
                        {obs.final_risk_level}
                      </span>
                      <span className="text-xs text-slate-400">
                        {(Math.max(...Object.values(obs.scores)) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── EVIDENCE INSPECTION MODAL ── */}
      {selectedEvidenceItem && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 pointer-events-auto">
          <div className="max-w-2xl w-full bg-[#0a0d15] border border-cyan-500/50 rounded-xl p-5 shadow-2xl font-mono text-slate-200">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 mb-4">
              <h3 className="text-sm font-bold text-cyan-300">
                EVIDENCE ITEM DETAILS // {selectedEvidenceItem.id}
              </h3>
              <button
                onClick={() => setSelectedEvidenceItem(null)}
                className="text-slate-400 hover:text-white font-bold"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs mb-4">
              <div className="p-3 rounded bg-slate-950 border border-white/10 text-sm">
                "{selectedEvidenceItem.text}"
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
                <div className="p-2 rounded bg-slate-900 border border-white/5">
                  <span className="text-slate-500 block uppercase">TYPE</span>
                  <span className="font-semibold text-slate-200 uppercase">{selectedEvidenceItem.activity_type || 'comment'}</span>
                </div>
                <div className="p-2 rounded bg-slate-900 border border-white/5">
                  <span className="text-slate-500 block uppercase">LANGUAGE</span>
                  <span className="font-semibold text-slate-200">{selectedEvidenceItem.language}</span>
                </div>
                <div className="p-2 rounded bg-slate-900 border border-white/5">
                  <span className="text-slate-500 block uppercase">RISK VERDICT</span>
                  <span className={`font-semibold ${selectedEvidenceItem.final_risk_level === 'High Risk' ? 'text-rose-400' : 'text-cyan-400'}`}>
                    {selectedEvidenceItem.final_risk_level}
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-900 border border-white/5">
                  <span className="text-slate-500 block uppercase">TIMESTAMP</span>
                  <span className="font-semibold text-slate-200">{selectedEvidenceItem.timestamp.slice(0, 19)}</span>
                </div>
              </div>

              {/* Per-Category Scores */}
              <div className="p-3 rounded bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-2">CATEGORY CLASSIFICATIONS</span>
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(selectedEvidenceItem.scores).map(([k, v]) => (
                    <div key={k} className="p-1.5 rounded bg-slate-900 flex justify-between items-center text-[11px]">
                      <span className="text-slate-400 uppercase">{k.slice(0, 7)}:</span>
                      <span className={`font-bold ${v >= 0.5 ? 'text-rose-400' : 'text-slate-300'}`}>
                        {(v * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {selectedEvidenceItem.final_explanation && (
                <div className="p-3 rounded bg-slate-950 border border-white/10 text-[11px] text-slate-300">
                  <span className="text-slate-500 block uppercase mb-1">MODEL EXPLANATION</span>
                  {selectedEvidenceItem.final_explanation}
                </div>
              )}
            </div>

            <div className="flex justify-end">
              <button
                onClick={() => setSelectedEvidenceItem(null)}
                className="px-4 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs font-semibold"
              >
                CLOSE
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── PROFILE COMPARISON MODAL ── */}
      {showComparisonModal && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4 pointer-events-auto">
          <div className="max-w-3xl w-full bg-[#0a0d16] border border-indigo-500/50 rounded-xl p-5 shadow-2xl font-mono text-slate-200">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 mb-4">
              <h3 className="text-sm font-bold text-indigo-300">
                PROFILE DOSSIER COMPARISON // OBSERVED ACTIVITY DELTA
              </h3>
              <button
                onClick={() => {
                  setShowComparisonModal(false);
                  setComparisonResult(null);
                }}
                className="text-slate-400 hover:text-white font-bold"
              >
                ✕
              </button>
            </div>

            <div className="flex items-center gap-3 mb-4">
              <span className="text-xs text-slate-400">COMPARE ACTIVE PROFILE WITH:</span>
              <select
                value={compareTargetId}
                onChange={(e) => setCompareTargetId(e.target.value)}
                className="bg-slate-950 border border-white/15 rounded px-3 py-1.5 text-xs text-indigo-200 focus:outline-none"
              >
                {profilesList
                  .filter((p) => p.id !== selectedProfileId)
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.platform}: {p.handle}
                    </option>
                  ))}
              </select>

              <button
                onClick={handleRunComparison}
                disabled={isComparing}
                className="px-4 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-all shadow-[0_0_12px_rgba(99,102,241,0.3)]"
              >
                {isComparing ? 'CALCULATING...' : 'COMPARE'}
              </button>
            </div>

            {comparisonResult && (
              <div className="space-y-4 text-xs">
                {/* Metric Delta Cards */}
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="p-3 rounded-lg bg-slate-950 border border-white/10">
                    <span className="text-[10px] text-slate-500 uppercase block">VOLUME DELTA</span>
                    <span className="text-base font-bold text-slate-100">
                      {comparisonResult.delta_metrics.volume_delta > 0 ? `+${comparisonResult.delta_metrics.volume_delta}` : comparisonResult.delta_metrics.volume_delta} ITEMS
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-white/10">
                    <span className="text-[10px] text-slate-500 uppercase block">TOXICITY RATE DELTA</span>
                    <span className={`text-base font-bold ${comparisonResult.delta_metrics.toxicity_rate_delta > 0 ? 'text-rose-400' : 'text-cyan-400'}`}>
                      {comparisonResult.delta_metrics.toxicity_rate_delta > 0 ? `+${comparisonResult.delta_metrics.toxicity_rate_delta}%` : `${comparisonResult.delta_metrics.toxicity_rate_delta}%`}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-white/10">
                    <span className="text-[10px] text-slate-500 uppercase block">HIGH-RISK RATE DELTA</span>
                    <span className={`text-base font-bold ${comparisonResult.delta_metrics.high_risk_rate_delta > 0 ? 'text-rose-400' : 'text-cyan-400'}`}>
                      {comparisonResult.delta_metrics.high_risk_rate_delta > 0 ? `+${comparisonResult.delta_metrics.high_risk_rate_delta}%` : `${comparisonResult.delta_metrics.high_risk_rate_delta}%`}
                    </span>
                  </div>
                </div>

                {/* Evidence Comparative Notes */}
                <div className="p-4 rounded-lg bg-slate-950 border border-white/10">
                  <h4 className="text-[10px] text-indigo-400 uppercase tracking-wider mb-2 font-bold">
                    COMPARATIVE FINDINGS (OBSERVED ACTIVITY ONLY)
                  </h4>
                  <ul className="space-y-1.5 text-slate-300">
                    {comparisonResult.comparative_notes.map((note, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-indigo-400">•</span>
                        <span>{note}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            <div className="flex justify-end mt-4">
              <button
                onClick={() => {
                  setShowComparisonModal(false);
                  setComparisonResult(null);
                }}
                className="px-4 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs font-semibold"
              >
                CLOSE
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
