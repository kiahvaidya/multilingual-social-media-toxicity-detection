import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SocialAnalysisResponse } from '../../types';
import { analyzeSocialUrl } from '../../api/client';

interface SocialAnalysisOverlayProps {
  onAddToDossier?: (item: any) => void;
  onGoToText?: () => void;
}

const PRESETS = [
  {
    platform: 'youtube',
    label: '📺 YouTube Video',
    url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
  },
  {
    platform: 'reddit',
    label: '🤖 Reddit Discussion',
    url: 'https://www.reddit.com/r/Python/comments/1g0v3z2/what_are_you_working_on/',
  },
  {
    platform: 'twitter',
    label: '🐦 Twitter / X Status',
    url: 'https://twitter.com/TwitterDev/status/1460323737035677698',
  },
];

export const SocialAnalysisOverlay: React.FC<SocialAnalysisOverlayProps> = ({ onAddToDossier, onGoToText }) => {
  const [url, setUrl] = useState(PRESETS[0].url);
  const [platform, setPlatform] = useState<string>('youtube');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SocialAnalysisResponse | null>(null);
  const [filter, setFilter] = useState<'ALL' | 'HIGH_RISK' | 'REVIEW' | 'SAFE'>('ALL');
  const [error, setError] = useState<string | null>(null);
  const [addedIds, setAddedIds] = useState<Record<string, boolean>>({});

  React.useEffect(() => {
    if (!result && !isAnalyzing) {
      handleRunAnalysis(PRESETS[0].url, PRESETS[0].platform);
    }
  }, []);

  const handleRunAnalysis = async (urlToRun?: string, platformToRun?: string) => {
    const targetUrl = urlToRun || url;
    if (!targetUrl.trim()) return;
    setIsAnalyzing(true);
    setError(null);

    try {
      const res = await analyzeSocialUrl(targetUrl, platformToRun || platform, 50, true, false);
      setResult(res);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch and analyze social stream.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAddComment = (comment: any) => {
    if (onAddToDossier) {
      onAddToDossier({
        id: comment.id,
        content: comment.text,
        type: 'SOCIAL',
        language: comment.language,
        risk_level: comment.risk_level,
        score: comment.confidence,
        jev_status: 'DETERMINISTIC_TRIAGE',
        status: 'PENDING',
        timestamp: new Date().toLocaleTimeString(),
      });
    }
    setAddedIds((prev) => ({ ...prev, [comment.id]: true }));
  };

  const getVerdictStyle = (risk: string) => {
    if (risk === 'High Risk') return { text: 'text-[#f43f5e]', border: 'border-[#f43f5e]/40', bg: 'bg-[#f43f5e]/15' };
    if (risk === 'Review') return { text: 'text-[#f59e0b]', border: 'border-[#f59e0b]/40', bg: 'bg-[#f59e0b]/15' };
    return { text: 'text-[#2dd4bf]', border: 'border-[#2dd4bf]/40', bg: 'bg-[#2dd4bf]/15' };
  };

  return (
    <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between p-6 md:p-12 pt-20 overflow-y-auto">
      {/* Top Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="font-mono text-xs tracking-widest text-[#f43f5e] uppercase">
            CHAMBER 07 // LIVE SOCIAL STREAM INGESTION
          </span>
          <h2 className="font-display font-bold text-2xl sm:text-4xl text-white mt-1">
            SOCIAL MEDIA COMMENT MODERATION
          </h2>
        </div>
      </div>

      {/* Mode Navigation Tabs */}
      <div className="flex flex-wrap items-center gap-2 mb-4 pointer-events-auto">
        {onGoToText && (
          <button
            onClick={onGoToText}
            className="px-4 py-2 rounded-xl bg-cyan-950/40 hover:bg-cyan-900/60 border border-cyan-500/40 hover:border-cyan-500/80 text-cyan-300 font-mono text-xs font-bold tracking-wider flex items-center gap-2 transition-all shadow-[0_0_15px_-3px_rgba(6,182,212,0.2)] hover:scale-[1.02]"
          >
            <span>← 📝 TAB: MULTILINGUAL TEXT</span>
          </button>
        )}
        <div className="px-4 py-2 rounded-xl bg-rose-950/80 border border-rose-500/60 text-rose-200 font-mono text-xs font-bold tracking-wider flex items-center gap-2 shadow-[0_0_15px_-3px_rgba(244,63,94,0.4)]">
          <span>🌐 TAB: SOCIAL MEDIA LINKS (ACTIVE)</span>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 my-auto pointer-events-auto">
        {/* Left Column: URL Terminal Form */}
        <div className="lg:col-span-5 bg-[#0a0c10]/90 border border-white/10 p-6 rounded-2xl backdrop-blur-md space-y-4">
          <div className="flex items-center justify-between">
            <label className="font-mono text-xs text-white/50 tracking-wider">
              SOURCE PLATFORM
            </label>
            <div className="flex items-center gap-1 bg-[#050608] p-1 rounded-lg border border-white/10 font-mono text-[10px]">
              {(['auto', 'youtube', 'reddit', 'twitter'] as const).map((p) => (
                <button
                  key={p}
                  onClick={() => setPlatform(p)}
                  className={`px-2.5 py-1 rounded transition ${
                    platform === p
                      ? 'bg-white/20 text-white font-bold border border-white/20'
                      : 'text-white/40 hover:text-white'
                  }`}
                >
                  {p === 'auto' ? '⚡ AUTO' : p.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="font-mono text-xs text-white/50 tracking-wider block mb-1.5">
              SOCIAL MEDIA URL / PERMALINK
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste YouTube, Reddit, or Twitter/X post URL..."
              className="w-full bg-[#050608] border border-white/10 rounded-xl p-3 text-sm font-sans text-white placeholder-white/30 focus:outline-none focus:border-[#f43f5e] focus:ring-1 focus:ring-[#f43f5e] transition-all"
            />
          </div>

          {/* Quick Presets */}
          <div>
            <span className="font-mono text-[11px] text-white/40 block mb-2">QUICK FORENSIC PRESETS:</span>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => {
                    setUrl(p.url);
                    setPlatform(p.platform);
                    handleRunAnalysis(p.url, p.platform);
                  }}
                  className="px-2.5 py-1 rounded-md text-[11px] font-mono border border-white/10 bg-white/5 hover:bg-white/15 text-white/70 hover:text-white transition-colors"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Trigger Action */}
          <div className="pt-2 flex items-center justify-between">
            <span className="font-mono text-[11px] text-white/40">
              ENGINE: MuRIL MULTI-LABEL
            </span>
            <button
              onClick={() => handleRunAnalysis()}
              disabled={isAnalyzing || !url.trim()}
              className="px-6 py-2.5 rounded-xl bg-[#f43f5e] hover:bg-[#e11d48] disabled:opacity-40 text-white font-mono font-bold text-xs tracking-wider transition-all flex items-center gap-2 shadow-[0_0_15px_-3px_rgba(244,63,94,0.4)]"
            >
              {isAnalyzing ? (
                <>
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-white border-t-transparent animate-spin" />
                  <span>SCRAPING & ANALYZING...</span>
                </>
              ) : (
                <span>INGEST & TRIAGE →</span>
              )}
            </button>
          </div>

          {error && (
            <p className="font-mono text-xs text-[#f43f5e] bg-[#f43f5e]/10 p-3 rounded-lg border border-[#f43f5e]/20">
              {error}
            </p>
          )}

          {result?.is_demo && result.message && (
            <div className="p-3 bg-[#f59e0b]/10 border border-[#f59e0b]/20 rounded-xl font-mono text-[11px] text-[#f59e0b] leading-relaxed">
              <span className="font-bold uppercase block mb-1">ℹ️ PREVIEW MODE ACTIVE:</span>
              {result.message}
            </div>
          )}
        </div>

        {/* Right Column: Ingested Stream Results */}
        <div className="lg:col-span-7 flex flex-col max-h-[65vh]">
          {result ? (
            <div className="bg-[#0a0c10]/90 border border-white/10 p-5 rounded-2xl backdrop-blur-md flex flex-col h-full space-y-4">
              {/* Telemetry Summary Header */}
              <div className="grid grid-cols-4 gap-2 font-mono text-center">
                <div className="p-2.5 rounded-xl bg-[#050608] border border-white/5">
                  <span className="text-[10px] text-white/40 uppercase block">TOTAL</span>
                  <span className="text-lg font-bold text-white">{result.total_comments}</span>
                </div>
                <div className="p-2.5 rounded-xl bg-[#f43f5e]/10 border border-[#f43f5e]/30">
                  <span className="text-[10px] text-[#f43f5e] uppercase block">HIGH RISK</span>
                  <span className="text-lg font-bold text-[#f43f5e]">{result.high_risk_count}</span>
                </div>
                <div className="p-2.5 rounded-xl bg-[#f59e0b]/10 border border-[#f59e0b]/30">
                  <span className="text-[10px] text-[#f59e0b] uppercase block">REVIEW</span>
                  <span className="text-lg font-bold text-[#f59e0b]">{result.review_count}</span>
                </div>
                <div className="p-2.5 rounded-xl bg-[#2dd4bf]/10 border border-[#2dd4bf]/30">
                  <span className="text-[10px] text-[#2dd4bf] uppercase block">SAFE</span>
                  <span className="text-lg font-bold text-[#2dd4bf]">{result.safe_count}</span>
                </div>
              </div>

              {/* Filter Tabs */}
              <div className="flex items-center justify-between border-b border-white/10 pb-2">
                <div className="flex gap-1 font-mono text-[11px]">
                  {(['ALL', 'HIGH_RISK', 'REVIEW', 'SAFE'] as const).map((f) => (
                    <button
                      key={f}
                      onClick={() => setFilter(f)}
                      className={`px-2.5 py-0.5 rounded-md transition ${
                        filter === f ? 'bg-white/20 text-white font-bold' : 'text-white/40 hover:text-white'
                      }`}
                    >
                      {f === 'ALL' ? 'ALL' : f === 'HIGH_RISK' ? 'HIGH RISK' : f}
                    </button>
                  ))}
                </div>
                <span className="font-mono text-[10px] text-white/40">
                  LATENCY: {result.latency_ms} ms
                </span>
              </div>

              {/* Scrollable Feed */}
              <div className="overflow-y-auto space-y-2.5 pr-1 flex-1">
                {result.comments
                  .filter((c) => {
                    if (filter === 'HIGH_RISK') return c.risk_level === 'High Risk';
                    if (filter === 'REVIEW') return c.risk_level === 'Review';
                    if (filter === 'SAFE') return c.risk_level === 'Safe';
                    return true;
                  })
                  .map((comment) => {
                    const isAdded = !!addedIds[comment.id];
                    const style = getVerdictStyle(comment.risk_level);
                    return (
                      <div
                        key={comment.id}
                        className={`p-3.5 rounded-xl border ${style.border} ${style.bg} transition-all space-y-2`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${style.border} ${style.text}`}>
                              {comment.badge_icon} {comment.risk_level}
                            </span>
                            <span className="font-mono text-[10px] text-white/50 bg-[#050608] px-1.5 py-0.5 rounded">
                              {comment.language}
                            </span>
                          </div>

                          <button
                            onClick={() => handleAddComment(comment)}
                            disabled={isAdded}
                            className={`px-2.5 py-1 rounded text-[10px] font-mono uppercase transition ${
                              isAdded
                                ? 'bg-[#2dd4bf]/20 text-[#2dd4bf] border border-[#2dd4bf]/40'
                                : 'bg-white/10 hover:bg-white/20 text-white border border-white/10'
                            }`}
                          >
                            {isAdded ? '✓ IN DOSSIER' : '+ DOSSIER'}
                          </button>
                        </div>

                        <p className="text-sm font-sans text-white/90 leading-relaxed">
                          {comment.text}
                        </p>

                        <div className="flex flex-wrap gap-1.5 pt-1">
                          {Object.entries(comment.is_toxic_flags).map(([label, flag]) => (
                            <span
                              key={label}
                              className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                                flag
                                  ? 'bg-[#f43f5e]/30 text-[#f43f5e] border border-[#f43f5e]/40 font-bold'
                                  : 'bg-black/30 text-white/30 border border-white/5'
                              }`}
                            >
                              {label}: {((comment.scores[label as keyof typeof comment.scores] || 0) * 100).toFixed(0)}%
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          ) : (
            <div className="h-full border border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center p-8 text-center text-white/30 font-mono text-xs space-y-2">
              <span className="text-2xl">🌐</span>
              <p>AWAITING SOCIAL MEDIA STREAM INGESTION</p>
              <p className="text-[11px] text-white/20 max-w-sm">
                Paste a YouTube video link, Reddit thread, or Twitter/X post to scrape and moderate top-level discourse.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Technical Indicator */}
      <div className="flex items-center justify-between font-mono text-[11px] text-white/40 border-t border-white/[0.08] pt-3 mt-4">
        <span>CONNECTOR STATUS: YOUTUBE V3 • REDDIT PRAW • TWITTER V2</span>
        <span>STREAM PARITY: 100% FROZEN THRESHOLDS</span>
      </div>
    </div>
  );
};
