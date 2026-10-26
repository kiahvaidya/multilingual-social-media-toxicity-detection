import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TextAnalysisResult } from '../../types';
import { analyzeText } from '../../api/client';

interface TextAnalysisOverlayProps {
  onResult: (res: TextAnalysisResult) => void;
  result: TextAnalysisResult | null;
  isAnalyzing: boolean;
  setIsAnalyzing: (v: boolean) => void;
  onGoToSocial?: () => void;
}

const PRESETS = [
  {
    label: 'Hindi Targeted',
    text: 'तुझे शर्म नहीं आती क्या? देशद्रोही साले निकल यहाँ से',
    lang: 'Hindi',
  },
  {
    label: 'Hinglish Colloquial Banter',
    text: 'ye banda total chutiya hai but video bahut funny banata hai lol',
    lang: 'Hinglish',
  },
  {
    label: 'Severe English Hate',
    text: 'I despise your kind and hope your community is eradicated from existence.',
    lang: 'English',
  },
  {
    label: 'Benign Ambiguous',
    text: 'This policy is absolute garbage and killing the industry.',
    lang: 'English',
  },
];

export const TextAnalysisOverlay: React.FC<TextAnalysisOverlayProps> = ({
  onResult,
  result,
  isAnalyzing,
  setIsAnalyzing,
  onGoToSocial,
}) => {
  const [inputText, setInputText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const isSocialUrl = inputText.toLowerCase().includes('youtube.com') ||
    inputText.toLowerCase().includes('youtu.be') ||
    inputText.toLowerCase().includes('reddit.com') ||
    inputText.toLowerCase().includes('twitter.com') ||
    inputText.toLowerCase().includes('x.com');

  const handleRunAnalysis = async (textToRun?: string) => {
    const text = textToRun || inputText;
    if (!text.trim()) return;
    setIsAnalyzing(true);
    setError(null);

    try {
      const res = await analyzeText(text);
      onResult(res);
    } catch (err: any) {
      setError(err?.message || 'Text analysis failed. Ensure backend api.py is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getVerdictStyle = (risk: string) => {
    if (risk === 'High Risk') return { text: 'text-[#f43f5e]', border: 'border-[#f43f5e]/40', bg: 'bg-[#f43f5e]/10' };
    if (risk === 'Review') return { text: 'text-[#f59e0b]', border: 'border-[#f59e0b]/40', bg: 'bg-[#f59e0b]/10' };
    return { text: 'text-[#2dd4bf]', border: 'border-[#2dd4bf]/40', bg: 'bg-[#2dd4bf]/10' };
  };

  return (
    <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between p-8 md:p-14 pt-24 overflow-y-auto">
      {/* Top Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="font-mono text-xs tracking-widest text-[#38bdf8] uppercase">
            CHAMBER 02 // TYPOGRAPHIC & LINGUISTIC PIPELINE
          </span>
          <h2 className="font-display font-bold text-3xl sm:text-4xl text-white mt-1">
            MuRIL DEEP EMBEDDING CHAMBER
          </h2>
        </div>

        {onGoToSocial && (
          <button
            onClick={onGoToSocial}
            className="pointer-events-auto px-4 py-2 rounded-full border border-rose-500/40 bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 font-mono text-xs flex items-center gap-2 transition-all shadow-[0_0_15px_-3px_rgba(244,63,94,0.3)]"
          >
            <span>🌐 SOCIAL MEDIA LINKS (07)</span>
            <span>→</span>
          </button>
        )}
      </div>

      {/* Mode Navigation Tabs */}
      <div className="flex flex-wrap items-center gap-2 mb-4 pointer-events-auto">
        <div className="px-4 py-2 rounded-xl bg-cyan-950/80 border border-cyan-500/50 text-cyan-300 font-mono text-xs font-bold tracking-wider flex items-center gap-2 shadow-[0_0_15px_-3px_rgba(6,182,212,0.3)]">
          <span>📝 TAB: MULTILINGUAL TEXT (ACTIVE)</span>
        </div>
        {onGoToSocial && (
          <button
            onClick={onGoToSocial}
            className="px-4 py-2 rounded-xl bg-rose-950/40 hover:bg-rose-900/60 border border-rose-500/40 hover:border-rose-500/80 text-rose-300 font-mono text-xs font-bold tracking-wider flex items-center gap-2 transition-all shadow-[0_0_15px_-3px_rgba(244,63,94,0.3)] hover:scale-[1.02]"
          >
            <span>🌐 TAB: SOCIAL MEDIA LINKS (YOUTUBE / REDDIT / X)</span>
            <span className="text-rose-400">→</span>
          </button>
        )}
      </div>

      {/* Main Interactive Work Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 my-auto pointer-events-auto">
        {/* Input Terminal Column */}
        <div className="lg:col-span-5 bg-[#0a0c10]/85 border border-white/10 p-6 rounded-2xl backdrop-blur-md">
          {isSocialUrl && onGoToSocial && (
            <div className="mb-3 p-3 bg-rose-950/40 border border-rose-500/30 rounded-xl font-mono text-xs text-rose-300 flex items-center justify-between">
              <span>🔗 Social link detected!</span>
              <button
                onClick={onGoToSocial}
                className="px-2.5 py-1 bg-rose-500 text-black font-bold rounded hover:bg-rose-400 transition"
              >
                OPEN SOCIAL STREAM →
              </button>
            </div>
          )}
          <label className="font-mono text-xs text-white/50 tracking-wider block mb-2">
            RAW LINGUISTIC INPUT [ENGLISH / HINDI / HINGLISH]
          </label>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && inputText.trim() && !isAnalyzing) {
                e.preventDefault();
                handleRunAnalysis();
              }
            }}
            placeholder="Enter multilingual text or select forensic preset... (⌘/Ctrl + Enter to run)"
            rows={4}
            className="w-full bg-[#050608] border border-white/10 rounded-xl p-3.5 text-sm font-sans text-white placeholder-white/30 focus:outline-none focus:border-[#38bdf8] focus:ring-1 focus:ring-[#38bdf8] transition-all resize-none"
          />

          {/* Quick Presets */}
          <div className="mt-3 flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => {
                  setInputText(p.text);
                  handleRunAnalysis(p.text);
                }}
                className="px-2.5 py-1 rounded-md text-[11px] font-mono border border-white/10 bg-white/5 hover:bg-white/15 text-white/70 hover:text-white transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Action Trigger */}
          <div className="mt-4 flex items-center justify-between">
            <span className="font-mono text-[11px] text-white/40">
              ENGINES: MuRIL (768-D) + JEV
            </span>
            <div className="flex items-center gap-2">
              {inputText && (
                <button
                  onClick={() => {
                    setInputText('');
                    setError(null);
                  }}
                  className="px-3 py-2.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-white/60 hover:text-white font-mono text-xs transition-all"
                >
                  CLEAR
                </button>
              )}
              <button
                onClick={() => handleRunAnalysis()}
                disabled={isAnalyzing || !inputText.trim()}
                className="px-5 py-2.5 rounded-xl bg-[#38bdf8] hover:bg-[#0284c7] disabled:opacity-40 text-black font-mono font-semibold text-xs tracking-wider transition-all flex items-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <span className="w-3 h-3 rounded-full border-2 border-black border-t-transparent animate-spin" />
                    <span>EXTRACTING...</span>
                  </>
                ) : (
                  <span>RUN PIPELINE →</span>
                )}
              </button>
            </div>
          </div>

          {error && (
            <p className="mt-3 font-mono text-xs text-[#f43f5e] bg-[#f43f5e]/10 p-2.5 rounded-lg border border-[#f43f5e]/20">
              {error}
            </p>
          )}
        </div>

        {/* Dynamic Verdict & Dimension Column */}
        <div className="lg:col-span-7 flex flex-col justify-center">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key={result.timestamp || result.text}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="bg-[#0a0c10]/85 border border-white/10 p-6 rounded-2xl backdrop-blur-md"
              >
                {/* Large Editorial Verdict */}
                <div className="flex flex-wrap items-baseline justify-between gap-4 border-b border-white/10 pb-4 mb-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-[10px] text-white/40 tracking-widest uppercase">
                        FINAL ADJUDICATION
                      </span>
                      {result.language && (
                        <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10 font-mono text-[10px] text-[#38bdf8]">
                          {result.language}
                        </span>
                      )}
                    </div>
                    <h3 className={`font-display font-extrabold text-4xl sm:text-5xl ${getVerdictStyle(result.risk_level).text}`}>
                      {result.risk_level.toUpperCase()}
                    </h3>
                    {result.explanation && (
                      <p className="mt-1.5 text-xs text-white/60 font-sans">
                        {result.explanation}
                      </p>
                    )}
                  </div>

                  <div className="text-right font-mono text-xs">
                    <span className="text-white/40 block">CONFIDENCE</span>
                    <span className="text-lg font-bold text-white">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* 6 Dimensions Grid */}
                <div className="mb-4">
                  <span className="font-mono text-[10px] text-white/40 tracking-wider uppercase block mb-2">
                    6-WAY TOXICITY TENSOR SCORES
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-xs">
                    {Object.entries(result.scores).map(([key, val]) => {
                      const thresh = result.thresholds?.[key] ?? 0.5;
                      const isHigh = result.is_toxic_flags?.[key] ?? val >= thresh;
                      return (
                        <div
                          key={key}
                          className={`p-2 rounded-lg border ${
                            isHigh ? 'border-[#f43f5e]/40 bg-[#f43f5e]/10 text-[#f43f5e]' : 'border-white/5 bg-white/5 text-white/70'
                          }`}
                        >
                          <div className="text-[10px] text-white/40 uppercase">{key.replace('_', ' ')}</div>
                          <div className="font-bold text-sm">{(val * 100).toFixed(0)}%</div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Pipeline Latency Trace */}
                {result.pipeline_trace && result.pipeline_trace.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-2">
                    {result.pipeline_trace.map((step, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 rounded bg-white/[0.03] border border-white/[0.06] font-mono text-[10px] text-white/50"
                      >
                        {step.stage}: <strong className="text-white/80">{step.duration_ms}ms</strong>
                      </span>
                    ))}
                  </div>
                )}

                {/* JEV Contextual Layer Readout */}
                {result.jev?.invoked && result.jev.decision && (
                  <div className="p-3.5 rounded-xl border border-[#f59e0b]/30 bg-[#f59e0b]/10 text-xs font-sans text-white/80">
                    <span className="font-mono font-bold text-[#f59e0b] block mb-1">
                      ⚖️ JEV CONTEXTUAL ADJUDICATION LAYER:
                    </span>
                    <p className="leading-relaxed">
                      {result.jev.decision.reason}
                    </p>
                    <div className="mt-2 font-mono text-[10px] text-white/50 flex gap-4">
                      <span>INTENT: {result.jev.decision.intent}</span>
                      <span>HUMAN REVIEW: {result.jev.decision.human_review_required ? 'YES' : 'NO'}</span>
                    </div>
                  </div>
                )}
              </motion.div>
            ) : (
              <div className="p-8 border border-dashed border-white/10 rounded-2xl text-center text-white/40 font-mono text-xs">
                AWAITING LINGUISTIC INPUT IN CHAMBER...
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Bottom Technical Indicator */}
      <div className="flex items-center justify-between font-mono text-[11px] text-white/40 border-t border-white/[0.08] pt-4">
        <span>MURIL EMBEDDING SPACE: 768 DIMENSIONS</span>
        <span>RADAR GEOMETRY: PROJECTED 3D WIREFRAME</span>
      </div>
    </div>
  );
};
