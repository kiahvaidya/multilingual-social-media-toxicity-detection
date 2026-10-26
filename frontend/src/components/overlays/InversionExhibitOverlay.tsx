import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface InversionExhibitOverlayProps {
  mode: 'inversion' | 'emergent';
  onToggleMode: (mode: 'inversion' | 'emergent') => void;
}

export const InversionExhibitOverlay: React.FC<InversionExhibitOverlayProps> = ({
  mode,
  onToggleMode,
}) => {
  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-6 sm:p-12 z-20">
      {/* Top Title & Scientific Context */}
      <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full bg-forensic-beam animate-ping" />
            <span className="text-[10px] font-mono tracking-widest uppercase text-forensic-beam">
              CURATED DEMONSTRATION // QUALITATIVE CASE STUDY
            </span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-display uppercase tracking-tight text-white">
            {mode === 'inversion' ? 'Contextual Inversion' : 'Emergent Toxicity'}
          </h1>
          <p className="text-sm font-sans text-white/50 max-w-xl mt-1">
            {mode === 'inversion'
              ? 'Qualitative demonstration of how benign visual context de-escalates high lexical toxicity signals (sarcasm, banter, playful threats).'
              : 'Qualitative demonstration of how innocuous text weaponizes hateful visual symbols, generating cross-modal harm undetectable by NLP alone.'}
          </p>
          <div className="mt-2 text-[10px] font-mono text-white/30 uppercase tracking-wider">
            [INTERACTIVE CASE DEMONSTRATION — DISTINCT FROM AGGREGATE MODEL BENCHMARKS]
          </div>
        </div>

        {/* Experiment Selector Switch */}
        <div className="pointer-events-auto flex items-center bg-obsidian/90 border border-white/10 p-1 backdrop-blur-md">
          <button
            onClick={() => onToggleMode('inversion')}
            className={`px-4 py-2 text-xs font-mono uppercase tracking-wider transition-all duration-300 ${
              mode === 'inversion'
                ? 'bg-forensic-safe/20 text-forensic-safe border border-forensic-safe/40 shadow-[0_0_15px_rgba(45,212,191,0.2)]'
                : 'text-white/40 hover:text-white'
            }`}
          >
            Case A: Inversion (Defusal)
          </button>
          <button
            onClick={() => onToggleMode('emergent')}
            className={`px-4 py-2 text-xs font-mono uppercase tracking-wider transition-all duration-300 ${
              mode === 'emergent'
                ? 'bg-forensic-risk/20 text-forensic-risk border border-forensic-risk/40 shadow-[0_0_15px_rgba(244,63,94,0.2)]'
                : 'text-white/40 hover:text-white'
            }`}
          >
            Case B: Emergence (Weaponization)
          </button>
        </div>
      </div>

      {/* Center 3D Space is free for the ContextualInversionExhibit 3D geometry */}

      {/* Bottom Telemetry & Vector Interaction Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pointer-events-auto max-w-5xl">
        {/* Step 1: Unimodal NLP Signal */}
        <div className="bg-obsidian/90 border border-white/10 p-4 backdrop-blur-md">
          <div className="text-[10px] font-mono uppercase tracking-wider text-white/40 mb-1">
            Demonstration Input 01 // Lexical Stream (MuRIL)
          </div>
          <div className="font-mono text-sm text-white/90 mb-2 truncate">
            {mode === 'inversion' ? '"I swear I will strangle you"' : '"Just look at them thriving"'}
          </div>
          <div className="flex items-center justify-between text-xs font-mono border-t border-white/5 pt-2">
            <span className="text-white/50">Unimodal Text Score:</span>
            <span
              className={
                mode === 'inversion' ? 'text-forensic-risk font-bold' : 'text-forensic-safe font-bold'
              }
            >
              {mode === 'inversion' ? '0.892 (HIGH RISK)' : '0.114 (SAFE)'}
            </span>
          </div>
        </div>

        {/* Step 2: Visual Semantics */}
        <div className="bg-obsidian/90 border border-white/10 p-4 backdrop-blur-md">
          <div className="text-[10px] font-mono uppercase tracking-wider text-white/40 mb-1">
            Demonstration Input 02 // Visual Context (CLIP ViT-B/32)
          </div>
          <div className="font-mono text-sm text-white/90 mb-2 truncate">
            {mode === 'inversion' ? 'Playful golden retriever pup' : 'Hate group insignia / targeted slur'}
          </div>
          <div className="flex items-center justify-between text-xs font-mono border-t border-white/5 pt-2">
            <span className="text-white/50">Visual Context:</span>
            <span className="text-forensic-beam font-bold">
              {mode === 'inversion' ? 'Affectionate / Benign' : 'Discriminatory Motif'}
            </span>
          </div>
        </div>

        {/* Step 3: 4-Way Cross-Modal Fusion & Verdict */}
        <div
          className={`border p-4 backdrop-blur-md transition-colors duration-500 ${
            mode === 'inversion'
              ? 'bg-forensic-safe/10 border-forensic-safe/30'
              : 'bg-forensic-risk/10 border-forensic-risk/30'
          }`}
        >
          <div className="text-[10px] font-mono uppercase tracking-wider text-white/40 mb-1">
            Demonstration Result // 4-Way Fusion Adjudication
          </div>
          <div className="flex items-baseline justify-between mb-2">
            <span className="font-display text-2xl font-bold tracking-tight text-white">
              {mode === 'inversion' ? 'SAFE' : 'HIGH RISK'}
            </span>
            <span
              className={`font-mono text-xs font-bold ${
                mode === 'inversion' ? 'text-forensic-safe' : 'text-forensic-risk'
              }`}
            >
              {mode === 'inversion' ? 'DEMO SCORE: 0.142' : 'DEMO SCORE: 0.887'}
            </span>
          </div>
          <div className="text-[11px] font-mono text-white/60 border-t border-white/10 pt-2 leading-relaxed">
            {mode === 'inversion'
              ? '||t - v|| difference vector nullifies threat flags in this case study. Lexical false positive defused.'
              : 't ⊙ v cross-product magnifies latent hate cues in this case study. Pure text search evaded.'}
          </div>
        </div>
      </div>
    </div>
  );
};
