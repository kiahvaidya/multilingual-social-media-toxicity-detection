import React from 'react';
import { motion } from 'framer-motion';

interface OverviewOverlayProps {
  onGoToText: () => void;
  onGoToMeme: () => void;
  onGoToInversion: () => void;
  onGoToResearch: () => void;
  onGoToSocial?: () => void;
  onGoToProfileWatch?: () => void;
}

export const OverviewOverlay: React.FC<OverviewOverlayProps> = ({
  onGoToText,
  onGoToMeme,
  onGoToInversion,
  onGoToResearch,
  onGoToSocial,
  onGoToProfileWatch,
}) => {
  return (
    <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between p-8 md:p-14 pt-24">
      {/* Top Banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <span className="font-mono text-xs tracking-widest text-[#38bdf8] uppercase">
            CHAMBER 01 // COMMAND CENTER
          </span>
          <h2 className="font-display font-bold text-3xl sm:text-4xl text-white mt-1">
            FORENSIC TELEMETRY STREAM
          </h2>
        </div>
      </motion.div>

      {/* Center Left Narrative Annotation */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
        className="max-w-md pointer-events-auto bg-[#0a0c10]/75 border border-white/10 p-6 rounded-2xl backdrop-blur-md"
      >
        <span className="font-mono text-[10px] text-[#f59e0b] tracking-widest uppercase block mb-1">
          OPERATIONAL SUMMARY
        </span>
        <h3 className="font-display font-semibold text-lg text-white mb-2">
          Multilingual Ingestion & Assistive Triaging
        </h3>
        <p className="font-sans text-sm text-white/70 leading-relaxed mb-4">
          Incoming comments and memes are decomposed across MuRIL (Hindi/English text) and CLIP (visual context). High-ambiguity banter and slang are routed through the JEV context gate.
        </p>

        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-white/10 font-mono text-xs">
          <button
            onClick={onGoToText}
            className="p-2.5 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 text-white text-left transition-colors flex items-center justify-between"
          >
            <span>ANALYSIS [TEXT]</span>
            <span>→</span>
          </button>
          <button
            onClick={onGoToMeme}
            className="p-2.5 rounded-lg border border-[#38bdf8]/30 bg-[#38bdf8]/10 hover:bg-[#38bdf8]/20 text-[#38bdf8] text-left transition-colors flex items-center justify-between"
          >
            <span>ANALYSIS [MEME]</span>
            <span>→</span>
          </button>
          {onGoToSocial && (
            <button
              onClick={onGoToSocial}
              className="p-2.5 rounded-lg border border-rose-500/30 bg-rose-950/20 hover:bg-rose-900/40 text-rose-300 text-left transition-colors flex items-center justify-between"
            >
              <span>[🌐 SOCIAL LINKS]</span>
              <span>→</span>
            </button>
          )}
          {onGoToProfileWatch && (
            <button
              onClick={onGoToProfileWatch}
              className="p-2.5 rounded-lg border border-cyan-500/30 bg-cyan-950/20 hover:bg-cyan-900/40 text-cyan-300 text-left transition-colors flex items-center justify-between"
            >
              <span>[🛡️ PROFILE WATCH]</span>
              <span>→</span>
            </button>
          )}
        </div>
      </motion.div>

      {/* Bottom Action Grid */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex flex-wrap items-center justify-between gap-4 pointer-events-auto border-t border-white/[0.08] pt-4"
      >
        <div className="flex items-center gap-4">
          <button
            onClick={onGoToInversion}
            className="px-4 py-2 rounded-full border border-white/15 bg-white/5 hover:bg-white/15 text-white font-mono text-xs transition-colors"
          >
            [ LAUNCH CONTEXTUAL INVERSION LAB ]
          </button>
          <button
            onClick={onGoToResearch}
            className="px-4 py-2 rounded-full border border-white/15 bg-white/5 hover:bg-white/15 text-white/70 hover:text-white font-mono text-xs transition-colors"
          >
            [ VIEW FINAL BENCHMARK AUDIT ]
          </button>
        </div>

        <div className="font-mono text-xs text-white/40">
          SPATIAL NODES ACTIVE: 4 // LATENCY: 28MS
        </div>
      </motion.div>
    </div>
  );
};
