import React from 'react';
import { motion } from 'framer-motion';

interface LandingOverlayProps {
  onEnter: () => void;
  onExploreMeme: () => void;
  onExploreSocial?: () => void;
  onExploreProfileWatch?: () => void;
}

export const LandingOverlay: React.FC<LandingOverlayProps> = ({
  onEnter,
  onExploreMeme,
  onExploreSocial,
  onExploreProfileWatch,
}) => {
  return (
    <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between p-8 md:p-14">
      {/* Top Left Tag */}
      <motion.div
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.0, delay: 0.2 }}
        className="pt-16"
      >
        <span className="font-mono text-xs tracking-widest text-[#38bdf8] uppercase border-l-2 border-[#38bdf8] pl-3 py-0.5">
          INVESTIGATION CHAMBER // 00
        </span>
      </motion.div>

      {/* Hero Editorial Centerpiece */}
      <motion.div
        initial={{ opacity: 0, y: 25 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.2, delay: 0.4 }}
        className="max-w-3xl"
      >
        <h1 className="font-display font-bold text-5xl sm:text-7xl md:text-8xl tracking-tight text-white leading-none mb-4 select-none">
          MULTIMODAL <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-white/80 to-white/30">
            TOXICITY
          </span>
        </h1>
        <p className="font-sans text-lg sm:text-2xl text-white/60 font-light tracking-wide max-w-xl mb-10">
          Context changes meaning. Words alone deceive. Memes weaponize intent.
        </p>

        {/* Minimal Editorial Triggers */}
        <div className="flex flex-wrap items-center gap-4 pointer-events-auto">
          <button
            onClick={onEnter}
            className="group px-7 py-3 rounded-full border border-white/20 bg-white/5 hover:bg-white text-white hover:text-black font-mono text-xs tracking-wider transition-all duration-300 flex items-center gap-3 backdrop-blur-md shadow-lg"
          >
            <span>ENTER INVESTIGATION</span>
            <span className="transform group-hover:translate-x-1 transition-transform">→</span>
          </button>

          <button
            onClick={onExploreMeme}
            className="px-6 py-3 rounded-full border border-transparent hover:border-white/10 text-white/60 hover:text-white font-mono text-xs tracking-wider transition-all"
          >
            [ LAUNCH MEME FUSION CORE ]
          </button>

          {onExploreSocial && (
            <button
              onClick={onExploreSocial}
              className="px-6 py-3 rounded-full border border-rose-500/40 bg-rose-950/30 hover:bg-rose-900/50 text-rose-300 font-mono text-xs tracking-wider transition-all flex items-center gap-2 backdrop-blur-md shadow-[0_0_15px_-3px_rgba(244,63,94,0.3)]"
            >
              <span>🌐 SOCIAL STREAM INGESTION</span>
              <span>→</span>
            </button>
          )}

          {onExploreProfileWatch && (
            <button
              onClick={onExploreProfileWatch}
              className="px-6 py-3 rounded-full border border-cyan-500/40 bg-cyan-950/30 hover:bg-cyan-900/50 text-cyan-300 font-mono text-xs tracking-wider transition-all flex items-center gap-2 backdrop-blur-md shadow-[0_0_15px_-3px_rgba(6,182,212,0.3)]"
            >
              <span>🛡️ PROFILE WATCH</span>
              <span>→</span>
            </button>
          )}
        </div>
      </motion.div>

      {/* Bottom Technical Telemetry Bar */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.0, delay: 0.8 }}
        className="flex flex-wrap items-center justify-between gap-4 font-mono text-[11px] text-white/40 border-t border-white/[0.06] pt-4"
      >
        <div className="flex items-center gap-6">
          <span>SPATIAL FIELD: 3,200 LINGUISTIC PARTICLES</span>
          <span className="hidden sm:inline">•</span>
          <span className="hidden sm:inline">DATASET QUARANTINE: MULTI3HATE HELD OUT</span>
        </div>
        <div className="flex items-center gap-4 text-[#38bdf8]">
          <span>INDIAN CONTEXT AWARENESS // MURIL + CLIP</span>
        </div>
      </motion.div>
    </div>
  );
};
