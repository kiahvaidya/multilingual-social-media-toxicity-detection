import React from 'react';
import { SceneMode } from '../3d/CameraController';

interface ForensicNavigationProps {
  currentMode: SceneMode;
  onSelectMode: (mode: SceneMode) => void;
  systemStatus?: string;
  deviceEngine?: string;
}

export const ForensicNavigation: React.FC<ForensicNavigationProps> = ({
  currentMode,
  onSelectMode,
  systemStatus = 'ONLINE',
  deviceEngine = 'MuRIL + CLIP + JEV',
}) => {
  const navItems: { mode: SceneMode; label: string; code: string }[] = [
    { mode: 'LANDING', label: 'FIELD', code: '00' },
    { mode: 'OVERVIEW', label: 'COMMAND', code: '01' },
    { mode: 'TEXT', label: 'TEXT CHAMBER', code: '02' },
    { mode: 'MEME', label: 'MEME FUSION', code: '03' },
    { mode: 'INVERSION', label: 'INVERSION LAB', code: '04' },
    { mode: 'QUEUE', label: 'DOSSIER', code: '05' },
    { mode: 'RESEARCH', label: 'RESEARCH', code: '06' },
    { mode: 'SOCIAL', label: '🌐 SOCIAL LINKS', code: '07' },
    { mode: 'PROFILE_WATCH', label: '📁 PROFILE DOSSIER', code: '08' },
  ];

  return (
    <header className="fixed top-0 left-0 w-full z-40 px-4 md:px-6 py-3.5 pointer-events-none flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] bg-gradient-to-b from-[#050608]/95 to-transparent backdrop-blur-md">
      {/* Brand Identity */}
      <div className="flex items-center gap-4 pointer-events-auto">
        <button
          onClick={() => onSelectMode('LANDING')}
          className="text-left group focus:outline-none"
        >
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#38bdf8] animate-pulse" />
            <h1 className="font-display font-bold text-sm tracking-widest text-[#f8fafc] group-hover:text-[#38bdf8] transition-colors">
              TOXISCOPE <span className="text-white/30 font-mono text-xs">// FORENSIC LAB</span>
            </h1>
          </div>
          <p className="font-mono text-[10px] text-white/40 tracking-wider">
            MULTIMODAL & MULTILINGUAL AI SECURITY
          </p>
        </button>
      </div>

      {/* Segmented Responsive Navigation */}
      <nav className="flex items-center gap-1 p-1 bg-[#0a0c10]/90 border border-white/10 rounded-full pointer-events-auto backdrop-blur-lg overflow-x-auto max-w-full">
        {navItems.map((item) => {
          const isActive = currentMode === item.mode;
          const isSocial = item.mode === 'SOCIAL';
          const isProfileWatch = item.mode === 'PROFILE_WATCH';
          return (
            <button
              key={item.mode}
              onClick={() => onSelectMode(item.mode)}
              className={`px-3 py-1 rounded-full text-xs font-mono transition-all duration-300 flex items-center gap-1.5 whitespace-nowrap ${
                isActive
                  ? isProfileWatch
                    ? 'bg-cyan-500/25 text-cyan-200 font-semibold border border-cyan-400/50 shadow-[0_0_15px_-3px_rgba(6,182,212,0.4)]'
                    : isSocial
                    ? 'bg-rose-500/25 text-rose-200 font-semibold border border-rose-500/50 shadow-[0_0_15px_-3px_rgba(244,63,94,0.4)]'
                    : 'bg-white/15 text-white font-medium border border-white/20 shadow-[0_0_15px_-3px_rgba(56,189,248,0.3)]'
                  : isProfileWatch
                  ? 'text-cyan-300/80 hover:text-cyan-200 hover:bg-cyan-950/40 border border-cyan-500/20'
                  : isSocial
                  ? 'text-rose-300/80 hover:text-rose-200 hover:bg-rose-950/40 border border-rose-500/20'
                  : 'text-white/50 hover:text-white/80 hover:bg-white/5'
              }`}
            >
              <span className={`text-[9px] ${isActive ? (isProfileWatch ? 'text-cyan-400' : isSocial ? 'text-rose-400' : 'text-[#38bdf8]') : 'text-white/30'}`}>
                {item.code}
              </span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Telemetry Indicator */}
      <div className="hidden sm:flex items-center gap-5 font-mono text-[11px] text-white/50 pointer-events-auto">
        <div className="flex items-center gap-1.5">
          <span className="text-white/30">CORE:</span>
          <span className="text-[#38bdf8] font-medium">{deviceEngine}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-white/30">STATUS:</span>
          <span className="text-[#2dd4bf] font-medium">{systemStatus}</span>
        </div>
      </div>
    </header>
  );
};
