import React from 'react';
import { ShieldCheck, Cpu, Terminal, Sparkles, RefreshCw } from 'lucide-react';

interface HeaderProps {
  currentTab: string;
  onResetSession?: () => void;
  systemHealth?: any;
}

export function Header({ currentTab, onResetSession, systemHealth }: HeaderProps) {
  const titles: Record<string, { title: string; subtitle: string }> = {
    overview: {
      title: 'AI Moderation Command Center',
      subtitle: 'Multilingual text classification, multimodal meme intelligence & real-time risk triage.'
    },
    analyze: {
      title: 'Content Analysis Workspace',
      subtitle: 'Deep neural evaluation across 6 toxicity dimensions with JEV contextual adjudication.'
    },
    meme: {
      title: 'Multimodal Meme Intelligence Lab',
      subtitle: 'Cross-modal interaction reasoning: Contextual Inversion, Emergent Toxicity & CLIP Fusion.'
    },
    queue: {
      title: 'Security Operations Moderation Queue',
      subtitle: 'Assistive 3-tier triage queue with audit trails and human-in-the-loop escalation.'
    },
    analytics: {
      title: 'Toxicity Telemetry & Analytics',
      subtitle: 'Distribution curves, risk prevalence, multilingual language coverage and JEV triggers.'
    },
    models: {
      title: 'Model Intelligence & Research Specifications',
      subtitle: 'MuRIL transformer architecture, CLIP ViT-B/32 vision backbone, and empirical benchmark evaluations.'
    }
  };

  const current = titles[currentTab] || titles.overview;

  return (
    <header className="h-20 px-8 py-4 glass-panel border-b border-slate-800/80 flex items-center justify-between sticky top-0 z-30">
      {/* Title & Context */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-slate-100 flex items-center gap-2.5">
          {current.title}
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-500/30">
            PROD_V2
          </span>
        </h1>
        <p className="text-xs text-slate-400 mt-0.5 max-w-2xl truncate">
          {current.subtitle}
        </p>
      </div>

      {/* Telemetry Chips */}
      <div className="hidden lg:flex items-center space-x-3 text-xs font-mono">
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-dark-900/90 border border-slate-800">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">ENGINE:</span>
          <span className="text-cyan-300 font-semibold">MuRIL BASE</span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-dark-900/90 border border-slate-800">
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          <span className="text-slate-400">VISION:</span>
          <span className="text-purple-300 font-semibold">CLIP ViT-B/32</span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-dark-900/90 border border-slate-800">
          <ShieldCheck className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-slate-400">JEV:</span>
          <span className="text-amber-300 font-semibold">ADJUDICATION ONLINE</span>
        </div>

        {onResetSession && (
          <button
            onClick={onResetSession}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 border border-slate-700 transition"
            title="Reset active session telemetry"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>RESET</span>
          </button>
        )}
      </div>
    </header>
  );
}
export default Header;
