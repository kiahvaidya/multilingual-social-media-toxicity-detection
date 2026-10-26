import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface QueueItem {
  id: string;
  timestamp: string;
  textSnippet: string;
  language: 'hi' | 'en' | 'hinglish' | 'bn' | 'mr';
  riskLevel: 'HIGH RISK' | 'REVIEW' | 'SAFE';
  confidence: number;
  type: 'meme' | 'text';
  category: string;
  hasImage: boolean;
}

const SAMPLE_QUEUE: QueueItem[] = [
  {
    id: 'CASE-9842',
    timestamp: '14:22:09 UTC',
    textSnippet: 'ये लोग कभी नहीं सुधरेंगे इनको देश से निकालो',
    language: 'hi',
    riskLevel: 'HIGH RISK',
    confidence: 0.942,
    type: 'meme',
    category: 'Identity Hate / Xenophobia',
    hasImage: true,
  },
  {
    id: 'CASE-9843',
    timestamp: '14:21:40 UTC',
    textSnippet: 'bhai tu toh sach me zeher lag raha hai aaj',
    language: 'hinglish',
    riskLevel: 'SAFE',
    confidence: 0.128,
    type: 'text',
    category: 'Colloquial Compliment (Defused)',
    hasImage: false,
  },
  {
    id: 'CASE-9844',
    timestamp: '14:19:12 UTC',
    textSnippet: 'Just an innocent meme about community gardening',
    language: 'en',
    riskLevel: 'HIGH RISK',
    confidence: 0.887,
    type: 'meme',
    category: 'Emergent Hate Symbolism',
    hasImage: true,
  },
  {
    id: 'CASE-9845',
    timestamp: '14:18:05 UTC',
    textSnippet: 'तुम्हाला काही अक्कल आहे का मूर्खासारखे वागता',
    language: 'mr',
    riskLevel: 'REVIEW',
    confidence: 0.651,
    type: 'text',
    category: 'Insult / Sarcasm Borderline',
    hasImage: false,
  },
  {
    id: 'CASE-9846',
    timestamp: '14:15:30 UTC',
    textSnippet: 'I swear I will strangle you if you spoil the movie',
    language: 'en',
    riskLevel: 'SAFE',
    confidence: 0.142,
    type: 'meme',
    category: 'Contextual Inversion (Playful)',
    hasImage: true,
  },
  {
    id: 'CASE-9847',
    timestamp: '14:11:58 UTC',
    textSnippet: 'এই নোংরা মানসিকতা সমাজে আর সহ্য করা যাবে না',
    language: 'bn',
    riskLevel: 'REVIEW',
    confidence: 0.612,
    type: 'text',
    category: 'Aggressive Denunciation',
    hasImage: false,
  },
  {
    id: 'CASE-9848',
    timestamp: '14:08:44 UTC',
    textSnippet: 'terko raste pe dekh lunga kal clg ke bahar',
    language: 'hinglish',
    riskLevel: 'HIGH RISK',
    confidence: 0.915,
    type: 'text',
    category: 'Direct Physical Threat',
    hasImage: false,
  },
];

interface QueueOverlayProps {
  onSelectCase: (item: QueueItem) => void;
}

export const QueueOverlay: React.FC<QueueOverlayProps> = ({ onSelectCase }) => {
  const [filter, setFilter] = useState<'ALL' | 'HIGH RISK' | 'REVIEW' | 'SAFE'>('ALL');
  const [selectedCase, setSelectedCase] = useState<QueueItem | null>(SAMPLE_QUEUE[0]);

  const filteredItems = SAMPLE_QUEUE.filter(
    (item) => filter === 'ALL' || item.riskLevel === filter
  );

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-6 sm:p-12 z-20 overflow-hidden">
      {/* Header & Filter Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-white/40" />
            <span className="text-[10px] font-mono tracking-widest uppercase text-white/50">
              AUDIT STREAM // MULTILINGUAL LIVE INGESTION
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-display uppercase tracking-tight text-white">
            Forensic Moderation Dossier
          </h1>
          <p className="text-xs font-mono text-white/40 mt-1">
            Live evaluation of cross-modal content flagged across 5 Indic & global language streams.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="pointer-events-auto flex items-center bg-obsidian/90 border border-white/10 p-1 backdrop-blur-md">
          {(['ALL', 'HIGH RISK', 'REVIEW', 'SAFE'] as const).map((lvl) => (
            <button
              key={lvl}
              onClick={() => setFilter(lvl)}
              className={`px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider transition-colors ${
                filter === lvl
                  ? lvl === 'HIGH RISK'
                    ? 'bg-forensic-risk/20 text-forensic-risk border border-forensic-risk/40'
                    : lvl === 'REVIEW'
                    ? 'bg-forensic-review/20 text-forensic-review border border-forensic-review/40'
                    : lvl === 'SAFE'
                    ? 'bg-forensic-safe/20 text-forensic-safe border border-forensic-safe/40'
                    : 'bg-white/20 text-white border border-white/40'
                  : 'text-white/40 hover:text-white'
              }`}
            >
              {lvl}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area: Split List & Selected Dossier Detail */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 my-4 flex-1 min-h-0 pointer-events-auto">
        {/* Editorial Investigation List (7 Cols) */}
        <div className="lg:col-span-7 overflow-y-auto pr-2 space-y-2 border border-white/5 bg-obsidian/50 p-2 backdrop-blur-sm">
          {filteredItems.map((item) => {
            const isSelected = selectedCase?.id === item.id;
            const badgeColor =
              item.riskLevel === 'HIGH RISK'
                ? 'text-forensic-risk border-forensic-risk/40 bg-forensic-risk/10'
                : item.riskLevel === 'REVIEW'
                ? 'text-forensic-review border-forensic-review/40 bg-forensic-review/10'
                : 'text-forensic-safe border-forensic-safe/40 bg-forensic-safe/10';

            return (
              <div
                key={item.id}
                onClick={() => setSelectedCase(item)}
                className={`p-3 border cursor-pointer transition-all duration-200 group flex items-start gap-4 ${
                  isSelected
                    ? 'border-white/40 bg-white/[0.07] shadow-[0_0_20px_rgba(255,255,255,0.03)]'
                    : 'border-white/5 bg-obsidian/60 hover:border-white/20 hover:bg-white/[0.02]'
                }`}
              >
                {/* Visual Icon / Type Indicator */}
                <div className="w-12 h-12 flex-shrink-0 bg-charcoal border border-white/10 flex flex-col items-center justify-center font-mono text-[9px] uppercase tracking-wider text-white/50">
                  <span>{item.type}</span>
                  <span className="text-[8px] text-white/30">{item.language}</span>
                </div>

                {/* Content Fragment */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="font-mono text-xs text-white/50">{item.id}</span>
                    <span className="font-mono text-[10px] text-white/30">{item.timestamp}</span>
                  </div>
                  <div className="font-sans text-sm text-white/90 truncate group-hover:text-white">
                    {item.textSnippet}
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <span
                      className={`text-[9px] font-mono px-2 py-0.5 border uppercase tracking-wider ${badgeColor}`}
                    >
                      {item.riskLevel}
                    </span>
                    <span className="text-[10px] font-mono text-white/40">{item.category}</span>
                  </div>
                </div>

                {/* Score */}
                <div className="text-right flex-shrink-0 font-mono text-xs font-bold text-white/80 self-center">
                  {(item.confidence * 100).toFixed(0)}%
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Dossier Forensic Breakdown (5 Cols) */}
        <div className="lg:col-span-5 bg-obsidian/95 border border-white/10 p-6 flex flex-col justify-between backdrop-blur-md">
          {selectedCase ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-white/10 pb-3">
                <span className="font-mono text-xs tracking-widest text-forensic-beam">
                  INVESTIGATION DOSSIER // {selectedCase.id}
                </span>
                <span className="text-[10px] font-mono text-white/40">{selectedCase.timestamp}</span>
              </div>

              <div>
                <div className="text-[10px] font-mono uppercase text-white/40 mb-1">Subject Content</div>
                <div className="p-3 bg-charcoal/80 border border-white/5 font-sans text-sm text-white leading-relaxed">
                  "{selectedCase.textSnippet}"
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="border border-white/5 p-2 bg-charcoal/40">
                  <span className="text-white/40 block text-[9px]">LANGUAGE MODEL</span>
                  <span className="text-white uppercase">{selectedCase.language} (MuRIL Fine-Tuned)</span>
                </div>
                <div className="border border-white/5 p-2 bg-charcoal/40">
                  <span className="text-white/40 block text-[9px]">MODALITY INGESTION</span>
                  <span className="text-white uppercase">
                    {selectedCase.hasImage ? 'Bimodal (CLIP + MuRIL)' : 'Unimodal (MuRIL)'}
                  </span>
                </div>
              </div>

              <div>
                <div className="text-[10px] font-mono uppercase text-white/40 mb-1">Adjudication Category</div>
                <div className="text-sm font-mono text-white font-medium">{selectedCase.category}</div>
              </div>

              <div className="border border-white/10 p-3 bg-void">
                <div className="flex justify-between items-center text-xs font-mono mb-2">
                  <span className="text-white/60">Risk Confidence:</span>
                  <span
                    className={
                      selectedCase.riskLevel === 'HIGH RISK'
                        ? 'text-forensic-risk font-bold'
                        : selectedCase.riskLevel === 'REVIEW'
                        ? 'text-forensic-review font-bold'
                        : 'text-forensic-safe font-bold'
                    }
                  >
                    {(selectedCase.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-white/5 h-1.5 overflow-hidden">
                  <div
                    className={`h-full ${
                      selectedCase.riskLevel === 'HIGH RISK'
                        ? 'bg-forensic-risk'
                        : selectedCase.riskLevel === 'REVIEW'
                        ? 'bg-forensic-review'
                        : 'bg-forensic-safe'
                    }`}
                    style={{ width: `${selectedCase.confidence * 100}%` }}
                  />
                </div>
              </div>

              <button
                onClick={() => onSelectCase(selectedCase)}
                className="w-full py-3 bg-white text-void font-mono text-xs uppercase tracking-widest font-bold hover:bg-forensic-beam transition-colors shadow-[0_0_20px_rgba(255,255,255,0.1)]"
              >
                Inspect in 3D Fusion Chamber →
              </button>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center font-mono text-xs text-white/40">
              Select an ingested case to load dossier telemetry.
            </div>
          )}

          <div className="border-t border-white/5 pt-3 text-[10px] font-mono text-white/30 flex justify-between">
            <span>POLICY ENGINE: JEV HYBRID ADJUDICATION</span>
            <span>STATUS: SYNCHRONIZED</span>
          </div>
        </div>
      </div>

      {/* Bottom Summary Bar */}
      <div className="flex items-center justify-between text-xs font-mono text-white/40 pt-2 border-t border-white/10">
        <span>SHOWING {filteredItems.length} OF {SAMPLE_QUEUE.length} FLAGGED CORPUS CASES</span>
        <span>LATENCY: 42ms // INFERENCE: LOCAL GPU</span>
      </div>
    </div>
  );
};
