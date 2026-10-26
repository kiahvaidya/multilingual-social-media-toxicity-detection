import React, { useState } from 'react';
import { 
  Layers, 
  Check, 
  Trash2, 
  AlertOctagon, 
  Flag, 
  Filter, 
  ShieldCheck, 
  ShieldAlert, 
  AlertTriangle,
  Image as ImageIcon,
  FileText
} from 'lucide-react';
import { ModerationQueueItem } from '../types';

interface QueuePageProps {
  items: ModerationQueueItem[];
  onUpdateStatus: (id: string, newStatus: ModerationQueueItem['status']) => void;
  onClearQueue: () => void;
}

export function QueuePage({ items, onUpdateStatus, onClearQueue }: QueuePageProps) {
  const [filter, setFilter] = useState<'ALL' | 'HIGH_RISK' | 'REVIEW' | 'SAFE' | 'JEV'>('ALL');

  const filteredItems = items.filter(item => {
    if (filter === 'HIGH_RISK') return item.risk_level === 'High Risk';
    if (filter === 'REVIEW') return item.risk_level === 'Review';
    if (filter === 'SAFE') return item.risk_level === 'Safe';
    if (filter === 'JEV') return item.jev_status === 'ADJUDICATED';
    return true;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-16">
      {/* 1. Header & Quick Filters */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-mono font-bold text-slate-100 flex items-center gap-2.5">
            <Layers className="w-5 h-5 text-cyan-400" />
            SECURITY OPERATIONS MODERATION QUEUE
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Assistive 3-tier triage queue with audit trails and human-in-the-loop escalation.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {items.length > 0 && (
            <button
              onClick={onClearQueue}
              className="text-xs font-mono text-slate-400 hover:text-slate-200 px-3 py-1.5 rounded-lg bg-dark-900 border border-slate-800 transition"
            >
              CLEAR QUEUE
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap gap-2 text-xs font-mono">
        {[
          { id: 'ALL', label: `ALL (${items.length})` },
          { id: 'HIGH_RISK', label: `HIGH RISK (${items.filter(i => i.risk_level === 'High Risk').length})` },
          { id: 'REVIEW', label: `REVIEW (${items.filter(i => i.risk_level === 'Review').length})` },
          { id: 'SAFE', label: `SAFE (${items.filter(i => i.risk_level === 'Safe').length})` },
          { id: 'JEV', label: `JEV ADJUDICATED (${items.filter(i => i.jev_status === 'ADJUDICATED').length})` },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setFilter(tab.id as any)}
            className={`px-3.5 py-1.5 rounded-xl border transition ${
              filter === tab.id
                ? 'bg-cyan-950/80 text-cyan-300 border-cyan-500/40 font-bold'
                : 'bg-dark-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Queue Table */}
      {filteredItems.length === 0 ? (
        <div className="glass-panel p-12 rounded-2xl border border-slate-800 text-center space-y-3">
          <Layers className="w-12 h-12 text-slate-600 mx-auto" />
          <h3 className="text-sm font-bold text-slate-300 font-mono">Queue is empty</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Analyze comments or memes in the workspace to automatically populate items into this triage console.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredItems.map((item) => {
            const isHigh = item.risk_level === 'High Risk';
            const isReview = item.risk_level === 'Review';
            return (
              <div
                key={item.id}
                className="p-4 rounded-xl glass-panel border border-slate-800/80 hover:border-slate-700 transition flex flex-col md:flex-row items-start md:items-center justify-between gap-4"
              >
                {/* Content & Metadata */}
                <div className="space-y-2 flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                    <span className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                      isHigh ? 'cyber-badge-high' : isReview ? 'cyber-badge-review' : 'cyber-badge-safe'
                    }`}>
                      {item.risk_level}
                    </span>
                    <span className="text-slate-400 bg-dark-900 px-2 py-0.5 rounded border border-slate-800 text-[11px]">
                      {item.language}
                    </span>
                    <span className="text-slate-400 text-[11px] flex items-center gap-1">
                      {item.type === 'MEME' ? <ImageIcon className="w-3 h-3 text-purple-400" /> : <FileText className="w-3 h-3 text-cyan-400" />}
                      {item.type}
                    </span>
                    <span className="text-slate-500 text-[10px]">{item.timestamp}</span>
                  </div>

                  <p className="text-xs text-slate-200 line-clamp-2 leading-relaxed">
                    {item.content}
                  </p>

                  <div className="text-[11px] font-mono text-slate-400 flex items-center space-x-4">
                    <span>Toxicity Score: <strong className="text-slate-200">{(item.score * 100).toFixed(1)}%</strong></span>
                    <span>Status: <strong className="text-cyan-400">{item.status}</strong></span>
                  </div>
                </div>

                {/* Operations Actions */}
                <div className="flex flex-wrap items-center gap-2 self-end md:self-center font-mono text-xs">
                  <button
                    onClick={() => onUpdateStatus(item.id, 'APPROVED')}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 border border-emerald-500/30 transition"
                    title="Approve as safe content"
                  >
                    <Check className="w-3.5 h-3.5" />
                    <span>APPROVE</span>
                  </button>

                  <button
                    onClick={() => onUpdateStatus(item.id, 'REMOVED')}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 border border-rose-500/30 transition"
                    title="Enforce removal"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>REMOVE</span>
                  </button>

                  <button
                    onClick={() => onUpdateStatus(item.id, 'ESCALATED')}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-amber-950/60 hover:bg-amber-900/60 text-amber-300 border border-amber-500/30 transition"
                    title="Escalate to senior review"
                  >
                    <AlertOctagon className="w-3.5 h-3.5" />
                    <span>ESCALATE</span>
                  </button>

                  <button
                    onClick={() => onUpdateStatus(item.id, 'FALSE_POSITIVE')}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
                    title="Mark as false alarm"
                  >
                    <Flag className="w-3.5 h-3.5" />
                    <span>FP</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
export default QueuePage;
