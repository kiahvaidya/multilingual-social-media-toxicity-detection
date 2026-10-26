import React from 'react';
import { motion } from 'framer-motion';
import { 
  ShieldAlert, 
  ShieldCheck, 
  AlertTriangle, 
  Languages, 
  Image as ImageIcon, 
  TrendingUp, 
  Activity,
  ArrowRight
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts';
import { NetworkScene } from '../components/three/NetworkScene';
import { ModerationQueueItem } from '../types';

interface OverviewPageProps {
  queueItems: ModerationQueueItem[];
  onNavigate: (tab: string) => void;
}

export function OverviewPage({ queueItems, onNavigate }: OverviewPageProps) {
  const totalAnalyzed = queueItems.length;
  const highRiskCount = queueItems.filter(i => i.risk_level === 'High Risk').length;
  const reviewCount = queueItems.filter(i => i.risk_level === 'Review').length;
  const safeCount = queueItems.filter(i => i.risk_level === 'Safe').length;
  const memeCount = queueItems.filter(i => i.type === 'MEME').length;

  // Language counts
  const langCounts: Record<string, number> = {};
  queueItems.forEach(i => {
    langCounts[i.language] = (langCounts[i.language] || 0) + 1;
  });
  const languageData = Object.entries(langCounts).map(([lang, count]) => ({ name: lang, value: count }));

  // Risk Distribution Data
  const riskDistribution = [
    { name: 'Safe', count: safeCount, color: '#10b981' },
    { name: 'Review', count: reviewCount, color: '#f59e0b' },
    { name: 'High Risk', count: highRiskCount, color: '#f43f5e' },
  ];

  const stats = [
    {
      label: 'COMMENTS ANALYZED',
      value: totalAnalyzed,
      subtext: 'Across all streams',
      icon: Activity,
      color: 'text-cyan-400',
      border: 'border-cyan-500/20',
      bg: 'bg-cyan-950/20'
    },
    {
      label: 'HIGH RISK DETECTED',
      value: highRiskCount,
      subtext: 'Critical toxic signals',
      icon: ShieldAlert,
      color: 'text-rose-400',
      border: 'border-rose-500/20',
      bg: 'bg-rose-950/20'
    },
    {
      label: 'ASSISTIVE REVIEW QUEUE',
      value: reviewCount,
      subtext: 'Borderline or JEV verified',
      icon: AlertTriangle,
      color: 'text-amber-400',
      border: 'border-amber-500/20',
      bg: 'bg-amber-950/20'
    },
    {
      label: 'INDIC & EN LANGUAGES',
      value: '17+',
      subtext: 'Native MuRIL coverage',
      icon: Languages,
      color: 'text-blue-400',
      border: 'border-blue-500/20',
      bg: 'bg-blue-950/20'
    },
    {
      label: 'MULTIMODAL MEMES',
      value: memeCount,
      subtext: 'MuRIL + CLIP ViT-B/32',
      icon: ImageIcon,
      color: 'text-purple-400',
      border: 'border-purple-500/20',
      bg: 'bg-purple-950/20'
    },
  ];

  return (
    <div className="space-y-8 pb-12">
      {/* 1. Top Metrics Bar */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {stats.map((s, idx) => {
          const Icon = s.icon;
          return (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.08, duration: 0.3 }}
              className={`p-4 rounded-xl glass-card border ${s.border} ${s.bg} flex flex-col justify-between`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">{s.label}</span>
                <Icon className={`w-4 h-4 ${s.color}`} />
              </div>
              <div className="mt-3">
                <span className={`text-2xl font-bold font-mono tracking-tight ${s.color}`}>
                  {s.value}
                </span>
                <p className="text-[11px] text-slate-400 mt-0.5 truncate">{s.subtext}</p>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* 2. Main Area: 3D Visualization (Left) & Live Moderation Feed (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left: 3D Neural Architecture Scene */}
        <div className="lg:col-span-7 flex flex-col">
          <div className="flex-1 min-h-[480px]">
            <NetworkScene />
          </div>
        </div>

        {/* Right: Live Moderation Stream */}
        <div className="lg:col-span-5 flex flex-col glass-panel rounded-2xl border border-slate-800/80 p-5 min-h-[480px]">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center space-x-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
              <h2 className="text-sm font-mono uppercase tracking-wider text-slate-100 font-bold">
                Live Moderation Intelligence
              </h2>
            </div>
            <button
              onClick={() => onNavigate('queue')}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
            >
              <span>VIEW ALL</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="mt-4 flex-1 overflow-y-auto space-y-3 max-h-[390px] pr-1">
            {queueItems.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-400">
                <Activity className="w-10 h-10 text-slate-600 mb-3 animate-pulse" />
                <p className="text-sm font-medium text-slate-300">No content analyzed yet in this session.</p>
                <p className="text-xs text-slate-400 mt-1 max-w-xs">
                  Run an analysis in Content Analyzer or Meme Intelligence to observe live classification decisions.
                </p>
                <button
                  onClick={() => onNavigate('analyze')}
                  className="mt-4 px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-dark-950 font-bold text-xs uppercase tracking-wider transition font-mono"
                >
                  Analyze Content
                </button>
              </div>
            ) : (
              queueItems.slice(0, 8).map((item) => {
                const isHigh = item.risk_level === 'High Risk';
                const isReview = item.risk_level === 'Review';
                return (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="p-3.5 rounded-xl glass-card border border-slate-800/80 hover:border-slate-700 space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className={`px-2 py-0.5 rounded font-mono font-bold text-[10px] uppercase tracking-wider ${
                        isHigh
                          ? 'cyber-badge-high'
                          : isReview
                          ? 'cyber-badge-review'
                          : 'cyber-badge-safe'
                      }`}>
                        {item.risk_level}
                      </span>
                      <span className="text-slate-400 font-mono text-[11px]">{item.language}</span>
                    </div>

                    <p className="text-xs text-slate-200 line-clamp-2 leading-relaxed">
                      {item.content}
                    </p>

                    <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800/40">
                      <span>Score: <strong className="text-slate-200">{(item.score * 100).toFixed(1)}%</strong></span>
                      <span>{item.type === 'MEME' ? '🖼️ MEME' : '📝 TEXT'}</span>
                      <span className="text-[10px] text-slate-400">{item.timestamp}</span>
                    </div>
                  </motion.div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* 3. Analytics Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Risk Level Triage Breakdown */}
        <div className="p-5 rounded-2xl glass-panel border border-slate-800">
          <h3 className="text-sm font-mono uppercase tracking-wider text-slate-200 font-bold mb-4 flex items-center justify-between">
            <span>Assistive Risk Distribution</span>
            <span className="text-xs text-slate-400 font-normal">Active Session Tiers</span>
          </h3>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskDistribution}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0b0d13', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Language Spread */}
        <div className="p-5 rounded-2xl glass-panel border border-slate-800">
          <h3 className="text-sm font-mono uppercase tracking-wider text-slate-200 font-bold mb-4 flex items-center justify-between">
            <span>Multilingual Language Representation</span>
            <span className="text-xs text-slate-400 font-normal">Detected Scripts</span>
          </h3>
          <div className="h-60 w-full flex items-center justify-center">
            {languageData.length === 0 ? (
              <span className="text-xs text-slate-400 font-mono">No multilingual samples processed yet.</span>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={languageData} layout="vertical">
                  <XAxis type="number" stroke="#64748b" fontSize={11} allowDecimals={false} />
                  <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={11} width={100} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0b0d13', borderColor: '#334155', borderRadius: '8px' }}
                  />
                  <Bar dataKey="value" fill="#38bdf8" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
export default OverviewPage;
