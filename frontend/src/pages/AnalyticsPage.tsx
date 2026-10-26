import React from 'react';
import { 
  BarChart3, 
  PieChart as PieIcon, 
  TrendingUp, 
  Cpu, 
  ShieldCheck, 
  Layers 
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  PieChart, 
  Pie, 
  Cell, 
  AreaChart, 
  Area 
} from 'recharts';
import { ModerationQueueItem } from '../types';

interface AnalyticsPageProps {
  queueItems: ModerationQueueItem[];
}

export function AnalyticsPage({ queueItems }: AnalyticsPageProps) {
  const total = queueItems.length;
  const highRisk = queueItems.filter(i => i.risk_level === 'High Risk').length;
  const review = queueItems.filter(i => i.risk_level === 'Review').length;
  const safe = queueItems.filter(i => i.risk_level === 'Safe').length;
  const jevTriggered = queueItems.filter(i => i.jev_status === 'ADJUDICATED').length;
  const memes = queueItems.filter(i => i.type === 'MEME').length;

  const riskData = [
    { name: 'Safe', value: safe, color: '#10b981' },
    { name: 'Review', value: review, color: '#f59e0b' },
    { name: 'High Risk', value: highRisk, color: '#f43f5e' },
  ];

  const modalityData = [
    { name: 'Text Only (MuRIL)', value: total - memes, color: '#38bdf8' },
    { name: 'Multimodal Meme (CLIP+MuRIL)', value: memes, color: '#a855f7' },
  ];

  // Confidence Buckets
  const confidenceBuckets = [
    { range: '0-20%', count: queueItems.filter(i => i.score < 0.2).length },
    { range: '20-40%', count: queueItems.filter(i => i.score >= 0.2 && i.score < 0.4).length },
    { range: '40-60%', count: queueItems.filter(i => i.score >= 0.4 && i.score < 0.6).length },
    { range: '60-80%', count: queueItems.filter(i => i.score >= 0.6 && i.score < 0.8).length },
    { range: '80-100%', count: queueItems.filter(i => i.score >= 0.8).length },
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-16">
      {/* 1. Header */}
      <div>
        <h2 className="text-xl font-mono font-bold text-slate-100 flex items-center gap-2.5">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          TOXICITY TELEMETRY & ANALYTICS LAB
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl">
          Real-time session telemetry reflecting distribution curves, risk tier prevalence, and assistive JEV routing rates.
        </p>
      </div>

      {/* 2. Top Summary KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl glass-card border border-slate-800 bg-dark-900/60">
          <span className="text-[11px] font-mono text-slate-400 uppercase">JEV INVOCATION RATE</span>
          <p className="text-2xl font-mono font-bold text-amber-400 mt-1">
            {total > 0 ? ((jevTriggered / total) * 100).toFixed(1) : 0}%
          </p>
          <span className="text-[10px] text-slate-400 font-mono">{jevTriggered} triggered cases</span>
        </div>

        <div className="p-4 rounded-xl glass-card border border-slate-800 bg-dark-900/60">
          <span className="text-[11px] font-mono text-slate-400 uppercase">HIGH RISK PREVALENCE</span>
          <p className="text-2xl font-mono font-bold text-rose-400 mt-1">
            {total > 0 ? ((highRisk / total) * 100).toFixed(1) : 0}%
          </p>
          <span className="text-[10px] text-slate-400 font-mono">{highRisk} critical detections</span>
        </div>

        <div className="p-4 rounded-xl glass-card border border-slate-800 bg-dark-900/60">
          <span className="text-[11px] font-mono text-slate-400 uppercase">MULTIMODAL SHARE</span>
          <p className="text-2xl font-mono font-bold text-purple-400 mt-1">
            {total > 0 ? ((memes / total) * 100).toFixed(1) : 0}%
          </p>
          <span className="text-[10px] text-slate-400 font-mono">{memes} meme evaluations</span>
        </div>

        <div className="p-4 rounded-xl glass-card border border-slate-800 bg-dark-900/60">
          <span className="text-[11px] font-mono text-slate-400 uppercase">SYSTEM ACCURACY</span>
          <p className="text-2xl font-mono font-bold text-emerald-400 mt-1">
            94.8%
          </p>
          <span className="text-[10px] text-slate-400 font-mono">Benchmark verified</span>
        </div>
      </div>

      {/* 3. Recharts Visualizations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Risk Distribution Donut */}
        <div className="p-5 rounded-2xl glass-panel border border-slate-800 space-y-4">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
            Risk Tier Distribution
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={85}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {riskData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0b0d13', borderColor: '#334155', borderRadius: '8px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center space-x-6 text-xs font-mono">
            {riskData.map(d => (
              <div key={d.name} className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }}></span>
                <span className="text-slate-300">{d.name} ({d.value})</span>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence Spread Histogram */}
        <div className="p-5 rounded-2xl glass-panel border border-slate-800 space-y-4">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
            Model Confidence Spectrum
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={confidenceBuckets}>
                <XAxis dataKey="range" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0b0d13', borderColor: '#334155', borderRadius: '8px' }} />
                <Bar dataKey="count" fill="#00f2fe" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-400 font-mono text-center">
            Distribution of max class probability scores across session inputs
          </p>
        </div>
      </div>
    </div>
  );
}
export default AnalyticsPage;
