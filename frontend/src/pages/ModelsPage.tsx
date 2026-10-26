import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Sparkles, 
  Layers, 
  ArrowRight, 
  ShieldCheck, 
  CheckCircle2, 
  AlertCircle,
  FileText,
  Eye,
  GitBranch
} from 'lucide-react';
import { fetchResearchMetrics } from '../api/client';

export function ModelsPage() {
  const [metricsData, setMetricsData] = useState<any | null>(null);

  useEffect(() => {
    fetchResearchMetrics().then(data => setMetricsData(data)).catch(() => {});
  }, []);

  return (
    <div className="space-y-10 max-w-6xl mx-auto pb-16">
      {/* 1. Page Header */}
      <div>
        <h2 className="text-xl font-mono font-bold text-slate-100 flex items-center gap-2.5">
          <Cpu className="w-5 h-5 text-cyan-400" />
          MODEL INTELLIGENCE & RESEARCH ARCHITECTURE
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl">
          Complete neural specification of the production MuRIL transformer, CLIP ViT-B/32 vision backbone, cross-modal interaction fusion layer, and JEV adjudication policy.
        </p>
      </div>

      {/* 2. Interactive Neural Architecture Topology */}
      <div className="p-8 rounded-2xl glass-panel border border-slate-800 space-y-8">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <GitBranch className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              Dual-Stream Multimodal Fusion Pipeline
            </h3>
          </div>
          <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
            PROD_MODEL_V2
          </span>
        </div>

        {/* Visual Graph Nodes */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          {/* Stream 1: MuRIL Text Branch */}
          <div className="p-5 rounded-2xl bg-dark-900 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-cyan-400">TEXT BRANCH</span>
              <span className="text-[10px] font-mono text-slate-500">768-D DENSE</span>
            </div>
            <h4 className="text-sm font-bold text-slate-100">google/muril-base-cased</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Fine-tuned multilingual BERT variant pre-trained on 17 Indic languages plus English. Emits 768-dimensional normalized CLS token embeddings.
            </p>
            <div className="text-[11px] font-mono text-cyan-300 bg-cyan-950/40 p-2 rounded border border-cyan-500/20">
              Projection: Linear(768 → 256) + LayerNorm + GELU
            </div>
          </div>

          {/* Central Interaction Fusion Core */}
          <div className="p-6 rounded-2xl bg-gradient-to-b from-dark-900 to-dark-950 border border-purple-500/40 shadow-glow-purple/20 space-y-3 relative text-center">
            <span className="text-xs font-mono font-extrabold text-purple-300 uppercase tracking-wider">
              CROSS-MODAL INTERACTION
            </span>
            <div className="p-3 bg-purple-950/40 rounded-xl border border-purple-500/30 font-mono text-xs text-purple-200">
              [ t , v , t ⊙ v , |t - v| ]
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Computes elementwise semantic alignment (product) and contextual disparity (absolute difference) across 1024 dimensions.
            </p>
            <div className="text-[10px] font-mono text-slate-400 pt-1 border-t border-slate-800">
              Modality Dropout: p = 0.15 (eliminates visual shortcut)
            </div>
          </div>

          {/* Stream 2: CLIP Vision Branch */}
          <div className="p-5 rounded-2xl bg-dark-900 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-purple-400">VISION BRANCH</span>
              <span className="text-[10px] font-mono text-slate-500">512-D DENSE</span>
            </div>
            <h4 className="text-sm font-bold text-slate-100">CLIP ViT-B/32 Vision</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Pretrained OpenAI Vision Transformer (patch size 32) generating 512-dimensional visual embeddings capturing meme composition & template context.
            </p>
            <div className="text-[11px] font-mono text-purple-300 bg-purple-950/40 p-2 rounded border border-purple-500/20">
              Projection: Linear(512 → 256) + LayerNorm + GELU
            </div>
          </div>
        </div>

        {/* Downstream Triage */}
        <div className="flex flex-col md:flex-row items-center justify-between p-4 rounded-xl bg-dark-900 border border-slate-800 text-xs font-mono gap-4">
          <div className="flex items-center space-x-2">
            <ArrowRight className="w-4 h-4 text-cyan-400" />
            <span className="text-slate-300">CLASSIFIER HEAD:</span>
            <span className="text-cyan-300 font-bold">Linear(1024 → 256) → GELU → Dropout(0.2) → Linear(256 → 1)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-slate-400">CALIBRATED THRESHOLD:</span>
            <span className="text-amber-400 font-bold px-2 py-0.5 rounded bg-amber-950 border border-amber-500/30">T* = 0.44</span>
          </div>
        </div>
      </div>

      {/* 3. Empirical Research Benchmarks */}
      <div className="space-y-4">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
          Empirical Validation Results & Generalization Evidence
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* PrideMM In-Domain */}
          <div className="p-6 rounded-2xl glass-panel border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-emerald-400">IN-DOMAIN (PrideMM Test, N=507)</span>
              <span className="text-[11px] font-mono text-slate-500">Verified Test Set</span>
            </div>
            <div className="grid grid-cols-3 gap-3 text-center font-mono py-2">
              <div className="p-2.5 bg-dark-900 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500">F1 SCORE</span>
                <p className="text-base font-bold text-slate-200">0.7097</p>
              </div>
              <div className="p-2.5 bg-dark-900 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500">AUROC</span>
                <p className="text-base font-bold text-slate-200">0.7790</p>
              </div>
              <div className="p-2.5 bg-dark-900 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500">ACCURACY</span>
                <p className="text-base font-bold text-slate-200">69.82%</p>
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Drastic improvement over text-only Model B (F1: 0.3198 / 0.0080). Shuffling images collapses AUROC to 0.5040 (-0.2750), proving active multimodal dependency.
            </p>
          </div>

          {/* Multi3Hate External Benchmark */}
          <div className="p-6 rounded-2xl glass-panel border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-cyan-400">HELD-OUT BENCHMARK (Multi3Hate, N=600)</span>
              <span className="text-[11px] font-mono text-slate-500">Strictly Untuned</span>
            </div>
            <div className="grid grid-cols-3 gap-3 text-center font-mono py-2">
              <div className="p-2.5 bg-dark-900 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500">PRECISION</span>
                <p className="text-base font-bold text-slate-200">0.6742</p>
              </div>
              <div className="p-2.5 bg-dark-900 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500">AUROC</span>
                <p className="text-base font-bold text-slate-200">0.5835</p>
              </div>
              <div className="p-2.5 bg-dark-900 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500">FPR</span>
                <p className="text-base font-bold text-emerald-400">30.00%</p>
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              FPR successfully curtailed from 56.67% down to 30.00%. Strong English transfer (F1: 0.6628, AUROC: 0.6411), with Indic transfer constrained by zero Hindi paired training data (AUROC: 0.5040).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
export default ModelsPage;
