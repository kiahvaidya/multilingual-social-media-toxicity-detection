import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Layers, 
  Upload, 
  Sparkles, 
  ShieldCheck, 
  ShieldAlert, 
  ArrowRight, 
  Eye, 
  RefreshCcw,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import { analyzeMeme, fetchCaseStudies } from '../api/client';
import { MemeAnalysisResult } from '../types';

export function MemePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MemeAnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [caseStudies, setCaseStudies] = useState<any | null>(null);

  useEffect(() => {
    fetchCaseStudies().then(data => setCaseStudies(data)).catch(() => {});
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setSelectedFile(f);
      setPreviewUrl(URL.createObjectURL(f));
      setResult(null);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await analyzeMeme(selectedFile, caption);
      setResult(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Meme analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  // Empirical Research Showcase Cases from real Multi3Hate benchmarks
  const empiricalShowcase = [
    {
      id: 17,
      type: 'CONTEXTUAL_INVERSION',
      title: 'Contextual Inversion (Hostile Text Defused)',
      text: "YOU AIN'T GETTIN' NO FRIED CHICKEN TILL' YOU FINISH ALL YOUR COOL-AID",
      textScore: 0.6451,
      multimodalScore: 0.0026,
      prediction: 'SAFE (CLEARED)',
      explanation: 'Text contains elevated slang & profanity flags in text-only MuRIL (64.5%), but visual template proves benign humorous domestic context. Multimodal score collapses to 0.26%.'
    },
    {
      id: 3,
      type: 'EMERGENT_TOXICITY',
      title: 'Emergent Toxicity (Innocuous Text + Hateful Visual)',
      text: "So you're telling me that Macbeth actually trusted three random witches.",
      textScore: 0.2970,
      multimodalScore: 0.9985,
      prediction: 'HATEFUL (DETECTED)',
      explanation: 'Text alone appears completely benign (29.7% toxicity), but pairing with derogatory targeted meme template introduces explicit hate speech. Multimodal fusion catches it at 99.8% confidence.'
    }
  ];

  return (
    <div className="space-y-10 pb-16 max-w-6xl mx-auto">
      {/* 1. Page Header */}
      <div>
        <h2 className="text-xl font-mono font-bold text-slate-100 flex items-center gap-2.5">
          <Layers className="w-5 h-5 text-purple-400" />
          MULTIMODAL MEME INTELLIGENCE LAB
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl">
          Understanding toxicity arising from the non-linear semantic interaction between language, imagery, and cultural template framing.
        </p>
      </div>

      {/* 2. Interactive Analysis Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        {/* Upload & Form Column */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <label className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold block">
            Meme Image Upload
          </label>

          <div className="border-2 border-dashed border-slate-700 hover:border-purple-500/60 rounded-2xl p-6 text-center cursor-pointer transition relative bg-dark-900/40">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            />
            {previewUrl ? (
              <div className="space-y-2">
                <img src={previewUrl} alt="Upload" className="max-h-56 mx-auto rounded-lg object-contain shadow-md" />
                <span className="text-[11px] font-mono text-purple-400 block">Click to select different image</span>
              </div>
            ) : (
              <div className="py-8 space-y-3">
                <Upload className="w-10 h-10 text-purple-400 mx-auto animate-bounce" />
                <p className="text-xs text-slate-300 font-medium">Drag & drop meme image here</p>
                <p className="text-[10px] text-slate-500 font-mono">PNG, JPG, WEBP</p>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <label className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold block">
              Meme Caption / Text
            </label>
            <input
              type="text"
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              placeholder="Enter text overlay or transcription..."
              className="w-full glass-input rounded-xl px-4 py-2.5 text-xs font-sans placeholder-slate-500 focus:ring-1 focus:ring-purple-400"
            />
          </div>

          <button
            onClick={handleRunAnalysis}
            disabled={loading || !selectedFile}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-400 hover:to-indigo-500 text-dark-950 font-bold font-mono text-xs uppercase tracking-wider transition disabled:opacity-50 disabled:cursor-not-allowed shadow-glow-purple/40"
          >
            {loading ? 'Executing MuRIL + CLIP Fusion...' : 'Run Multimodal Analysis'}
          </button>
        </div>

        {/* Live Result / Interactive Diagram Column */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col justify-between">
          {result ? (
            <div className="space-y-6">
              {/* Primary Decision Banner */}
              <div className={`p-4 rounded-xl border flex items-center justify-between ${
                result.is_hateful
                  ? 'border-rose-500/40 bg-rose-950/20'
                  : 'border-emerald-500/40 bg-emerald-950/20'
              }`}>
                <div>
                  <span className={`text-xs font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded ${
                    result.is_hateful ? 'cyber-badge-high' : 'cyber-badge-safe'
                  }`}>
                    {result.is_hateful ? 'TOXIC / HATEFUL MEME' : 'BENIGN / SAFE MEME'}
                  </span>
                  <p className="text-xs text-slate-300 font-mono mt-1">
                    {result.interaction_description}
                  </p>
                </div>
                <div className="text-right font-mono">
                  <span className="text-[10px] text-slate-400 block">MULTIMODAL SCORE</span>
                  <span className="text-2xl font-bold text-slate-100">
                    {(result.multimodal_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Three-Stream Neural Signal Cards */}
              <div className="grid grid-cols-3 gap-3 font-mono text-xs text-center">
                <div className="p-3.5 rounded-xl bg-dark-900 border border-slate-800">
                  <span className="text-slate-400 text-[10px] block">TEXT SIGNAL (MuRIL)</span>
                  <span className="text-base font-bold text-cyan-400 block mt-1">
                    {(result.text_only_score * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-dark-900 border border-slate-800">
                  <span className="text-slate-400 text-[10px] block">VISION (CLIP)</span>
                  <span className="text-base font-bold text-purple-400 block mt-1">
                    {(result.image_only_score * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-dark-900 border border-purple-500/40 shadow-glow-purple/20">
                  <span className="text-purple-300 text-[10px] block">CROSS-MODAL [t*v, |t-v|]</span>
                  <span className="text-base font-bold text-slate-100 block mt-1">
                    {(result.multimodal_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Interactive Neural Flow Diagram */}
              <div className="p-4 rounded-xl bg-dark-950 border border-slate-800 space-y-2">
                <span className="text-[10px] font-mono uppercase text-slate-400 block">
                  Neural Flow Architecture
                </span>
                <div className="flex items-center justify-between text-xs font-mono py-2 text-slate-300">
                  <span className="px-2 py-1 bg-cyan-950 text-cyan-400 rounded border border-cyan-500/30">
                    Text: 768-D
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                  <span className="px-2.5 py-1 bg-purple-950 text-purple-300 rounded border border-purple-500/30 font-bold">
                    Interaction Matrix (1024-D)
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                  <span className={`px-2 py-1 rounded border font-bold ${
                    result.is_hateful ? 'bg-rose-950 text-rose-300 border-rose-500/30' : 'bg-emerald-950 text-emerald-300 border-emerald-500/30'
                  }`}>
                    {result.is_hateful ? 'HATE LOGIT' : 'SAFE LOGIT'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-500">
              <Sparkles className="w-12 h-12 text-purple-400/40 mb-3" />
              <p className="text-sm font-medium text-slate-300">Ready for Multimodal Evaluation</p>
              <p className="text-xs text-slate-400 mt-1 max-w-sm">
                Upload a meme to decompose text toxicity vs visual template cues in real time.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 3. Research Showcase: Contextual Inversion & Emergent Toxicity */}
      <div className="space-y-4 pt-4 border-t border-slate-800">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <Eye className="w-4 h-4 text-cyan-400" />
            Empirical Research Phenomenon Showcase
          </h3>
          <span className="text-[11px] font-mono text-slate-400">Multi3Hate Empirical Benchmark Data</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {empiricalShowcase.map((cs) => {
            const isInversion = cs.type === 'CONTEXTUAL_INVERSION';
            return (
              <div
                key={cs.id}
                className={`p-5 rounded-2xl glass-card border ${
                  isInversion ? 'border-emerald-500/30 hover:border-emerald-500/50' : 'border-rose-500/30 hover:border-rose-500/50'
                } space-y-4`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                    isInversion ? 'cyber-badge-safe' : 'cyber-badge-high'
                  }`}>
                    {cs.type.replace(/_/g, ' ')}
                  </span>
                  <span className="text-xs font-mono text-slate-400">Benchmark ID #{cs.id}</span>
                </div>

                <div className="p-3 rounded-lg bg-dark-900 border border-slate-800 font-mono text-xs text-slate-300 line-clamp-2">
                  "{cs.text}"
                </div>

                {/* Score Transition Flow */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-dark-950/80 border border-slate-800/80 text-xs font-mono">
                  <div>
                    <span className="text-[10px] text-slate-500 block">TEXT ALONE</span>
                    <span className="text-sm font-bold text-slate-300">
                      {(cs.textScore * 100).toFixed(1)}%
                    </span>
                  </div>

                  <ArrowRight className="w-4 h-4 text-cyan-400" />

                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">MULTIMODAL FUSION</span>
                    <span className={`text-sm font-bold ${isInversion ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {(cs.multimodalScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <p className="text-xs text-slate-400 leading-relaxed">
                  {cs.explanation}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
export default MemePage;
