import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Sparkles, 
  ShieldAlert, 
  ShieldCheck, 
  AlertTriangle, 
  FileText, 
  Layers, 
  CheckCircle2, 
  HelpCircle,
  Cpu,
  Upload,
  ArrowRight,
  Globe,
  Link2,
  ExternalLink,
  MessageSquare,
  PlusCircle,
  Check,
  Filter
} from 'lucide-react';
import { analyzeText, analyzeMeme, analyzeBatch, analyzeSocialUrl } from '../api/client';
import { TextAnalysisResult, MemeAnalysisResult, ModerationQueueItem, SocialAnalysisResponse } from '../types';

interface AnalyzePageProps {
  onAddToQueue: (item: ModerationQueueItem) => void;
}

export function AnalyzePage({ onAddToQueue }: AnalyzePageProps) {
  const [activeTab, setActiveTab] = useState<'text' | 'meme' | 'batch' | 'social'>('text');
  
  // Text Analysis States
  const [inputText, setInputText] = useState('');
  const [enableJev, setEnableJev] = useState(true);
  const [loading, setLoading] = useState(false);
  const [textResult, setTextResult] = useState<TextAnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Meme Analysis States
  const [memeFile, setMemeFile] = useState<File | null>(null);
  const [memePreview, setMemePreview] = useState<string | null>(null);
  const [memeCaption, setMemeCaption] = useState('');
  const [memeResult, setMemeResult] = useState<MemeAnalysisResult | null>(null);

  // Batch CSV States
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchResults, setBatchResults] = useState<any | null>(null);

  // Social Media Link States
  const [socialUrl, setSocialUrl] = useState('');
  const [socialPlatform, setSocialPlatform] = useState<'auto' | 'youtube' | 'reddit' | 'twitter'>('auto');
  const [socialResults, setSocialResults] = useState<SocialAnalysisResponse | null>(null);
  const [socialFilter, setSocialFilter] = useState<'ALL' | 'HIGH_RISK' | 'REVIEW' | 'SAFE'>('ALL');
  const [addedSocialIds, setAddedSocialIds] = useState<Record<string, boolean>>({});

  const sampleTexts = [
    { label: "Harmless English", text: "Great match today! The team really pulled through in the second half." },
    { label: "Hindi Colloquial", text: "bhai kya bawaal movie hai, maza aa gaya ekdum" },
    { label: "Contextual Slang", text: "Bro you totally killed that presentation today, insane performance" },
    { label: "Hostile Threat", text: "I will find where you live and destroy you, watch your back." },
    { label: "Hate Speech", text: "These people are pests who do not belong in our country and should be eliminated." },
  ];

  const handleAnalyzeText = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setErrorMsg(null);
    setTextResult(null);

    try {
      const res = await analyzeText(inputText, undefined, enableJev);
      setTextResult(res);
      
      // Auto-add to session queue
      onAddToQueue({
        id: `txt_${Date.now()}`,
        content: res.text,
        type: 'TEXT',
        language: res.language,
        risk_level: res.risk_level,
        score: res.confidence,
        jev_status: res.jev.invoked ? 'ADJUDICATED' : 'BYPASSED',
        status: 'PENDING',
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (err: any) {
      setErrorMsg(err.message || 'Analysis temporarily unavailable. Please verify API connection.');
    } finally {
      setLoading(false);
    }
  };

  const handleMemeUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setMemeFile(file);
      setMemePreview(URL.createObjectURL(file));
      setMemeResult(null);
    }
  };

  const handleAnalyzeMeme = async () => {
    if (!memeFile) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await analyzeMeme(memeFile, memeCaption);
      setMemeResult(res);
      
      onAddToQueue({
        id: `meme_${Date.now()}`,
        content: memeCaption || '[Meme Image Analyzed]',
        imageUrl: memePreview || undefined,
        type: 'MEME',
        language: res.language,
        risk_level: res.risk_level,
        score: res.multimodal_score,
        jev_status: 'MULTIMODAL_FUSION',
        status: 'PENDING',
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (err: any) {
      setErrorMsg(err.message || 'Meme analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeSocial = async () => {
    if (!socialUrl.trim()) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await analyzeSocialUrl(socialUrl, socialPlatform, 50, true, enableJev);
      setSocialResults(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to fetch and analyze comments from social URL.');
    } finally {
      setLoading(false);
    }
  };

  const handleAddSocialItemToQueue = (item: any) => {
    onAddToQueue({
      id: item.id || `soc_${Date.now()}`,
      content: item.text,
      type: 'SOCIAL',
      language: item.language,
      risk_level: item.risk_level,
      score: item.confidence,
      jev_status: item.is_ambiguous ? 'ADJUDICATED' : 'BYPASSED',
      status: 'PENDING',
      timestamp: new Date().toLocaleTimeString(),
    });
    setAddedSocialIds(prev => ({ ...prev, [item.id]: true }));
  };

  const handleAddAllHighRiskSocial = () => {
    if (!socialResults?.comments) return;
    socialResults.comments
      .filter(c => c.risk_level === 'High Risk' && !addedSocialIds[c.id])
      .forEach(c => handleAddSocialItemToQueue(c));
  };

  const categoryLabels: Record<string, { label: string; desc: string }> = {
    toxic: { label: 'Toxicity', desc: 'General rude, disrespectful, or unreasonable language' },
    severe_toxic: { label: 'Severe Toxicity', desc: 'Extremely aggressive, hateful, or harmful speech' },
    obscene: { label: 'Obscenity', desc: 'Vulgar, profane, or sexually explicit terms' },
    threat: { label: 'Threat', desc: 'Intent to inflict harm, violence, or damage' },
    insult: { label: 'Insult', desc: 'Targeted disrespect, belittlement, or mockery' },
    identity_hate: { label: 'Identity Hate', desc: 'Hate targeted at protected groups or demographics' }
  };

  return (
    <div className="space-y-8 pb-12 max-w-6xl mx-auto">
      {/* 1. Mode Navigation Tabs */}
      <div className="flex space-x-2 border-b border-slate-800 pb-3">
        <button
          onClick={() => setActiveTab('text')}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl font-mono text-xs uppercase tracking-wider font-bold transition ${
            activeTab === 'text'
              ? 'bg-cyan-950 text-cyan-300 border border-cyan-500/40 shadow-glow-cyan/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>Multilingual Text</span>
        </button>

        <button
          onClick={() => setActiveTab('meme')}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl font-mono text-xs uppercase tracking-wider font-bold transition ${
            activeTab === 'meme'
              ? 'bg-purple-950 text-purple-300 border border-purple-500/40 shadow-glow-purple/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Meme / Multimodal</span>
        </button>

        <button
          onClick={() => setActiveTab('batch')}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl font-mono text-xs uppercase tracking-wider font-bold transition ${
            activeTab === 'batch'
              ? 'bg-blue-950 text-blue-300 border border-blue-500/40'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <Upload className="w-4 h-4" />
          <span>Batch Ingestion</span>
        </button>

        <button
          onClick={() => setActiveTab('social')}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl font-mono text-xs uppercase tracking-wider font-bold transition ${
            activeTab === 'social'
              ? 'bg-rose-950 text-rose-300 border border-rose-500/40 shadow-glow-rose/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <Globe className="w-4 h-4" />
          <span>Social Media Links</span>
        </button>
      </div>

      {/* 2. TEXT ANALYSIS MODE */}
      {activeTab === 'text' && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                Raw Comment or Post Text
              </label>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="jev_toggle"
                  checked={enableJev}
                  onChange={(e) => setEnableJev(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-cyan-500 cursor-pointer"
                />
                <label htmlFor="jev_toggle" className="text-xs font-mono text-slate-300 cursor-pointer flex items-center gap-1">
                  <span>Enable JEV Contextual Adjudication</span>
                  <span className="text-[10px] text-amber-400 bg-amber-950/60 px-1.5 py-0.2 rounded border border-amber-500/30">ASSIST</span>
                </label>
              </div>
            </div>

            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste social media content in English, Hindi, or 15+ Indic languages..."
              rows={4}
              className="w-full glass-input rounded-xl p-4 text-sm leading-relaxed placeholder-slate-500 focus:ring-1 focus:ring-cyan-400 resize-none font-sans"
            />

            {/* Quick Test Samples */}
            <div className="flex flex-wrap gap-2 pt-1 items-center">
              <span className="text-xs font-mono text-slate-400 mr-2">Presets:</span>
              {sampleTexts.map((s) => (
                <button
                  key={s.label}
                  onClick={() => setInputText(s.text)}
                  className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-dark-900 border border-slate-800 text-slate-300 hover:border-cyan-500/50 hover:text-cyan-300 transition"
                >
                  {s.label}
                </button>
              ))}
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={handleAnalyzeText}
                disabled={loading || !inputText.trim()}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-dark-950 font-bold font-mono text-xs uppercase tracking-wider flex items-center space-x-2 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-glow-cyan/40"
              >
                {loading ? (
                  <>
                    <span className="w-4 h-4 border-2 border-dark-950 border-t-transparent rounded-full animate-spin"></span>
                    <span>Analyzing Neural Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Run Toxicity Analysis</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {errorMsg && (
            <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-300 text-xs font-mono">
              {errorMsg}
            </div>
          )}

          {/* Inference Results View */}
          <AnimatePresence>
            {textResult && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-6"
              >
                {/* 1. Neural Pipeline Trace Indicator */}
                <div className="p-4 rounded-xl glass-panel border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
                  <div className="flex items-center space-x-2">
                    <span className="text-slate-400 uppercase">PIPELINE EXECUTION:</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {textResult.pipeline_trace.map((step, idx) => (
                      <div key={step.stage} className="flex items-center space-x-2">
                        <span className="px-2.5 py-1 rounded bg-slate-900 border border-cyan-500/30 text-cyan-300 text-[11px] flex items-center gap-1.5">
                          <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                          {step.stage}
                          <span className="text-[10px] text-slate-500">({step.duration_ms}ms)</span>
                        </span>
                        {idx < textResult.pipeline_trace.length - 1 && (
                          <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* 2. Primary Decision Banner */}
                <div className={`p-6 rounded-2xl glass-panel border flex flex-col md:flex-row items-start md:items-center justify-between gap-6 ${
                  textResult.risk_level === 'High Risk'
                    ? 'border-rose-500/40 bg-rose-950/15 shadow-glow-rose/20'
                    : textResult.risk_level === 'Review'
                    ? 'border-amber-500/40 bg-amber-950/15 shadow-glow-amber/20'
                    : 'border-emerald-500/40 bg-emerald-950/15 shadow-glow-emerald/20'
                }`}>
                  <div className="flex items-center space-x-4">
                    <div className={`w-14 h-14 rounded-2xl flex items-center justify-center font-bold text-2xl ${
                      textResult.risk_level === 'High Risk'
                        ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                        : textResult.risk_level === 'Review'
                        ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                        : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                    }`}>
                      {textResult.risk_level === 'High Risk' ? <ShieldAlert className="w-8 h-8" /> : textResult.risk_level === 'Review' ? <AlertTriangle className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />}
                    </div>
                    <div>
                      <div className="flex items-center space-x-3">
                        <span className={`px-3 py-1 rounded-md text-sm font-mono font-extrabold uppercase tracking-wider ${
                          textResult.risk_level === 'High Risk'
                            ? 'cyber-badge-high'
                            : textResult.risk_level === 'Review'
                            ? 'cyber-badge-review'
                            : 'cyber-badge-safe'
                        }`}>
                          {textResult.risk_level}
                        </span>
                        <span className="text-xs font-mono text-slate-400 bg-dark-900 px-2 py-0.5 rounded border border-slate-800">
                          {textResult.language}
                        </span>
                      </div>
                      <p className="text-sm text-slate-300 mt-2 font-medium">
                        {textResult.explanation}
                      </p>
                    </div>
                  </div>

                  <div className="text-right flex md:flex-col items-center md:items-end justify-between w-full md:w-auto font-mono">
                    <span className="text-xs text-slate-400 uppercase">Max Probability</span>
                    <span className="text-3xl font-extrabold tracking-tight text-slate-100">
                      {(textResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* 3. Six Toxicity Dimension Scores */}
                <div className="p-6 rounded-2xl glass-panel border border-slate-800 space-y-4">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold flex items-center justify-between">
                    <span>Multi-Label Probability Vector (MuRIL)</span>
                    <span className="text-[11px] text-slate-500">Validation-Optimized Decision Thresholds</span>
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(textResult.scores).map(([cat, score]) => {
                      const th = textResult.thresholds[cat as keyof typeof textResult.thresholds] || 0.5;
                      const isFlagged = textResult.is_toxic_flags[cat as keyof typeof textResult.is_toxic_flags];
                      const meta = categoryLabels[cat] || { label: cat, desc: '' };
                      return (
                        <div
                          key={cat}
                          className={`p-4 rounded-xl glass-card border ${
                            isFlagged ? 'border-rose-500/40 bg-rose-950/10' : 'border-slate-800/80 bg-dark-900/40'
                          } space-y-2.5`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold text-slate-100">{meta.label}</span>
                            <span className={`text-[11px] font-mono font-bold px-2 py-0.5 rounded ${
                              isFlagged ? 'bg-rose-950 text-rose-300 border border-rose-500/40' : 'bg-slate-900 text-slate-400'
                            }`}>
                              {isFlagged ? 'TRIGGERED' : 'CLEAR'}
                            </span>
                          </div>

                          {/* Progress Bar with Threshold Marker */}
                          <div className="relative pt-1">
                            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min(score * 100, 100)}%` }}
                                transition={{ duration: 0.5, ease: 'easeOut' }}
                                className={`h-full rounded-full ${
                                  isFlagged ? 'bg-gradient-to-r from-amber-500 to-rose-500' : 'bg-gradient-to-r from-cyan-500 to-blue-500'
                                }`}
                              />
                            </div>
                            {/* Threshold Marker Pin */}
                            <div 
                              className="absolute top-0 w-0.5 h-3.5 bg-amber-400/80 -translate-x-1/2"
                              style={{ left: `${th * 100}%` }}
                              title={`Threshold: ${(th * 100).toFixed(0)}%`}
                            />
                          </div>

                          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
                            <span>Score: <strong className="text-slate-200">{(score * 100).toFixed(1)}%</strong></span>
                            <span>Threshold: <strong className="text-amber-400">{(th * 100).toFixed(0)}%</strong></span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* 4. JEV Contextual Adjudication Card */}
                {textResult.jev.invoked && textResult.jev.decision && (
                  <div className="p-6 rounded-2xl glass-panel border border-amber-500/30 bg-amber-950/10 space-y-4 shadow-glow-amber/10">
                    <div className="flex items-center justify-between pb-3 border-b border-amber-500/20">
                      <div className="flex items-center space-x-2.5">
                        <Sparkles className="w-5 h-5 text-amber-400" />
                        <h4 className="text-sm font-mono font-bold uppercase tracking-wider text-amber-300">
                          JEV Contextual Adjudication Result
                        </h4>
                      </div>
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-500/30">
                        {textResult.jev.decision.provider_type}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
                      <div className="p-3 rounded-xl bg-dark-900/60 border border-slate-800">
                        <span className="text-slate-500">INFERRED INTENT</span>
                        <p className="text-slate-200 font-bold text-sm mt-1 uppercase">
                          {textResult.jev.decision.intent.replace(/_/g, ' ')}
                        </p>
                      </div>

                      <div className="p-3 rounded-xl bg-dark-900/60 border border-slate-800">
                        <span className="text-slate-500">CONTEXTUAL TOXICITY</span>
                        <p className={`font-bold text-sm mt-1 uppercase ${textResult.jev.decision.contextual_toxicity ? 'text-rose-400' : 'text-emerald-400'}`}>
                          {textResult.jev.decision.contextual_toxicity ? 'CONFIRMED TOXIC' : 'BENIGN / HARLESS'}
                        </p>
                      </div>

                      <div className="p-3 rounded-xl bg-dark-900/60 border border-slate-800">
                        <span className="text-slate-500">HUMAN REVIEW REQUIRED</span>
                        <p className="text-slate-200 font-bold text-sm mt-1 uppercase">
                          {textResult.jev.decision.human_review_required ? 'YES (ESCALATED)' : 'NO (RESOLVED)'}
                        </p>
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-dark-950/80 border border-slate-800/80 text-xs text-slate-300 leading-relaxed font-sans">
                      <strong className="font-mono text-amber-400 text-xs block mb-1">Reasoning Analysis:</strong>
                      {textResult.jev.decision.reason}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* 3. MEME ANALYSIS MODE */}
      {activeTab === 'meme' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-stretch">
            {/* Upload Zone */}
            <div className="md:col-span-5 glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col justify-between space-y-4">
              <div>
                <label className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold flex items-center gap-2 mb-3">
                  <Upload className="w-4 h-4 text-purple-400" />
                  Meme Image Upload
                </label>

                <div className="border-2 border-dashed border-slate-700 hover:border-purple-500/60 rounded-2xl p-6 text-center cursor-pointer transition relative bg-dark-900/30">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleMemeUpload}
                    className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                  />
                  {memePreview ? (
                    <div className="relative group">
                      <img src={memePreview} alt="Meme Preview" className="max-h-60 mx-auto rounded-lg object-contain shadow-lg" />
                      <span className="text-[11px] font-mono text-purple-400 mt-2 block">Click to change image</span>
                    </div>
                  ) : (
                    <div className="space-y-2 py-6">
                      <Layers className="w-10 h-10 text-slate-500 mx-auto" />
                      <p className="text-xs text-slate-300 font-medium">Click or drag meme image here</p>
                      <p className="text-[10px] text-slate-500 font-mono">PNG, JPG, WEBP supported</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-3">
                <label className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold block">
                  Optional Overlay Text / Caption
                </label>
                <input
                  type="text"
                  value={memeCaption}
                  onChange={(e) => setMemeCaption(e.target.value)}
                  placeholder="Enter meme text if not detected via OCR..."
                  className="w-full glass-input rounded-xl px-4 py-2.5 text-xs font-sans placeholder-slate-500 focus:ring-1 focus:ring-purple-400"
                />
              </div>

              <button
                onClick={handleAnalyzeMeme}
                disabled={loading || !memeFile}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-400 hover:to-indigo-500 text-dark-950 font-bold font-mono text-xs uppercase tracking-wider transition disabled:opacity-50 disabled:cursor-not-allowed shadow-glow-purple/40"
              >
                {loading ? 'Evaluating Multimodal Tensors...' : 'Run Multimodal Meme Analysis'}
              </button>
            </div>

            {/* Multimodal Results Card */}
            <div className="md:col-span-7 glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col justify-between">
              {memeResult ? (
                <div className="space-y-6">
                  {/* Decision Banner */}
                  <div className={`p-4 rounded-xl border flex items-center justify-between ${
                    memeResult.is_hateful
                      ? 'border-rose-500/40 bg-rose-950/20'
                      : 'border-emerald-500/40 bg-emerald-950/20'
                  }`}>
                    <div>
                      <span className={`text-xs font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded ${
                        memeResult.is_hateful ? 'cyber-badge-high' : 'cyber-badge-safe'
                      }`}>
                        {memeResult.is_hateful ? 'HATEFUL MEME DETECTED' : 'SAFE MEME'}
                      </span>
                      <p className="text-xs text-slate-300 font-mono mt-1.5">
                        {memeResult.interaction_description}
                      </p>
                    </div>
                    <div className="text-right font-mono">
                      <span className="text-[10px] text-slate-400 block">FUSION SCORE</span>
                      <span className="text-2xl font-bold text-slate-100">
                        {(memeResult.multimodal_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Modality Interaction Breakdown */}
                  <div className="grid grid-cols-3 gap-3 font-mono text-xs text-center">
                    <div className="p-3 rounded-xl bg-dark-900 border border-slate-800">
                      <span className="text-slate-500 text-[10px]">TEXT SIGNAL</span>
                      <p className="text-slate-200 font-bold text-sm mt-1">
                        {(memeResult.text_only_score * 100).toFixed(1)}%
                      </p>
                    </div>

                    <div className="p-3 rounded-xl bg-dark-900 border border-slate-800">
                      <span className="text-slate-500 text-[10px]">VISION SIGNAL</span>
                      <p className="text-slate-200 font-bold text-sm mt-1">
                        {(memeResult.image_only_score * 100).toFixed(1)}%
                      </p>
                    </div>

                    <div className="p-3 rounded-xl bg-dark-900 border border-purple-500/30">
                      <span className="text-purple-400 text-[10px]">FUSED [t*v, |t-v|]</span>
                      <p className="text-purple-300 font-bold text-sm mt-1">
                        {(memeResult.multimodal_score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Interaction Type Badge */}
                  <div className="p-3.5 rounded-xl bg-dark-900/60 border border-slate-800 text-xs">
                    <span className="font-mono text-slate-400 text-[11px] uppercase block mb-1">
                      Cross-Modal Semantic Relationship:
                    </span>
                    <span className="font-mono text-sm font-bold text-cyan-300">
                      {memeResult.interaction_type}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-500">
                  <Layers className="w-12 h-12 text-slate-600 mb-3" />
                  <p className="text-sm font-medium text-slate-300">Awaiting Meme Upload</p>
                  <p className="text-xs text-slate-400 mt-1 max-w-sm">
                    Upload a meme and run the model to observe cross-modal fusion outputs between MuRIL and CLIP.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 4. BATCH ANALYSIS MODE */}
      {activeTab === 'batch' && (
        <div className="glass-panel p-8 rounded-2xl border border-slate-800 text-center space-y-4">
          <Upload className="w-10 h-10 text-cyan-400 mx-auto" />
          <h3 className="text-sm font-mono uppercase tracking-wider text-slate-200 font-bold">
            Bulk Social Media Ingestion
          </h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Upload CSV with a 'text' column for high-throughput batch toxicity triage with aggregate telemetry.
          </p>
          <div className="pt-2">
            <button
              onClick={() => {
                // Run synthetic demonstration batch with real project samples
                const demoItems = [
                  { id: '1', text: 'You are so talented, totally crushed it today!' },
                  { id: '2', text: 'bhai kya bawaal hai ekdum' },
                  { id: '3', text: 'I will hunt you down and destroy your family.' },
                  { id: '4', text: 'These immigrants are filthy animals that ruin everything.' },
                  { id: '5', text: 'Bro this sick video literally killed me of laughter!' },
                ];
                setLoading(true);
                analyzeBatch(demoItems).then((res) => {
                  setBatchResults(res);
                  setLoading(false);
                });
              }}
              className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono text-xs uppercase tracking-wider transition"
            >
              {loading ? 'Processing Batch...' : 'Run Quick Demonstration Batch (5 Samples)'}
            </button>
          </div>

          {batchResults && (
            <div className="mt-6 text-left space-y-4 border-t border-slate-800 pt-4">
              <div className="grid grid-cols-4 gap-3 text-xs font-mono text-center">
                <div className="p-3 bg-dark-900 rounded-lg border border-slate-800">
                  <span className="text-slate-500">TOTAL</span>
                  <p className="text-lg font-bold text-slate-200">{batchResults.total_analyzed}</p>
                </div>
                <div className="p-3 bg-dark-900 rounded-lg border border-rose-500/30">
                  <span className="text-rose-400">HIGH RISK</span>
                  <p className="text-lg font-bold text-rose-300">{batchResults.high_risk_count}</p>
                </div>
                <div className="p-3 bg-dark-900 rounded-lg border border-amber-500/30">
                  <span className="text-amber-400">REVIEW</span>
                  <p className="text-lg font-bold text-amber-300">{batchResults.review_count}</p>
                </div>
                <div className="p-3 bg-dark-900 rounded-lg border border-emerald-500/30">
                  <span className="text-emerald-400">SAFE</span>
                  <p className="text-lg font-bold text-emerald-300">{batchResults.safe_count}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 5. SOCIAL MEDIA URL INGESTION MODE */}
      {activeTab === 'social' && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <label className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold flex items-center gap-2">
                  <Globe className="w-4 h-4 text-rose-400" />
                  Live Social Media Stream Ingestion
                </label>
                <p className="text-xs text-slate-400 mt-1">
                  Fetch and moderate comments directly from YouTube videos, Reddit discussions, or Twitter/X tweets.
                </p>
              </div>

              {/* Platform Selector */}
              <div className="flex items-center space-x-1 p-1 bg-dark-900 rounded-xl border border-slate-800">
                {(['auto', 'youtube', 'reddit', 'twitter'] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setSocialPlatform(p)}
                    className={`px-3 py-1 rounded-lg text-[11px] font-mono uppercase transition ${
                      socialPlatform === p
                        ? 'bg-rose-950 text-rose-300 border border-rose-500/40 font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {p === 'auto' ? '⚡ Auto' : p === 'twitter' ? 'Twitter (X)' : p}
                  </button>
                ))}
              </div>
            </div>

            {/* URL Input Form */}
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Link2 className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="url"
                  value={socialUrl}
                  onChange={(e) => setSocialUrl(e.target.value)}
                  placeholder="Paste YouTube, Reddit, or Twitter/X URL..."
                  className="w-full glass-input rounded-xl pl-10 pr-4 py-3 text-sm placeholder-slate-500 focus:ring-1 focus:ring-rose-400 font-sans"
                />
              </div>

              <button
                onClick={handleAnalyzeSocial}
                disabled={loading || !socialUrl.trim()}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-400 hover:to-pink-500 text-dark-950 font-bold font-mono text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-glow-rose/40"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-dark-950 border-t-transparent rounded-full animate-spin" />
                    <span>Fetching & Triaging...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Fetch & Moderate</span>
                  </>
                )}
              </button>
            </div>

            {/* Presets */}
            <div className="flex flex-wrap gap-2 pt-1 items-center">
              <span className="text-xs font-mono text-slate-400 mr-2">Quick Presets:</span>
              <button
                onClick={() => {
                  setSocialPlatform('youtube');
                  setSocialUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ');
                }}
                className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-dark-900 border border-slate-800 text-slate-300 hover:border-rose-500/50 hover:text-rose-300 transition flex items-center gap-1.5"
              >
                <span>📺 YouTube Video</span>
              </button>
              <button
                onClick={() => {
                  setSocialPlatform('reddit');
                  setSocialUrl('https://www.reddit.com/r/technology/comments/multilingual_ai_debate');
                }}
                className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-dark-900 border border-slate-800 text-slate-300 hover:border-orange-500/50 hover:text-orange-300 transition flex items-center gap-1.5"
              >
                <span>🤖 Reddit Thread</span>
              </button>
              <button
                onClick={() => {
                  setSocialPlatform('twitter');
                  setSocialUrl('https://twitter.com/social_media/status/1234567890');
                }}
                className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-dark-900 border border-slate-800 text-slate-300 hover:border-cyan-500/50 hover:text-cyan-300 transition flex items-center gap-1.5"
              >
                <span>🐦 Twitter / X Status</span>
              </button>
            </div>
          </div>

          {/* Social Ingestion Results */}
          {socialResults && (
            <div className="space-y-6">
              {/* Notice Banner */}
              {socialResults.is_demo && socialResults.message && (
                <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-500/30 text-amber-200 text-xs flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    <span>{socialResults.message}</span>
                  </div>
                  <span className="text-[10px] font-mono bg-amber-900/60 text-amber-300 px-2 py-0.5 rounded border border-amber-500/30">
                    PREVIEW MODE
                  </span>
                </div>
              )}

              {/* Summary Stats Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
                <div className="p-3 bg-dark-900 rounded-xl border border-slate-800 text-center">
                  <span className="text-slate-500 text-[10px] block">COMMENTS FETCHED</span>
                  <p className="text-xl font-bold text-slate-200 mt-1">{socialResults.total_comments}</p>
                </div>
                <div className="p-3 bg-dark-900 rounded-xl border border-rose-500/30 text-center">
                  <span className="text-rose-400 text-[10px] block">HIGH RISK</span>
                  <p className="text-xl font-bold text-rose-300 mt-1">{socialResults.high_risk_count}</p>
                </div>
                <div className="p-3 bg-dark-900 rounded-xl border border-amber-500/30 text-center">
                  <span className="text-amber-400 text-[10px] block">REVIEW NEEDED</span>
                  <p className="text-xl font-bold text-amber-300 mt-1">{socialResults.review_count}</p>
                </div>
                <div className="p-3 bg-dark-900 rounded-xl border border-emerald-500/30 text-center">
                  <span className="text-emerald-400 text-[10px] block">SAFE CONTENT</span>
                  <p className="text-xl font-bold text-emerald-300 mt-1">{socialResults.safe_count}</p>
                </div>
                <div className="p-3 bg-dark-900 rounded-xl border border-cyan-500/30 text-center">
                  <span className="text-cyan-400 text-[10px] block">LATENCY (CPU)</span>
                  <p className="text-xl font-bold text-cyan-300 mt-1">{socialResults.latency_ms} ms</p>
                </div>
              </div>

              {/* Filter & Batch Actions Bar */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-4 bg-dark-900/80 rounded-xl border border-slate-800">
                <div className="flex items-center space-x-2 text-xs font-mono">
                  <Filter className="w-3.5 h-3.5 text-slate-400" />
                  <span className="text-slate-400">Filter:</span>
                  {(['ALL', 'HIGH_RISK', 'REVIEW', 'SAFE'] as const).map((f) => (
                    <button
                      key={f}
                      onClick={() => setSocialFilter(f)}
                      className={`px-2.5 py-1 rounded-lg transition text-[11px] ${
                        socialFilter === f
                          ? 'bg-slate-700 text-white font-bold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {f === 'ALL' ? 'All' : f === 'HIGH_RISK' ? 'High Risk' : f === 'REVIEW' ? 'Review' : 'Safe'}
                    </button>
                  ))}
                </div>

                <button
                  onClick={handleAddAllHighRiskSocial}
                  disabled={socialResults.high_risk_count === 0}
                  className="px-4 py-2 rounded-lg bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-500/30 font-mono text-xs flex items-center space-x-1.5 transition disabled:opacity-40"
                >
                  <PlusCircle className="w-3.5 h-3.5" />
                  <span>Send All High Risk to Queue</span>
                </button>
              </div>

              {/* Comments Feed */}
              <div className="space-y-3">
                {socialResults.comments
                  .filter((c) => {
                    if (socialFilter === 'HIGH_RISK') return c.risk_level === 'High Risk';
                    if (socialFilter === 'REVIEW') return c.risk_level === 'Review';
                    if (socialFilter === 'SAFE') return c.risk_level === 'Safe';
                    return true;
                  })
                  .map((comment) => {
                    const isAdded = !!addedSocialIds[comment.id];
                    return (
                      <div
                        key={comment.id}
                        className={`p-4 rounded-xl border transition ${
                          comment.risk_level === 'High Risk'
                            ? 'bg-rose-950/20 border-rose-500/40'
                            : comment.risk_level === 'Review'
                            ? 'bg-amber-950/20 border-amber-500/40'
                            : 'bg-dark-900/60 border-slate-800'
                        }`}
                      >
                        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                          <div className="space-y-2 flex-1">
                            <div className="flex items-center space-x-2">
                              <span
                                className={`text-[11px] font-mono px-2 py-0.5 rounded font-bold uppercase ${
                                  comment.risk_level === 'High Risk'
                                    ? 'bg-rose-950 text-rose-300 border border-rose-500/40'
                                    : comment.risk_level === 'Review'
                                    ? 'bg-amber-950 text-amber-300 border border-amber-500/40'
                                    : 'bg-emerald-950 text-emerald-300 border border-emerald-500/40'
                                }`}
                              >
                                {comment.badge_icon} {comment.risk_level}
                              </span>
                              <span className="text-[11px] font-mono text-slate-400 bg-dark-950 px-2 py-0.5 rounded border border-slate-800">
                                {comment.language}
                              </span>
                              <span className="text-[11px] font-mono text-slate-500">
                                Confidence: {(comment.confidence * 100).toFixed(1)}%
                              </span>
                            </div>

                            <p className="text-sm text-slate-200 leading-relaxed font-sans">
                              {comment.text}
                            </p>

                            <p className="text-xs text-slate-400 italic">
                              {comment.explanation}
                            </p>

                            {/* Active flags */}
                            <div className="flex flex-wrap gap-1.5 pt-1">
                              {Object.entries(comment.is_toxic_flags).map(([label, flag]) => (
                                <span
                                  key={label}
                                  className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                                    flag
                                      ? 'bg-rose-950/80 text-rose-300 border border-rose-500/40 font-bold'
                                      : 'bg-dark-950 text-slate-600 border border-slate-800'
                                  }`}
                                >
                                  {label}: {((comment.scores[label as keyof typeof comment.scores] || 0) * 100).toFixed(0)}%
                                </span>
                              ))}
                            </div>
                          </div>

                          <div className="shrink-0 flex sm:flex-col items-end gap-2">
                            <button
                              onClick={() => handleAddSocialItemToQueue(comment)}
                              disabled={isAdded}
                              className={`px-3 py-1.5 rounded-lg text-xs font-mono uppercase flex items-center space-x-1 transition ${
                                isAdded
                                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-500/30'
                                  : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                              }`}
                            >
                              {isAdded ? (
                                <>
                                  <Check className="w-3.5 h-3.5" />
                                  <span>In Queue</span>
                                </>
                              ) : (
                                <>
                                  <PlusCircle className="w-3.5 h-3.5" />
                                  <span>Queue</span>
                                </>
                              )}
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {!socialResults && (
            <div className="text-center p-8 rounded-2xl border border-slate-800/60 bg-dark-900/30 space-y-3">
              <MessageSquare className="w-10 h-10 text-slate-600 mx-auto" />
              <p className="text-sm font-medium text-slate-300">Ready for Social Media Ingestion</p>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                Paste any YouTube video link, Reddit submission URL, or Twitter/X post to scrape and evaluate top-level comments against our multilingual MuRIL transformer.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
export default AnalyzePage;

