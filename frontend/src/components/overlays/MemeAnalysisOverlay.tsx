import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MemeAnalysisResult } from '../../types';
import { analyzeMeme, analyzeMemeBase64 } from '../../api/client';

interface MemeAnalysisOverlayProps {
  onResult: (res: MemeAnalysisResult, imageUrl: string) => void;
  result: MemeAnalysisResult | null;
  isAnalyzing: boolean;
  setIsAnalyzing: (v: boolean) => void;
}

// Built-in curated meme presets for instant testing
const MEME_PRESETS = [
  {
    label: 'Case A: Contextual Inversion',
    text: 'bring your dog we have chicken?',
    description: 'Lexical hostility neutralized by benign puppy picnic image',
    type: 'INVERSION',
  },
  {
    label: 'Case B: Emergent Toxicity',
    text: 'SHUT UP OSAMA! HE IS MY SON!',
    description: 'Innocuous wording weaponized by anti-minority visual template',
    type: 'EMERGENT',
  },
  {
    label: 'Case C: Hindi OCR Slang',
    text: 'ANKUR TU CHUTIYA HA EK HI DEGREE DO BAAR KARNE WAALE',
    description: 'Colloquial banter / vulgarity without directed hate',
    type: 'BANTER',
  },
  {
    label: 'Case D: Benign Culture Meme',
    text: 'When chai is ready but friends are still arguing about cricket',
    description: 'Pure cultural humor with zero identity hate',
    type: 'CLEAN',
  },
];

export const MemeAnalysisOverlay: React.FC<MemeAnalysisOverlayProps> = ({
  onResult,
  result,
  isAnalyzing,
  setIsAnalyzing,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customText, setCustomText] = useState(MEME_PRESETS[0].text);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (!result && !isAnalyzing) {
      handleRunMemeAnalysis(undefined, MEME_PRESETS[0].text);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleRunMemeAnalysis = async (fileToUse?: File, textToUse?: string) => {
    const file = fileToUse || selectedFile;
    const text = textToUse !== undefined ? textToUse : customText;

    if (!file && !text) return;
    setIsAnalyzing(true);
    setError(null);

    try {
      if (file) {
        const res = await analyzeMeme(file, text);
        const url = URL.createObjectURL(file);
        onResult(res, url);
      } else {
        // Fallback demo mock canvas for text preset if no physical image uploaded
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(0, 0, 512, 512);
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 6;
          ctx.strokeRect(20, 20, 472, 472);
          ctx.fillStyle = '#f8fafc';
          ctx.font = 'bold 24px monospace';
          ctx.fillText('MEME EVIDENCE', 40, 100);
          ctx.fillStyle = '#94a3b8';
          ctx.font = '20px sans-serif';
          ctx.fillText(text.slice(0, 30), 40, 240);
        }
        const base64 = canvas.toDataURL('image/png');
        const res = await analyzeMemeBase64(base64, text);
        onResult(res, base64);
      }
    } catch (err: any) {
      setError(err?.message || 'Multimodal analysis failed. Ensure backend api.py is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between p-8 md:p-14 pt-24 overflow-y-auto">
      {/* Top Banner */}
      <div className="flex items-center justify-between">
        <div>
          <span className="font-mono text-xs tracking-widest text-[#38bdf8] uppercase">
            CHAMBER 03 // 4-WAY MULTIMODAL FUSION CORE
          </span>
          <h2 className="font-display font-bold text-3xl sm:text-4xl text-white mt-1">
            CROSS-MODAL TOXICITY INVESTIGATION
          </h2>
        </div>
      </div>

      {/* Main Investigation Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 my-auto pointer-events-auto">
        {/* Left: Upload & Presets Column */}
        <div className="lg:col-span-5 bg-[#0a0c10]/85 border border-white/10 p-6 rounded-2xl backdrop-blur-md">
          {/* File Upload Drop Area */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="w-full border-2 border-dashed border-white/15 hover:border-[#38bdf8]/50 rounded-xl p-5 text-center cursor-pointer transition-colors bg-white/[0.02]"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />
            {previewUrl ? (
              <div className="flex items-center gap-4 text-left">
                <img
                  src={previewUrl}
                  alt="Evidence"
                  className="w-14 h-14 object-cover rounded-lg border border-white/20"
                />
                <div className="font-mono text-xs text-white">
                  <span className="text-[#2dd4bf] block">IMAGE ATTACHED</span>
                  <span className="text-white/40 text-[10px]">CLICK TO REPLACE EVIDENCE</span>
                </div>
              </div>
            ) : (
              <div className="font-mono text-xs text-white/50">
                <span className="text-white block font-medium mb-1">UPLOAD MEME IMAGE</span>
                <span>PNG, JPG, WEBP // DRAG EVIDENCE HERE</span>
              </div>
            )}
          </div>

          {/* Optional OCR Text Input */}
          <div className="mt-4">
            <label className="font-mono text-[10px] text-white/40 tracking-wider uppercase block mb-1">
              OPTIONAL TEXT / OCR OVERRIDE
            </label>
            <input
              type="text"
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              placeholder="OCR extracted text..."
              className="w-full bg-[#050608] border border-white/10 rounded-lg p-2.5 text-xs font-sans text-white focus:outline-none focus:border-[#38bdf8]"
            />
          </div>

          {/* Curated Presets */}
          <div className="mt-4">
            <span className="font-mono text-[10px] text-white/40 uppercase block mb-2">
              CURATED RESEARCH PRESETS
            </span>
            <div className="space-y-1.5">
              {MEME_PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => {
                    setCustomText(p.text);
                    handleRunMemeAnalysis(undefined, p.text);
                  }}
                  className="w-full text-left p-2 rounded-lg border border-white/5 bg-white/[0.02] hover:bg-white/[0.06] transition-colors"
                >
                  <div className="flex items-center justify-between font-mono text-xs text-[#38bdf8]">
                    <span>{p.label}</span>
                    <span className="text-[10px] text-white/30">RUN</span>
                  </div>
                  <div className="font-sans text-[11px] text-white/60 truncate mt-0.5">
                    "{p.text}"
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Run Action */}
          <div className="mt-5 flex items-center justify-between border-t border-white/10 pt-4">
            <span className="font-mono text-[11px] text-white/40">
              MuRIL (768) + CLIP (512)
            </span>
            <button
              onClick={() => handleRunMemeAnalysis()}
              disabled={isAnalyzing || (!selectedFile && !customText.trim())}
              className="px-5 py-2.5 rounded-xl bg-[#38bdf8] hover:bg-[#0284c7] disabled:opacity-40 text-black font-mono font-semibold text-xs tracking-wider transition-all flex items-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <span className="w-3 h-3 rounded-full border-2 border-black border-t-transparent animate-spin" />
                  <span>FUSING STREAMS...</span>
                </>
              ) : (
                <span>FUSE & ADJUDICATE →</span>
              )}
            </button>
          </div>

          {error && (
            <p className="mt-3 font-mono text-xs text-[#f43f5e] bg-[#f43f5e]/10 p-2.5 rounded-lg border border-[#f43f5e]/20">
              {error}
            </p>
          )}
        </div>

        {/* Right: Fusion Verdict & Interaction Decomposition */}
        <div className="lg:col-span-7 flex flex-col justify-center">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key={result.timestamp || result.text}
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.96 }}
                className="bg-[#0a0c10]/85 border border-white/10 p-6 rounded-2xl backdrop-blur-md"
              >
                {/* Large Editorial Verdict */}
                <div className="flex flex-wrap items-baseline justify-between gap-4 border-b border-white/10 pb-4 mb-4">
                  <div>
                    <span className="font-mono text-[10px] text-white/40 tracking-widest uppercase block">
                      MULTIMODAL DECISION
                    </span>
                    <h3 className={`font-display font-extrabold text-4xl sm:text-5xl ${
                      result.is_hateful ? 'text-[#f43f5e]' : 'text-[#2dd4bf]'
                    }`}>
                      {result.risk_level.toUpperCase()}
                    </h3>
                  </div>

                  <div className="text-right font-mono text-xs">
                    <span className="text-white/40 block">FUSION SCORE (T*={result.threshold.toFixed(2)})</span>
                    <span className="text-2xl font-bold text-white">
                      {(result.multimodal_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Interaction Type Callout */}
                <div className="p-3.5 rounded-xl border border-white/10 bg-white/5 mb-4">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-xs font-bold text-[#38bdf8]">
                      INTERACTION: {result.interaction_type.replace('_', ' ')}
                    </span>
                    <span className="font-mono text-[10px] text-white/40">
                      LATENCY: {result.latency_ms}MS
                    </span>
                  </div>
                  <p className="font-sans text-xs text-white/80 leading-relaxed">
                    {result.interaction_description}
                  </p>
                </div>

                {/* Modality Stream Breakdown */}
                <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                  <div className="p-3 rounded-xl border border-[#38bdf8]/20 bg-[#38bdf8]/5">
                    <span className="text-white/40 text-[10px] uppercase block">TEXT / OCR STREAM</span>
                    <span className="text-lg font-bold text-[#38bdf8]">
                      {(result.text_only_score * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-white/40 block mt-0.5">MuRIL (768-D)</span>
                  </div>

                  <div className="p-3 rounded-xl border border-[#f59e0b]/20 bg-[#f59e0b]/5">
                    <span className="text-white/40 text-[10px] uppercase block">VISUAL CONTEXT STREAM</span>
                    <span className="text-lg font-bold text-[#f59e0b]">
                      {(result.image_only_score * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-white/40 block mt-0.5">CLIP ViT-B/32 (512-D)</span>
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="p-8 border border-dashed border-white/10 rounded-2xl text-center text-white/40 font-mono text-xs">
                AWAITING MEME EVIDENCE FOR 4-WAY CROSS-MODAL FUSION...
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Bottom Technical Indicator */}
      <div className="flex items-center justify-between font-mono text-[11px] text-white/40 border-t border-white/[0.08] pt-4">
        <span>FUSION INTERACTION: [t_proj || v_proj || t*v || |t-v|]</span>
        <span>BENCHMARK CALIBRATED THRESHOLD: T* = 0.44</span>
      </div>
    </div>
  );
};
