import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  OFFICIAL_PRODUCTION_BASELINE,
  FINAL_CANDIDATE_RESULTS,
  EXPERIMENT_STATUS,
  EVALUATED_LANGUAGE_SLICES,
  SHUFFLED_IMAGE_RESULTS,
  ABLATION_STUDY_METRICS,
} from '../../data/researchMetrics';

export const ResearchOverlay: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'benchmarks' | 'languages' | 'ablations' | 'architecture'>('benchmarks');

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-6 sm:p-12 z-20 overflow-hidden">
      {/* Editorial Research Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-forensic-beam" />
            <span className="text-[10px] font-mono tracking-widest uppercase text-forensic-beam">
              MONOGRAPH // EMPIRICAL RESEARCH FINDINGS & BENCHMARKS
            </span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-display uppercase tracking-tight text-white">
            Multimodal Toxicity Monograph
          </h1>
          <p className="text-xs sm:text-sm font-sans text-white/50 max-w-2xl mt-1">
            Auditing Multi3Hate Zero-Leakage Evaluation, Language Slices, and Image-Text Pairing Sensitivity.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="pointer-events-auto flex items-center bg-obsidian/90 border border-white/10 p-1 backdrop-blur-md">
          {[
            { id: 'benchmarks', label: '01. Multi3Hate Benchmark' },
            { id: 'languages', label: '02. Language Slices' },
            { id: 'ablations', label: '03. Shuffled Image & Ablations' },
            { id: 'architecture', label: '04. 4-Way Fusion Math' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-3 py-1.5 text-xs font-mono uppercase tracking-wider transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-void font-bold shadow-[0_0_15px_rgba(255,255,255,0.2)]'
                  : 'text-white/40 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Tab Content Area */}
      <div className="flex-1 overflow-y-auto my-4 pr-2 pointer-events-auto">
        <AnimatePresence mode="wait">
          {/* ========================================================
              CHAPTER 01: MULTI3HATE BENCHMARK (OFFICIAL VS CANDIDATE)
             ======================================================== */}
          {activeTab === 'benchmarks' && (
            <motion.div
              key="benchmarks"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6 max-w-6xl"
            >
              {/* Executive Abstract & Official Decision Banner */}
              <div className="bg-obsidian/90 border border-white/10 p-6 backdrop-blur-md">
                <div className="flex items-center justify-between gap-4 mb-3 flex-wrap">
                  <div className="text-[10px] font-mono uppercase text-forensic-beam tracking-widest">
                    HELD-OUT BENCHMARK AUDIT // MULTI3HATE (N=600)
                  </div>
                  <div className="px-3 py-1 bg-white/10 border border-white/20 text-white font-mono text-xs uppercase tracking-wider">
                    STATUS: {EXPERIMENT_STATUS.officialWinner === 'CURRENT_OFFICIAL_MODEL' ? 'OFFICIAL MODEL RETAINED' : 'CANDIDATE ADOPTED'}
                  </div>
                </div>
                <p className="font-sans text-sm text-white/80 leading-relaxed">
                  {EXPERIMENT_STATUS.statusExplanation}
                </p>
                <div className="mt-3 text-xs font-mono text-white/50 border-t border-white/5 pt-2">
                  Zero-Leakage Guarantee: The Multi3Hate test set remained completely isolated from training, hyperparameter search, and threshold selection.
                </div>
              </div>

              {/* Comparative Results Table */}
              <div className="bg-obsidian/90 border border-white/10 p-6 backdrop-blur-md">
                <div className="text-xs font-mono uppercase text-white/50 tracking-wider mb-4 flex justify-between items-center">
                  <span>Multi3Hate Evaluation (Official Model vs Final Candidate)</span>
                  <span className="text-[10px] text-white/40">N=600 (300 English, 300 Hindi)</span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs">
                    <thead>
                      <tr className="border-b border-white/10 text-white/40 uppercase tracking-wider">
                        <th className="pb-3 font-normal">Model</th>
                        <th className="pb-3 font-normal">F1 Score</th>
                        <th className="pb-3 font-normal">AUROC</th>
                        <th className="pb-3 font-normal">FPR</th>
                        <th className="pb-3 font-normal">Precision</th>
                        <th className="pb-3 font-normal">Recall</th>
                        <th className="pb-3 font-normal">Hindi F1</th>
                        <th className="pb-3 font-normal">Hindi Recall</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {/* Official Baseline */}
                      <tr className="text-white bg-white/[0.04]">
                        <td className="py-3 font-bold flex items-center gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-forensic-safe" />
                          Official Production Baseline (Retained)
                        </td>
                        <td className="py-3 font-bold text-forensic-safe">{OFFICIAL_PRODUCTION_BASELINE.f1.toFixed(4)}</td>
                        <td className="py-3 font-bold text-forensic-safe">{OFFICIAL_PRODUCTION_BASELINE.auroc.toFixed(4)}</td>
                        <td className="py-3 font-bold text-forensic-safe">{OFFICIAL_PRODUCTION_BASELINE.fpr.toFixed(4)}</td>
                        <td className="py-3 text-white/60">—</td>
                        <td className="py-3 text-white/60">—</td>
                        <td className="py-3">{OFFICIAL_PRODUCTION_BASELINE.hindi_f1?.toFixed(4)}</td>
                        <td className="py-3">{OFFICIAL_PRODUCTION_BASELINE.hindi_recall?.toFixed(4)}</td>
                      </tr>

                      {/* Final Candidate */}
                      <tr className="text-white/80">
                        <td className="py-3 font-medium text-white/90">Final Candidate (Hard Negatives + Top-4 FT)</td>
                        <td className="py-3 font-medium">{FINAL_CANDIDATE_RESULTS.f1.toFixed(4)}</td>
                        <td className="py-3 font-medium">{FINAL_CANDIDATE_RESULTS.auroc.toFixed(4)}</td>
                        <td className="py-3 font-medium">{FINAL_CANDIDATE_RESULTS.fpr.toFixed(4)}</td>
                        <td className="py-3">{FINAL_CANDIDATE_RESULTS.precision?.toFixed(4)}</td>
                        <td className="py-3">{FINAL_CANDIDATE_RESULTS.recall?.toFixed(4)}</td>
                        <td className="py-3 text-forensic-beam font-medium">{FINAL_CANDIDATE_RESULTS.hindi_f1?.toFixed(4)}</td>
                        <td className="py-3 text-forensic-beam font-medium">{FINAL_CANDIDATE_RESULTS.hindi_recall?.toFixed(4)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Predefined Decision Criteria Checklist */}
                <div className="mt-6 pt-4 border-t border-white/10">
                  <div className="text-[11px] font-mono uppercase text-white/60 tracking-wider mb-3">
                    Predefined Adoption Criteria Assessment:
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs font-mono">
                    <div className="p-3 bg-charcoal/40 border border-white/5">
                      <div className="flex justify-between text-white/60 mb-1">
                        <span>F1 ≥ 0.60:</span>
                        <span className="text-forensic-risk font-bold">DID NOT MEET</span>
                      </div>
                      <div className="text-white/40 text-[11px]">Achieved: {FINAL_CANDIDATE_RESULTS.f1.toFixed(4)}</div>
                    </div>

                    <div className="p-3 bg-charcoal/40 border border-white/5">
                      <div className="flex justify-between text-white/60 mb-1">
                        <span>FPR ≤ 0.35:</span>
                        <span className="text-forensic-risk font-bold">DID NOT MEET</span>
                      </div>
                      <div className="text-white/40 text-[11px]">Achieved: {FINAL_CANDIDATE_RESULTS.fpr.toFixed(4)}</div>
                    </div>

                    <div className="p-3 bg-charcoal/40 border border-white/5">
                      <div className="flex justify-between text-white/60 mb-1">
                        <span>AUROC ≥ 0.584:</span>
                        <span className="text-forensic-risk font-bold">DID NOT MEET</span>
                      </div>
                      <div className="text-white/40 text-[11px]">Achieved: {FINAL_CANDIDATE_RESULTS.auroc.toFixed(4)}</div>
                    </div>

                    <div className="p-3 bg-charcoal/40 border border-white/5">
                      <div className="flex justify-between text-white/60 mb-1">
                        <span>Hindi F1 ≥ 0.55:</span>
                        <span className="text-forensic-risk font-bold">DID NOT MEET</span>
                      </div>
                      <div className="text-white/40 text-[11px]">Achieved: {FINAL_CANDIDATE_RESULTS.hindi_f1?.toFixed(4)}</div>
                    </div>

                    <div className="p-3 bg-charcoal/40 border border-white/5">
                      <div className="flex justify-between text-white/60 mb-1">
                        <span>Hindi Recall ≥ 0.50:</span>
                        <span className="text-forensic-risk font-bold">DID NOT MEET</span>
                      </div>
                      <div className="text-white/40 text-[11px]">Achieved: {FINAL_CANDIDATE_RESULTS.hindi_recall?.toFixed(4)}</div>
                    </div>

                    <div className="p-3 bg-charcoal/40 border border-white/5">
                      <div className="flex justify-between text-white/60 mb-1">
                        <span>Coupling ΔAUROC &gt; 0.02:</span>
                        <span className="text-forensic-safe font-bold">MET (+0.0391)</span>
                      </div>
                      <div className="text-white/40 text-[11px]">Achieved: +{SHUFFLED_IMAGE_RESULTS.delta_auroc.toFixed(4)}</div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-white/5 text-[10px] font-mono text-white/40 flex justify-between">
                  <span>Source artifact: final_hard_negative_pass/comparison_against_official.json</span>
                  <span>Recommendation: KEEP CURRENT</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* ========================================================
              CHAPTER 02: LANGUAGE SLICES (ENGLISH & HINDI ONLY)
             ======================================================== */}
          {activeTab === 'languages' && (
            <motion.div
              key="languages"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6 max-w-6xl"
            >
              <div className="bg-obsidian/90 border border-white/10 p-6 backdrop-blur-md">
                <div className="text-xs font-mono uppercase text-forensic-beam tracking-wider mb-2">
                  EVALUATED MULTI3HATE LANGUAGE SLICES (ENGLISH & HINDI, N=600)
                </div>
                <p className="font-sans text-xs text-white/70 leading-relaxed mb-4">
                  The Multi3Hate held-out evaluation consists of two balanced language partitions: English (N=300) and Hindi (N=300). Metrics below represent actual empirical results from <span className="font-mono text-white">per_language_metrics.json</span>.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                  {EVALUATED_LANGUAGE_SLICES.map((slice) => (
                    <div key={slice.language} className="p-4 bg-charcoal/50 border border-white/10 space-y-3">
                      <div className="flex justify-between items-center border-b border-white/10 pb-2">
                        <span className="font-display text-lg text-white font-bold">{slice.language} Partition</span>
                        <span className="font-mono text-xs text-white/40">N=300 (150 Toxic / 150 Safe)</span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                        <div>
                          <span className="text-white/40 block text-[10px]">F1 SCORE</span>
                          <span className="text-white font-bold text-base">{slice.f1.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-white/40 block text-[10px]">AUROC</span>
                          <span className="text-white font-bold text-base">{slice.auroc.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-white/40 block text-[10px]">RECALL</span>
                          <span className="text-white font-medium">{slice.recall.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-white/40 block text-[10px]">PRECISION</span>
                          <span className="text-white font-medium">{slice.precision.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-white/40 block text-[10px]">FALSE POSITIVE RATE</span>
                          <span className="text-white font-medium">{slice.fpr.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-white/40 block text-[10px]">ACCURACY</span>
                          <span className="text-white font-medium">{slice.accuracy.toFixed(4)}</span>
                        </div>
                      </div>

                      <div className="border-t border-white/5 pt-2 text-[11px] font-mono text-white/50 flex justify-between">
                        <span>TP: {slice.tp} | FP: {slice.fp}</span>
                        <span>TN: {slice.tn} | FN: {slice.fn}</span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-6 p-4 bg-charcoal/30 border border-white/5 text-xs font-mono space-y-2">
                  <div className="text-white font-bold">HINDI SHIFT AUDIT:</div>
                  <div className="text-white/70">
                    • Official Baseline Hindi F1: <span className="text-white font-bold">0.2906</span> (Recall: 0.1889, FPR: 0.2250)
                  </div>
                  <div className="text-white/70">
                    • Final Candidate Hindi F1: <span className="text-white font-bold">0.4245</span> (Recall: 0.3278, FPR: 0.3250)
                  </div>
                  <div className="text-white/50 text-[11px] pt-1">
                    The final candidate improved Hindi recall (+73.5% relative gain over baseline) through hard-negative tuning, but incurred increased false positives that precluded adoption under the strict FPR ≤ 0.35 requirement.
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-white/5 text-[10px] font-mono text-white/40">
                  Note: Multi3Hate benchmarks evaluate English and Hindi partitions only. Other Indic language observations are qualitative exploratory research.
                </div>
              </div>
            </motion.div>
          )}

          {/* ========================================================
              CHAPTER 03: SHUFFLED IMAGE CONTROL & ABLATIONS
             ======================================================== */}
          {activeTab === 'ablations' && (
            <motion.div
              key="ablations"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6 max-w-6xl"
            >
              <div className="bg-obsidian/90 border border-white/10 p-6 backdrop-blur-md">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2 h-2 rounded-full bg-forensic-beam" />
                  <span className="text-xs font-mono uppercase text-forensic-beam tracking-wider font-bold">
                    SHUFFLED IMAGE CONTROL EXPERIMENT
                  </span>
                </div>
                <p className="font-sans text-sm text-white/80 leading-relaxed mb-4">
                  {SHUFFLED_IMAGE_RESULTS.description}
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4">
                  <div className="p-4 bg-charcoal/40 border border-white/10">
                    <div className="text-[10px] font-mono text-white/40 uppercase mb-1">Paired (True Image-Text Pairing)</div>
                    <div className="text-2xl font-mono font-bold text-white">{SHUFFLED_IMAGE_RESULTS.paired_auroc.toFixed(4)} AUROC</div>
                    <p className="text-xs text-white/60 mt-1 font-mono">True cross-modal correspondence intact</p>
                  </div>
                  <div className="p-4 bg-charcoal/40 border border-white/10">
                    <div className="text-[10px] font-mono text-white/40 uppercase mb-1">Shuffled Images (Decorrelated Pairing)</div>
                    <div className="text-2xl font-mono font-bold text-white/50">{SHUFFLED_IMAGE_RESULTS.shuffled_auroc.toFixed(4)} AUROC</div>
                    <p className="text-xs text-white/60 mt-1 font-mono">Random image permuted across batch</p>
                  </div>
                </div>

                <div className="p-4 bg-charcoal/60 border border-white/20 font-mono text-xs text-white">
                  <span className="text-forensic-beam font-bold block mb-1">
                    Image-text pairing sensitivity: ΔAUROC = +{SHUFFLED_IMAGE_RESULTS.delta_auroc.toFixed(4)}
                  </span>
                  Performance drops by 0.0391 when image pairings are decorrelated, confirming that the model utilizes cross-modal correspondence rather than operating as an isolated text classifier.
                </div>
              </div>

              {/* Full Ablation Table */}
              <div className="bg-obsidian/90 border border-white/10 p-6 backdrop-blur-md">
                <div className="text-xs font-mono uppercase text-white/50 tracking-wider mb-4">
                  Full Modality Ablation Matrix (Multi3Hate, N=600)
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs">
                    <thead>
                      <tr className="border-b border-white/10 text-white/40 uppercase tracking-wider">
                        <th className="pb-3 font-normal">Condition</th>
                        <th className="pb-3 font-normal">Description</th>
                        <th className="pb-3 font-normal">F1</th>
                        <th className="pb-3 font-normal">AUROC</th>
                        <th className="pb-3 font-normal">Accuracy</th>
                        <th className="pb-3 font-normal">FPR</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {ABLATION_STUDY_METRICS.map((item) => (
                        <tr key={item.condition} className="text-white/80">
                          <td className="py-3 font-medium text-white">{item.condition}</td>
                          <td className="py-3 text-white/50 text-[11px]">{item.description}</td>
                          <td className="py-3 font-bold text-white">{item.f1.toFixed(4)}</td>
                          <td className="py-3">{item.auroc.toFixed(4)}</td>
                          <td className="py-3">{item.accuracy.toFixed(4)}</td>
                          <td className="py-3">{item.fpr.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="mt-4 pt-3 border-t border-white/5 text-[10px] font-mono text-white/40 flex justify-between">
                  <span>Source artifact: final_hard_negative_pass/ablation_results.json</span>
                  <span>Evaluated under zero-leakage held-out quarantine</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* ========================================================
              CHAPTER 04: 4-WAY FUSION MATHEMATICAL ARCHITECTURE
             ======================================================== */}
          {activeTab === 'architecture' && (
            <motion.div
              key="architecture"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6 max-w-6xl"
            >
              <div className="bg-obsidian/90 border border-white/10 p-6 backdrop-blur-md">
                <div className="text-xs font-mono uppercase text-forensic-beam tracking-wider mb-2">
                  4-WAY CROSS-MODAL INTERACTION FORMULATION
                </div>
                <p className="font-sans text-sm text-white/80 leading-relaxed mb-4">
                  The model projects text features (MuRIL) and visual features (CLIP ViT-B/32) into a shared 256-dimensional space, then constructs a 1024-dimensional joint representation vector:
                </p>

                <div className="p-4 bg-void border border-white/10 font-mono text-sm text-forensic-beam overflow-x-auto my-4 text-center">
                  h_fusion = [ t_proj ∥ v_proj ∥ (t_proj ⊙ v_proj) ∥ |t_proj - v_proj| ] ∈ ℝ^1024
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mt-6 text-xs font-mono">
                  <div className="p-3 bg-charcoal/60 border border-white/5">
                    <span className="text-white font-bold block mb-1">t_proj (256d)</span>
                    <span className="text-white/50">Linear projection of MuRIL semantic embedding of OCR tokens.</span>
                  </div>
                  <div className="p-3 bg-charcoal/60 border border-white/5">
                    <span className="text-white font-bold block mb-1">v_proj (256d)</span>
                    <span className="text-white/50">Linear projection of CLIP visual feature vector from image plane.</span>
                  </div>
                  <div className="p-3 bg-charcoal/60 border border-white/5">
                    <span className="text-white font-bold block mb-1">t_proj ⊙ v_proj (256d)</span>
                    <span className="text-white/50">Hadamard element-wise product capturing mutual cross-modal reinforcement.</span>
                  </div>
                  <div className="p-3 bg-charcoal/60 border border-white/5">
                    <span className="text-white font-bold block mb-1">|t_proj - v_proj| (256d)</span>
                    <span className="text-white/50">L1 difference vector capturing cross-modal contradiction & dissonance.</span>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-white/5 text-[11px] font-mono text-white/50">
                  Total representation tensor dimensionality: 256 × 4 = 1024 dimensions, feeding into classification MLP with temperature scaling.
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Citation & Metadata */}
      <div className="flex items-center justify-between text-xs font-mono text-white/40 pt-2 border-t border-white/10">
        <span>PROJECT: MULTILINGUAL SOCIAL MEDIA TOXICITY DETECTION // RESEARCH COMPENDIUM</span>
        <span>SOURCE: final_hard_negative_pass ARTIFACTS</span>
      </div>
    </div>
  );
};
