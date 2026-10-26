import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { CameraController, SceneMode } from './CameraController';
import { MemeticIntelligenceField } from './MemeticIntelligenceField';
import { TypographicTextChamber } from './TypographicTextChamber';
import { MemeInvestigationChamber } from './MemeInvestigationChamber';
import { ContextualInversionExhibit } from './ContextualInversionExhibit';
import { CommandCenterNodes } from './CommandCenterNodes';
import { ProfileWatchChamber } from './ProfileWatchChamber';
import {
  TextAnalysisResult,
  MemeAnalysisResult,
  ProfileRecord,
  AggregateMetrics,
  ObservationRecord,
  PatternEventRecord,
} from '../../types';

interface ForensicCanvasProps {
  mode: SceneMode;
  textResult?: TextAnalysisResult | null;
  memeResult?: MemeAnalysisResult | null;
  memeImageUrl?: string | null;
  inversionMode?: 'INVERSION' | 'EMERGENT';
  isAnalyzing?: boolean;
  profileWatchProfile?: ProfileRecord | null;
  profileWatchAggregates?: AggregateMetrics | null;
  profileWatchObservations?: ObservationRecord[];
  profileWatchEvents?: PatternEventRecord[];
  isScanningProfile?: boolean;
}

export const ForensicCanvas: React.FC<ForensicCanvasProps> = ({
  mode,
  textResult,
  memeResult,
  memeImageUrl,
  inversionMode = 'INVERSION',
  isAnalyzing = false,
  profileWatchProfile,
  profileWatchAggregates,
  profileWatchObservations = [],
  profileWatchEvents = [],
  isScanningProfile = false,
}) => {
  // Determine overall environment tension tone
  const activeRisk = mode === 'TEXT' 
    ? textResult?.risk_level 
    : mode === 'MEME' 
    ? memeResult?.risk_level 
    : mode === 'PROFILE_WATCH'
    ? (profileWatchAggregates && profileWatchAggregates.high_risk_rate > 20
        ? 'High Risk'
        : profileWatchAggregates && profileWatchAggregates.review_count > 0
        ? 'Review'
        : 'Safe')
    : null;

  const keyLightColor = activeRisk === 'High Risk' 
    ? '#f43f5e' 
    : activeRisk === 'Review' 
    ? '#f59e0b' 
    : '#38bdf8';

  return (
    <div className="fixed inset-0 w-full h-full pointer-events-auto z-0 bg-[#050608]">
      <Canvas
        camera={{ position: [0, 1.5, 14], fov: 48 }}
        dpr={[1, 1.75]} // Adaptive high-DPR capped for 60 FPS
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
        }}
      >
        {/* Volumetric Dark Atmosphere */}
        <color attach="background" args={['#050608']} />
        <fog attach="fog" args={['#050608', 7, 26]} />

        {/* Cinematic Lighting System */}
        <ambientLight intensity={0.45} />
        <directionalLight
          position={[6, 8, 8]}
          intensity={1.2}
          color={keyLightColor}
        />
        <pointLight position={[-8, -4, 4]} intensity={0.6} color="#64748b" />
        <pointLight position={[0, 5, -6]} intensity={0.8} color="#1e293b" />

        {/* Unified Camera Controller */}
        <CameraController mode={mode} />

        <Suspense fallback={null}>
          {/* 1. Underlying Spatial Memetic Intelligence Field (Always active) */}
          <MemeticIntelligenceField verdictRisk={activeRisk} />

          {/* 2. Scene-Specific 3D Architectural Modules */}
          {mode === 'OVERVIEW' && <CommandCenterNodes />}

          {mode === 'TEXT' && (
            <TypographicTextChamber
              text={textResult?.text}
              scores={textResult?.scores}
              isAnalyzing={isAnalyzing}
            />
          )}

          {mode === 'MEME' && (
            <MemeInvestigationChamber
              imageUrl={memeImageUrl}
              ocrText={memeResult?.text}
              result={memeResult}
              isAnalyzing={isAnalyzing}
            />
          )}

          {mode === 'INVERSION' && (
            <ContextualInversionExhibit modeType={inversionMode} />
          )}

          {mode === 'PROFILE_WATCH' && (
            <ProfileWatchChamber
              activeProfile={profileWatchProfile}
              aggregates={profileWatchAggregates}
              observations={profileWatchObservations}
              events={profileWatchEvents}
              isScanning={isScanningProfile}
            />
          )}
        </Suspense>
      </Canvas>
    </div>
  );
};
