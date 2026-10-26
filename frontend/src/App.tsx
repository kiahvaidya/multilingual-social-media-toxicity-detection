import React, { useState, useEffect } from 'react';
import { ForensicCanvas } from './components/3d/ForensicCanvas';
import { ForensicNavigation } from './components/hud/ForensicNavigation';
import { SceneMode } from './components/3d/CameraController';

// Cinematic Overlays
import { LandingOverlay } from './components/overlays/LandingOverlay';
import { OverviewOverlay } from './components/overlays/OverviewOverlay';
import { TextAnalysisOverlay } from './components/overlays/TextAnalysisOverlay';
import { MemeAnalysisOverlay } from './components/overlays/MemeAnalysisOverlay';
import { InversionExhibitOverlay } from './components/overlays/InversionExhibitOverlay';
import { QueueOverlay, QueueItem } from './components/overlays/QueueOverlay';
import { ResearchOverlay } from './components/overlays/ResearchOverlay';
import { SocialAnalysisOverlay } from './components/overlays/SocialAnalysisOverlay';
import { ProfileWatchOverlay } from './components/overlays/ProfileWatchOverlay';

import {
  TextAnalysisResult,
  MemeAnalysisResult,
  SystemHealth,
  ProfileRecord,
  AggregateMetrics,
  ObservationRecord,
  PatternEventRecord,
} from './types';
import { checkHealth } from './api/client';

export function App() {
  const [sceneMode, setSceneMode] = useState<SceneMode>('LANDING');
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);

  // Analysis state
  const [textResult, setTextResult] = useState<TextAnalysisResult | null>(null);
  const [isAnalyzingText, setIsAnalyzingText] = useState(false);

  const [memeResult, setMemeResult] = useState<MemeAnalysisResult | null>(null);
  const [memeImageUrl, setMemeImageUrl] = useState<string | null>(null);
  const [isAnalyzingMeme, setIsAnalyzingMeme] = useState(false);

  // Profile Watch 3D synchronization state
  const [pwProfile, setPwProfile] = useState<ProfileRecord | null>(null);
  const [pwAggregates, setPwAggregates] = useState<AggregateMetrics | null>(null);
  const [pwObservations, setPwObservations] = useState<ObservationRecord[]>([]);
  const [pwEvents, setPwEvents] = useState<PatternEventRecord[]>([]);
  const [isScanningPw, setIsScanningPw] = useState(false);

  // Inversion experiment mode
  const [inversionMode, setInversionMode] = useState<'INVERSION' | 'EMERGENT'>('INVERSION');

  // Backend Health Ping
  useEffect(() => {
    checkHealth()
      .then((health) => setSystemHealth(health))
      .catch((err) => {
        console.warn('Backend health check notification:', err);
      });
  }, []);

  const handleQueueSelect = (item: QueueItem) => {
    if (item.type === 'meme') {
      setSceneMode('MEME');
    } else {
      setSceneMode('TEXT');
    }
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-void select-none font-sans text-white">
      {/* 1. Underlying WebGL Spatial Universe (THE INTERFACE) */}
      <ForensicCanvas
        mode={sceneMode}
        textResult={textResult}
        memeResult={memeResult}
        memeImageUrl={memeImageUrl}
        inversionMode={inversionMode}
        isAnalyzing={isAnalyzingText || isAnalyzingMeme}
        profileWatchProfile={pwProfile}
        profileWatchAggregates={pwAggregates}
        profileWatchObservations={pwObservations}
        profileWatchEvents={pwEvents}
        isScanningProfile={isScanningPw}
      />

      {/* 2. Film Grain Noise Filter & Vignette Overlays for Depth */}
      <div className="pointer-events-none fixed inset-0 z-10 bg-noise opacity-[0.035] mix-blend-overlay" />
      <div className="pointer-events-none fixed inset-0 z-10 bg-[radial-gradient(circle_at_center,transparent_30%,rgba(5,6,8,0.7)_100%)]" />

      {/* 3. Global Forensic HUD Navigation */}
      <ForensicNavigation
        currentMode={sceneMode}
        onSelectMode={(mode) => setSceneMode(mode)}
        systemStatus={systemHealth ? 'ONLINE' : 'ACTIVE (LOCAL)'}
        deviceEngine={systemHealth?.device ? `MuRIL + CLIP [${systemHealth.device.toUpperCase()}]` : 'MuRIL + CLIP + JEV'}
      />

      {/* 4. Active Chamber Overlays (Rendered as physical HUDs over 3D coordinates) */}
      <main className="relative z-20 w-full h-full">
        {sceneMode === 'LANDING' && (
          <LandingOverlay
            onEnter={() => setSceneMode('OVERVIEW')}
            onExploreMeme={() => setSceneMode('MEME')}
            onExploreSocial={() => setSceneMode('SOCIAL')}
            onExploreProfileWatch={() => setSceneMode('PROFILE_WATCH')}
          />
        )}

        {sceneMode === 'OVERVIEW' && (
          <OverviewOverlay
            onGoToText={() => setSceneMode('TEXT')}
            onGoToMeme={() => setSceneMode('MEME')}
            onGoToInversion={() => setSceneMode('INVERSION')}
            onGoToResearch={() => setSceneMode('RESEARCH')}
            onGoToSocial={() => setSceneMode('SOCIAL')}
            onGoToProfileWatch={() => setSceneMode('PROFILE_WATCH')}
          />
        )}

        {sceneMode === 'TEXT' && (
          <TextAnalysisOverlay
            onResult={(res) => setTextResult(res)}
            result={textResult}
            isAnalyzing={isAnalyzingText}
            setIsAnalyzing={setIsAnalyzingText}
            onGoToSocial={() => setSceneMode('SOCIAL')}
          />
        )}

        {sceneMode === 'MEME' && (
          <MemeAnalysisOverlay
            onResult={(res, imgUrl) => {
              setMemeResult(res);
              setMemeImageUrl(imgUrl);
            }}
            result={memeResult}
            isAnalyzing={isAnalyzingMeme}
            setIsAnalyzing={setIsAnalyzingMeme}
          />
        )}

        {sceneMode === 'INVERSION' && (
          <InversionExhibitOverlay
            mode={inversionMode.toLowerCase() as 'inversion' | 'emergent'}
            onToggleMode={(m) => setInversionMode(m.toUpperCase() as 'INVERSION' | 'EMERGENT')}
          />
        )}

        {sceneMode === 'QUEUE' && (
          <QueueOverlay onSelectCase={handleQueueSelect} />
        )}

        {sceneMode === 'RESEARCH' && (
          <ResearchOverlay />
        )}

        {sceneMode === 'SOCIAL' && (
          <SocialAnalysisOverlay onGoToText={() => setSceneMode('TEXT')} />
        )}

        {sceneMode === 'PROFILE_WATCH' && (
          <ProfileWatchOverlay
            onGoToSocial={() => setSceneMode('SOCIAL')}
            onGoToText={() => setSceneMode('TEXT')}
            onProfileStateChange={(p, agg, obs, evs, scanning) => {
              setPwProfile(p);
              setPwAggregates(agg);
              setPwObservations(obs);
              setPwEvents(evs);
              setIsScanningPw(scanning);
            }}
          />
        )}
      </main>
    </div>
  );
}

export default App;
