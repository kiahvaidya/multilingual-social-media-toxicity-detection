import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import {
  AggregateMetrics,
  ObservationRecord,
  PatternEventRecord,
  ProfileRecord,
} from '../../types';

interface ProfileWatchChamberProps {
  activeProfile?: ProfileRecord | null;
  aggregates?: AggregateMetrics | null;
  observations?: ObservationRecord[];
  events?: PatternEventRecord[];
  isScanning?: boolean;
}

const CATEGORY_NODES = [
  { key: 'toxic', label: 'TOXIC', angle: 0 },
  { key: 'severe_toxic', label: 'SEV_TOXIC', angle: Math.PI / 3 },
  { key: 'obscene', label: 'OBSCENE', angle: (2 * Math.PI) / 3 },
  { key: 'threat', label: 'THREAT', angle: Math.PI },
  { key: 'insult', label: 'INSULT', angle: (4 * Math.PI) / 3 },
  { key: 'identity_hate', label: 'IDENTITY', angle: (5 * Math.PI) / 3 },
];

export const ProfileWatchChamber: React.FC<ProfileWatchChamberProps> = ({
  activeProfile,
  aggregates,
  observations = [],
  events = [],
  isScanning = false,
}) => {
  const centralMeshRef = useRef<THREE.Mesh>(null);
  const outerRingRef = useRef<THREE.Group>(null);
  const temporalRingRef = useRef<THREE.Group>(null);

  // Overall profile activity risk tone
  const profileRisk = aggregates?.profile_activity_risk || 'LOW';
  const primaryColor = useMemo(() => {
    switch (profileRisk) {
      case 'HIGH':
        return '#f43f5e'; // Crimson
      case 'ELEVATED':
        return '#f59e0b'; // Amber
      case 'MODERATE':
        return '#eab308'; // Gold
      case 'LOW':
      default:
        return '#06b6d4'; // Cyan
    }
  }, [profileRisk]);

  // Map real historical observations into 3D temporal field
  const particleNodes = useMemo(() => {
    if (!observations || observations.length === 0) return [];

    // Chronological order
    const sorted = [...observations].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    const total = sorted.length;

    return sorted.map((obs, idx) => {
      // Angle mapped to position in chronological timeline
      const progress = total > 1 ? idx / (total - 1) : 0.5;
      const angle = progress * Math.PI * 1.8 - Math.PI * 0.9;
      const baseRadius = 2.2 + (idx % 3) * 0.35;

      // Height Y mapped to toxicity intensity
      const maxScore = Math.max(...Object.values(obs.scores || { toxic: 0 }));
      const y = (maxScore - 0.5) * 1.4;

      const x = Math.cos(angle) * baseRadius;
      const z = Math.sin(angle) * baseRadius;

      // Color based strictly on verified verdict
      let color = '#06b6d4'; // SAFE = cyan
      let size = 0.065;
      if (obs.final_risk_level === 'High Risk') {
        color = '#f43f5e'; // HIGH RISK = crimson
        size = 0.12;
      } else if (obs.final_risk_level === 'Review') {
        color = '#f59e0b'; // REVIEW = amber
        size = 0.09;
      }

      return {
        id: obs.id,
        x,
        y,
        z,
        color,
        size,
        risk: obs.final_risk_level,
        type: obs.activity_type || 'comment',
      };
    });
  }, [observations]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const speed = isScanning ? 1.8 : 0.3;

    if (centralMeshRef.current) {
      centralMeshRef.current.rotation.y = t * speed;
      centralMeshRef.current.rotation.x = Math.sin(t * 0.5) * 0.12;
    }
    if (outerRingRef.current) {
      outerRingRef.current.rotation.y = -t * (speed * 0.4);
    }
    if (temporalRingRef.current) {
      temporalRingRef.current.rotation.z = t * 0.08;
    }
  });

  return (
    <group position={[0, 0.4, 0]}>
      {/* ── 1. CENTRAL PROFILE DOSSIER CORE ── */}
      <group position={[0, 0, 0]}>
        {/* Core Wireframe Octahedron */}
        <mesh ref={centralMeshRef}>
          <octahedronGeometry args={[0.75, 1]} />
          <meshStandardMaterial
            color={primaryColor}
            emissive={primaryColor}
            emissiveIntensity={isScanning ? 0.95 : 0.45}
            wireframe
            roughness={0.2}
          />
        </mesh>

        {/* Inner Luminous Sphere */}
        <mesh>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial color={primaryColor} transparent opacity={0.35} />
        </mesh>

        {/* Outer Orbiting Scanning Rings */}
        <group ref={outerRingRef}>
          <mesh rotation={[Math.PI / 3, 0, 0]}>
            <ringGeometry args={[1.1, 1.14, 48]} />
            <meshBasicMaterial
              color={primaryColor}
              transparent
              opacity={0.4}
              side={THREE.DoubleSide}
            />
          </mesh>
          <mesh rotation={[-Math.PI / 4, 0, 0]}>
            <ringGeometry args={[1.35, 1.38, 48]} />
            <meshBasicMaterial
              color="#38bdf8"
              transparent
              opacity={0.22}
              side={THREE.DoubleSide}
            />
          </mesh>
        </group>

        {/* Profile Identifier Header */}
        <Text
          position={[0, 1.25, 0]}
          fontSize={0.19}
          color="#f8fafc"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {activeProfile?.handle || 'NO PROFILE LOADED'}
        </Text>

        <Text
          position={[0, 1.0, 0]}
          fontSize={0.11}
          color={primaryColor}
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {activeProfile
            ? `[ ${activeProfile.platform.toUpperCase()} // DOSSIER RISK: ${profileRisk} ]`
            : '[ ENTER PUBLIC URL TO INGEST ACTIVITY ]'}
        </Text>
      </group>

      {/* ── 2. CATEGORY SATELLITE NODES ── */}
      <group>
        {CATEGORY_NODES.map((cat) => {
          const radius = 3.4;
          const x = Math.cos(cat.angle) * radius;
          const z = Math.sin(cat.angle) * radius;
          const catAvg = aggregates?.average_scores?.[cat.key] ?? 0;
          const catCount = aggregates?.category_flag_counts?.[cat.key] ?? 0;
          const isElevated = catAvg >= 0.45 || catCount > 0;
          const nodeColor = isElevated ? '#f43f5e' : '#64748b';

          return (
            <group key={cat.key} position={[x, 0, z]}>
              <mesh>
                <sphereGeometry args={[0.13, 12, 12]} />
                <meshStandardMaterial
                  color={nodeColor}
                  emissive={nodeColor}
                  emissiveIntensity={isElevated ? 0.75 : 0.2}
                  roughness={0.3}
                />
              </mesh>
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <ringGeometry args={[0.18, 0.2, 24]} />
                <meshBasicMaterial color={nodeColor} transparent opacity={0.3} side={THREE.DoubleSide} />
              </mesh>
              <Text
                position={[0, 0.32, 0]}
                fontSize={0.1}
                color="#cbd5e1"
                font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
                anchorX="center"
              >
                {cat.label}
              </Text>
              <Text
                position={[0, -0.3, 0]}
                fontSize={0.09}
                color={nodeColor}
                font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
                anchorX="center"
              >
                {`${(catAvg * 100).toFixed(0)}% (${catCount})`}
              </Text>
            </group>
          );
        })}
      </group>

      {/* ── 3. ACTUAL HISTORICAL OBSERVATION NODES ── */}
      <group>
        {particleNodes.map((p) => (
          <mesh key={p.id} position={[p.x, p.y, p.z]}>
            <sphereGeometry args={[p.size, 10, 10]} />
            <meshStandardMaterial
              color={p.color}
              emissive={p.color}
              emissiveIntensity={p.risk === 'High Risk' ? 0.8 : 0.55}
              roughness={0.2}
            />
          </mesh>
        ))}
      </group>

      {/* ── 4. CHRONOLOGICAL TEMPORAL FIELD ARCS ── */}
      <group ref={temporalRingRef} rotation={[Math.PI / 6, 0, 0]}>
        <mesh>
          <ringGeometry args={[2.1, 2.12, 64]} />
          <meshBasicMaterial color="#06b6d4" transparent opacity={0.2} side={THREE.DoubleSide} />
        </mesh>
        <mesh rotation={[0, Math.PI / 4, 0]}>
          <ringGeometry args={[3.8, 3.82, 64]} />
          <meshBasicMaterial color="#f43f5e" transparent opacity={0.15} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* ── 5. BEHAVIORAL PATTERN SPATIAL SIGNALS ── */}
      <group>
        {events.slice(0, 3).map((ev, i) => {
          const yOffset = -1.25 - i * 0.42;
          const isCritical = ev.severity === 'CRITICAL' || ev.severity === 'HIGH';
          const signalColor = isCritical ? '#f43f5e' : '#f59e0b';

          return (
            <group key={ev.id || i} position={[0, yOffset, 0]}>
              <mesh position={[-2.4, 0, 0]}>
                <boxGeometry args={[0.07, 0.07, 0.07]} />
                <meshStandardMaterial
                  color={signalColor}
                  emissive={signalColor}
                  emissiveIntensity={0.8}
                />
              </mesh>
              <Text
                position={[0, 0, 0]}
                fontSize={0.11}
                color={signalColor}
                font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
                anchorX="center"
              >
                {`[SIGNAL: ${ev.pattern_type}] ${ev.title.slice(0, 48)}`}
              </Text>
            </group>
          );
        })}
      </group>
    </group>
  );
};
