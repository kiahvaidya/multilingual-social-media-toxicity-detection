import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { ToxicityScores } from '../../types';

interface TypographicTextChamberProps {
  text?: string;
  scores?: ToxicityScores | null;
  stage?: string;
  isAnalyzing?: boolean;
}

const DIMENSIONS: (keyof ToxicityScores)[] = [
  'toxic',
  'severe_toxic',
  'obscene',
  'threat',
  'insult',
  'identity_hate',
];

export const TypographicTextChamber: React.FC<TypographicTextChamberProps> = ({
  text,
  scores,
  stage = 'IDLE',
  isAnalyzing = false,
}) => {
  const groupRef = useRef<THREE.Group>(null);

  // Tokens breakdown
  const tokens = useMemo(() => {
    if (!text) return ['RAW', 'TEXT', 'PIPELINE', 'INITIALIZING'];
    return text.split(/\s+/).slice(0, 16);
  }, [text]);

  // Compute 3D Radar Geometry from scores
  const radarPoints = useMemo(() => {
    const numAxes = DIMENSIONS.length;
    const pts: THREE.Vector3[] = [];
    const radiusBase = 1.8;

    for (let i = 0; i < numAxes; i++) {
      const dim = DIMENSIONS[i];
      const val = scores ? scores[dim] : 0.15;
      const angle = (i / numAxes) * Math.PI * 2;
      const r = radiusBase * (0.2 + val * 0.9);
      pts.push(new THREE.Vector3(Math.cos(angle) * r, Math.sin(angle) * r, 0));
    }
    return pts;
  }, [scores]);

  const radarGeometry = useMemo(() => {
    const geom = new THREE.BufferGeometry().setFromPoints([...radarPoints, radarPoints[0]]);
    return geom;
  }, [radarPoints]);

  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    const t = clock.getElapsedTime();

    if (isAnalyzing) {
      groupRef.current.rotation.y = Math.sin(t * 1.5) * 0.12;
      groupRef.current.position.y = 0.2 + Math.cos(t * 2.0) * 0.05;
    } else {
      groupRef.current.rotation.y = Math.sin(t * 0.5) * 0.04;
      groupRef.current.position.y = 0.1 + Math.sin(t * 0.8) * 0.03;
    }
  });

  return (
    <group ref={groupRef} position={[1.5, 0, 0]}>
      {/* 3D Wireframe Dimension Ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.75, 1.78, 32]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>

      {/* Axis Strands */}
      {DIMENSIONS.map((dim, idx) => {
        const angle = (idx / DIMENSIONS.length) * Math.PI * 2;
        const x = Math.cos(angle) * 2.0;
        const y = Math.sin(angle) * 2.0;
        const val = scores ? (scores[dim] * 100).toFixed(0) : '0';
        const isTriggered = scores ? scores[dim] >= 0.5 : false;

        return (
          <group key={dim} position={[0, 0, 0]}>
            {/* Axis Ray */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array([0, 0, 0, x, y, 0]), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#64748b" transparent opacity={0.3} />
            </line>

            {/* Dimension Node Indicator */}
            <mesh position={[x * 1.08, y * 1.08, 0]}>
              <sphereGeometry args={[0.06, 12, 12]} />
              <meshBasicMaterial color={isTriggered ? '#f43f5e' : '#38bdf8'} />
            </mesh>

            {/* Label */}
            <Text
              position={[x * 1.25, y * 1.25, 0]}
              fontSize={0.11}
              color={isTriggered ? '#f43f5e' : '#94a3b8'}
              font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
              anchorX="center"
              anchorY="middle"
            >
              {`${dim.toUpperCase()} ${scores ? val + '%' : ''}`}
            </Text>
          </group>
        );
      })}

      {/* Live Polygon connecting toxicity scores */}
      <primitive object={new THREE.LineLoop(radarGeometry, new THREE.LineBasicMaterial({ color: '#38bdf8', linewidth: 2, transparent: true, opacity: 0.85 }))} />


      {/* Center Core: MuRIL Encoder Representation Node */}
      <mesh position={[0, 0, 0]}>
        <octahedronGeometry args={[0.35, 1]} />
        <meshStandardMaterial
          color={isAnalyzing ? '#38bdf8' : '#1e293b'}
          roughness={0.2}
          metalness={0.8}
          wireframe
        />
      </mesh>

      {/* Floating Spatial Token Cloud */}
      {tokens.map((tok, i) => {
        const phi = (i / tokens.length) * Math.PI * 2;
        const radius = 2.8 + (i % 3) * 0.3;
        const x = Math.cos(phi) * radius;
        const y = Math.sin(phi) * radius * 0.7;
        const z = ((i % 5) - 2) * 0.4;

        return (
          <Text
            key={`${tok}-${i}`}
            position={[x, y, z]}
            fontSize={0.12}
            color="#cbd5e1"
            font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
            anchorX="center"
            anchorY="middle"
            fillOpacity={0.65}
          >
            {tok}
          </Text>
        );
      })}
    </group>
  );
};
