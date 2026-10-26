import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface MetricNodeProps {
  position: [number, number, number];
  title: string;
  value: string;
  subtext: string;
  color: string;
}

const MetricNode: React.FC<MetricNodeProps> = ({ position, title, value, subtext, color }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = clock.getElapsedTime() * 0.3;
    }
  });

  return (
    <group position={position}>
      {/* Node Cage */}
      <mesh ref={meshRef}>
        <octahedronGeometry args={[0.32, 0]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.35}
          roughness={0.2}
          wireframe
        />
      </mesh>

      {/* Orbit Ring */}
      <mesh rotation={[Math.PI / 3, 0, 0]}>
        <ringGeometry args={[0.45, 0.47, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.25} side={THREE.DoubleSide} />
      </mesh>

      {/* Title */}
      <Text
        position={[0, 0.65, 0]}
        fontSize={0.11}
        color="#94a3b8"
        font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
        anchorX="center"
      >
        {title}
      </Text>

      {/* Value */}
      <Text
        position={[0, -0.65, 0]}
        fontSize={0.24}
        color="#f8fafc"
        font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
        anchorX="center"
      >
        {value}
      </Text>

      {/* Subtext */}
      <Text
        position={[0, -0.92, 0]}
        fontSize={0.09}
        color={color}
        font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
        anchorX="center"
      >
        {subtext}
      </Text>
    </group>
  );
};

export const CommandCenterNodes: React.FC = () => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.15) * 0.08;
    }
  });

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {/* Central Spatial Ground Reticle */}
      <mesh position={[0, -1.8, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[3.0, 3.03, 64]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, -1.8, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[5.0, 5.03, 64]} />
        <meshBasicMaterial color="#64748b" transparent opacity={0.1} side={THREE.DoubleSide} />
      </mesh>

      {/* Metric 1: Analyzed Messages */}
      <MetricNode
        position={[-3.6, 0.4, 0]}
        title="INGESTED CORPUS"
        value="12,481"
        subtext="LIVE TELEMETRY STREAM"
        color="#38bdf8"
      />

      {/* Metric 2: High Risk */}
      <MetricNode
        position={[-1.2, 1.2, -1.5]}
        title="CRITICAL HIGH RISK"
        value="1,832"
        subtext="AUTO-FLAGGED IDENTITY HATE"
        color="#f43f5e"
      />

      {/* Metric 3: Ambiguity / Review */}
      <MetricNode
        position={[1.2, 1.2, -1.5]}
        title="BORDERLINE AMBIGUOUS"
        value="2,194"
        subtext="ROUTED TO JEV CONTEXT"
        color="#f59e0b"
      />

      {/* Metric 4: Languages Active */}
      <MetricNode
        position={[3.6, 0.4, 0]}
        title="ACTIVE LINGUISTIC NODES"
        value="5"
        subtext="EN / HI / HINGLISH / BN / MR"
        color="#2dd4bf"
      />
    </group>
  );
};
