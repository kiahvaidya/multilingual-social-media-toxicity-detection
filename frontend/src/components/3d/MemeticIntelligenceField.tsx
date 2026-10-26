import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface MemeticIntelligenceFieldProps {
  verdictRisk?: 'Safe' | 'Review' | 'High Risk' | null;
  activityLevel?: number; // 0 to 1
}

export const MemeticIntelligenceField: React.FC<MemeticIntelligenceFieldProps> = ({
  verdictRisk,
  activityLevel = 0.5,
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const count = 3200;

  // Generate particle coordinates and semantic clusters
  const [positions, colors, scales, baseData] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const sca = new Float32Array(count);
    const data: { origX: number; origY: number; origZ: number; speed: number; phase: number; cluster: number }[] = [];

    const colorEnglish = new THREE.Color('#94a3b8');     // Slate/steel for English
    const colorHindi = new THREE.Color('#f59e0b');       // Amber/warm gold for Hindi
    const colorHinglish = new THREE.Color('#2dd4bf');    // Cyan/teal for Hinglish banter
    const colorNeutral = new THREE.Color('#475569');     // Deep slate background

    for (let i = 0; i < count; i++) {
      // 3 semantic clusters + diffuse background
      const cluster = i % 4;
      let x = 0, y = 0, z = 0;

      if (cluster === 0) {
        // English stream: broad horizontal arc
        const u = Math.random() * Math.PI * 2;
        const r = 4.5 + Math.random() * 3.5;
        x = Math.cos(u) * r + (Math.random() - 0.5) * 1.5;
        y = (Math.random() - 0.5) * 3.0;
        z = Math.sin(u) * (r * 0.7) + (Math.random() - 0.5) * 2.0;
        colorEnglish.toArray(col, i * 3);
      } else if (cluster === 1) {
        // Hindi stream: ascending helical strand
        const t = Math.random() * 8.0 - 4.0;
        const angle = t * 1.2;
        const radius = 3.2 + Math.random() * 1.8;
        x = Math.cos(angle) * radius + (Math.random() - 0.5);
        y = t * 0.8 + (Math.random() - 0.5) * 0.8;
        z = Math.sin(angle) * radius + (Math.random() - 0.5);
        colorHindi.toArray(col, i * 3);
      } else if (cluster === 2) {
        // Hinglish/Context stream: diagonal bridge
        const t = (Math.random() - 0.5) * 7.0;
        x = t + (Math.random() - 0.5) * 1.5;
        y = t * 0.4 + (Math.random() - 0.5) * 2.0;
        z = -t * 0.6 + (Math.random() - 0.5) * 2.0;
        colorHinglish.toArray(col, i * 3);
      } else {
        // Ambient background cosmos
        x = (Math.random() - 0.5) * 22.0;
        y = (Math.random() - 0.5) * 14.0;
        z = (Math.random() - 0.5) * 18.0;
        colorNeutral.toArray(col, i * 3);
      }

      pos[i * 3] = x;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = z;

      sca[i] = Math.random() * 1.8 + 0.6;
      data.push({
        origX: x,
        origY: y,
        origZ: z,
        speed: 0.2 + Math.random() * 0.4,
        phase: Math.random() * Math.PI * 2,
        cluster,
      });
    }

    return [pos, col, sca, data];
  }, []);

  // Frame animation
  useFrame(({ clock, pointer }) => {
    if (!pointsRef.current) return;
    const posAttr = pointsRef.current.geometry.attributes.position;
    const colAttr = pointsRef.current.geometry.attributes.color;
    const t = clock.getElapsedTime();

    // Determine field tension
    let speedMult = 1.0;
    let waveAmp = 0.08;

    if (verdictRisk === 'High Risk') {
      speedMult = 2.4;
      waveAmp = 0.24;
    } else if (verdictRisk === 'Review') {
      speedMult = 1.6;
      waveAmp = 0.16;
    } else if (verdictRisk === 'Safe') {
      speedMult = 0.6;
      waveAmp = 0.04;
    }

    // Pointer 3D ray projection approximation
    const mouseX = pointer.x * 6.0;
    const mouseY = pointer.y * 4.0;

    for (let i = 0; i < count; i++) {
      const d = baseData[i];
      const i3 = i * 3;

      // Natural harmonic wave
      const wave = Math.sin(t * d.speed * speedMult + d.phase) * waveAmp;
      const wave2 = Math.cos(t * d.speed * 0.7 * speedMult + d.phase) * (waveAmp * 0.8);

      let px = d.origX + wave;
      let py = d.origY + wave2;
      let pz = d.origZ + wave;

      // Cursor gentle repulsion
      const dx = px - mouseX;
      const dy = py - mouseY;
      const distSq = dx * dx + dy * dy;
      if (distSq < 9.0) {
        const force = (1.0 - Math.sqrt(distSq) / 3.0) * 0.65;
        px += dx * force;
        py += dy * force;
      }

      posAttr.array[i3] = px;
      posAttr.array[i3 + 1] = py;
      posAttr.array[i3 + 2] = pz;

      // Dynamic color shift during High Risk / Review
      if (verdictRisk === 'High Risk' && d.cluster !== 3) {
        // Shift toward crimson
        colAttr.array[i3] = THREE.MathUtils.lerp(colAttr.array[i3], 0.95, 0.04);
        colAttr.array[i3 + 1] = THREE.MathUtils.lerp(colAttr.array[i3 + 1], 0.2, 0.04);
        colAttr.array[i3 + 2] = THREE.MathUtils.lerp(colAttr.array[i3 + 2], 0.25, 0.04);
      } else if (verdictRisk === 'Safe' && d.cluster !== 3) {
        // Shift toward cool cyan/teal
        colAttr.array[i3] = THREE.MathUtils.lerp(colAttr.array[i3], 0.18, 0.03);
        colAttr.array[i3 + 1] = THREE.MathUtils.lerp(colAttr.array[i3 + 1], 0.83, 0.03);
        colAttr.array[i3 + 2] = THREE.MathUtils.lerp(colAttr.array[i3 + 2], 0.75, 0.03);
      }
    }

    posAttr.needsUpdate = true;
    if (verdictRisk) {
      colAttr.needsUpdate = true;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.065}
        vertexColors
        transparent
        opacity={0.78}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
};
