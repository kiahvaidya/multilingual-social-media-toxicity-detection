import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { MemeAnalysisResult } from '../../types';

interface MemeInvestigationChamberProps {
  imageUrl?: string | null;
  ocrText?: string | null;
  result?: MemeAnalysisResult | null;
  isAnalyzing?: boolean;
}

// Fallback texture for canvas plane
const createPlaceholderCanvas = () => {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 512, 512);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 4;
    ctx.strokeRect(16, 16, 480, 480);
    ctx.fillStyle = '#64748b';
    ctx.font = '28px monospace';
    ctx.fillText('WAITING FOR MEME EVIDENCE', 50, 260);
  }
  return new THREE.CanvasTexture(canvas);
};

export const MemeInvestigationChamber: React.FC<MemeInvestigationChamberProps> = ({
  imageUrl,
  ocrText,
  result,
  isAnalyzing = false,
}) => {
  const chamberRef = useRef<THREE.Group>(null);
  const textStreamRef = useRef<THREE.Points>(null);
  const imageStreamRef = useRef<THREE.Points>(null);
  const fusionNodeRef = useRef<THREE.Mesh>(null);
  const placeholderTex = useMemo(() => createPlaceholderCanvas(), []);

  // Try texture or fallback
  const texture = useMemo(() => {
    if (!imageUrl) return placeholderTex;
    try {
      const loader = new THREE.TextureLoader();
      return loader.load(imageUrl);
    } catch {
      return placeholderTex;
    }
  }, [imageUrl, placeholderTex]);

  // Particle Stream for Text/OCR -> MuRIL (Cyan/Blue stream)
  const [textStreamPos, textCount] = useMemo(() => {
    const n = 140;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const t = i / n;
      pos[i * 3] = -1.2 + t * 2.4 + (Math.random() - 0.5) * 0.2;
      pos[i * 3 + 1] = 0.8 - t * 0.8 + (Math.random() - 0.5) * 0.2;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
    }
    return [pos, n];
  }, []);

  // Particle Stream for Image -> CLIP (Amber/Purple stream)
  const [imageStreamPos, imageCount] = useMemo(() => {
    const n = 140;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const t = i / n;
      pos[i * 3] = -1.2 + t * 2.4 + (Math.random() - 0.5) * 0.2;
      pos[i * 3 + 1] = -0.8 + t * 0.8 + (Math.random() - 0.5) * 0.2;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
    }
    return [pos, n];
  }, []);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (chamberRef.current) {
      chamberRef.current.position.y = Math.sin(t * 0.8) * 0.04;
    }

    // Animate flow of text particles
    if (textStreamRef.current) {
      const pos = textStreamRef.current.geometry.attributes.position;
      for (let i = 0; i < textCount; i++) {
        let x = pos.array[i * 3] + 0.035;
        if (x > 1.2) x = -1.2;
        pos.array[i * 3] = x;
      }
      pos.needsUpdate = true;
    }

    // Animate flow of visual particles
    if (imageStreamRef.current) {
      const pos = imageStreamRef.current.geometry.attributes.position;
      for (let i = 0; i < imageCount; i++) {
        let x = pos.array[i * 3] + 0.035;
        if (x > 1.2) x = -1.2;
        pos.array[i * 3] = x;
      }
      pos.needsUpdate = true;
    }

    // Pulse fusion node
    if (fusionNodeRef.current) {
      const scale = isAnalyzing 
        ? 1.0 + Math.sin(t * 8.0) * 0.15 
        : 1.0 + Math.sin(t * 2.0) * 0.05;
      fusionNodeRef.current.scale.set(scale, scale, scale);
      fusionNodeRef.current.rotation.y = t * 0.8;
      fusionNodeRef.current.rotation.x = t * 0.4;
    }
  });

  const interactionType = result?.interaction_type || 'EXPLICIT_HATE';
  const fusionColor = interactionType === 'CONTEXTUAL_INVERSION' 
    ? '#2dd4bf' // Turquoise calm
    : interactionType === 'EMERGENT_TOXICITY' 
    ? '#f43f5e' // Crimson emergent
    : result?.is_hateful ? '#f43f5e' : '#38bdf8';

  return (
    <group ref={chamberRef} position={[0, 0, 0]}>
      {/* 1. FLOATING PHYSICAL MEME EVIDENCE PLANE */}
      <group position={[-1.8, 0, 0]}>
        {/* Frame Bevel */}
        <mesh position={[0, 0, -0.05]}>
          <boxGeometry args={[2.08, 2.08, 0.08]} />
          <meshStandardMaterial color="#0f172a" roughness={0.4} metalness={0.7} />
        </mesh>

        {/* Reticle bounding corners */}
        <lineSegments position={[0, 0, 0.02]}>
          <edgesGeometry args={[new THREE.PlaneGeometry(2.1, 2.1)]} />
          <lineBasicMaterial color="#38bdf8" transparent opacity={0.4} />
        </lineSegments>

        {/* Image Plane */}
        <mesh position={[0, 0, 0]}>
          <planeGeometry args={[2.0, 2.0]} />
          <meshBasicMaterial map={texture} side={THREE.DoubleSide} />
        </mesh>

        <Text
          position={[0, -1.2, 0]}
          fontSize={0.09}
          color="#94a3b8"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {imageUrl ? 'EVIDENCE: INGESTED' : 'WAITING FOR MEME UPLOAD'}
        </Text>
      </group>

      {/* 2. DUAL REPRESENTATION NODES */}
      {/* Node A: MuRIL Text/OCR Representation */}
      <group position={[0.2, 0.9, 0]}>
        <mesh>
          <sphereGeometry args={[0.22, 16, 16]} />
          <meshStandardMaterial color="#38bdf8" wireframe roughness={0.3} />
        </mesh>
        <Text
          position={[0, 0.38, 0]}
          fontSize={0.1}
          color="#38bdf8"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {`MuRIL (768-d): ${result ? (result.text_only_score * 100).toFixed(0) + '%' : 'READY'}`}
        </Text>
      </group>

      {/* Node B: CLIP ViT-B/32 Visual Context Representation */}
      <group position={[0.2, -0.9, 0]}>
        <mesh>
          <sphereGeometry args={[0.22, 16, 16]} />
          <meshStandardMaterial color="#f59e0b" wireframe roughness={0.3} />
        </mesh>
        <Text
          position={[0, -0.38, 0]}
          fontSize={0.1}
          color="#f59e0b"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {`CLIP (512-d): ${result ? (result.image_only_score * 100).toFixed(0) + '%' : 'READY'}`}
        </Text>
      </group>

      {/* Stream A: Text particles */}
      <points ref={textStreamRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[textStreamPos, 3]}
          />
        </bufferGeometry>
        <pointsMaterial size={0.045} color="#38bdf8" transparent opacity={0.8} />
      </points>

      {/* Stream B: Image particles */}
      <points ref={imageStreamRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[imageStreamPos, 3]}
          />
        </bufferGeometry>
        <pointsMaterial size={0.045} color="#f59e0b" transparent opacity={0.8} />
      </points>

      {/* 3. CENTERPIECE: 4-WAY CROSS-MODAL FUSION CORE */}
      <group position={[2.0, 0, 0]}>
        {/* Interaction Orbit Rings */}
        <mesh rotation={[Math.PI / 4, 0, 0]}>
          <torusGeometry args={[0.7, 0.015, 16, 48]} />
          <meshBasicMaterial color={fusionColor} transparent opacity={0.4} />
        </mesh>
        <mesh rotation={[-Math.PI / 4, Math.PI / 3, 0]}>
          <torusGeometry args={[0.85, 0.015, 16, 48]} />
          <meshBasicMaterial color="#94a3b8" transparent opacity={0.25} />
        </mesh>

        {/* 4-Way Interaction Fusion Core Node */}
        <mesh ref={fusionNodeRef}>
          <dodecahedronGeometry args={[0.38, 0]} />
          <meshStandardMaterial
            color={fusionColor}
            emissive={fusionColor}
            emissiveIntensity={0.65}
            roughness={0.1}
            metalness={0.9}
            wireframe
          />
        </mesh>

        {/* Energy Strands connecting Text and Image nodes to Fusion core */}
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[new Float32Array([-1.8, 0.9, 0, 0, 0, 0]), 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#38bdf8" transparent opacity={0.6} />
        </line>
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[new Float32Array([-1.8, -0.9, 0, 0, 0, 0]), 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#f59e0b" transparent opacity={0.6} />
        </line>

        {/* Fusion Equation Readout */}
        <Text
          position={[0, 1.05, 0]}
          fontSize={0.1}
          color="#f8fafc"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          [t_proj || v_proj || t*v || |t-v|]
        </Text>

        <Text
          position={[0, -1.05, 0]}
          fontSize={0.11}
          color={fusionColor}
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {result 
            ? `SCORE: ${(result.multimodal_score * 100).toFixed(1)}% | T*=${result.threshold.toFixed(2)}`
            : 'FUSION CHAMBER READY'}
        </Text>
      </group>
    </group>
  );
};
