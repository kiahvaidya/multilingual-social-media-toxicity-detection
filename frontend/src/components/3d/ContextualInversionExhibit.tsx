import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface ContextualInversionExhibitProps {
  modeType: 'INVERSION' | 'EMERGENT';
  stepProgress?: number; // 0 to 1
}

export const ContextualInversionExhibit: React.FC<ContextualInversionExhibitProps> = ({
  modeType,
  stepProgress = 0.5,
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const textFieldRef = useRef<THREE.Mesh>(null);
  const imageFieldRef = useRef<THREE.Mesh>(null);
  const fusionRingRef = useRef<THREE.Mesh>(null);

  const isInversion = modeType === 'INVERSION';

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(t * 0.6) * 0.05;
    }

    if (textFieldRef.current) {
      textFieldRef.current.rotation.y = t * 0.4;
    }
    if (imageFieldRef.current) {
      imageFieldRef.current.rotation.y = -t * 0.5;
    }
    if (fusionRingRef.current) {
      fusionRingRef.current.rotation.z = t * 0.6;
    }
  });

  // Dynamic visual states based on inversion vs emergent
  // Inversion: Text starts red/toxic, Image is cool cyan/benign, Fused relaxes to cool cyan (Safe)
  // Emergent: Text starts cool/benign, Image is toxic red, Fused spikes to crimson (High Risk)
  const textColor = isInversion ? '#f43f5e' : '#38bdf8';
  const textLabel = isInversion 
    ? 'LEXICAL SIGNAL: HIGH ("bring your dog we have chicken?")'
    : 'LEXICAL SIGNAL: LOW ("SHUT UP OSAMA! HE IS MY SON!")';

  const imageColor = isInversion ? '#2dd4bf' : '#f43f5e';
  const imageLabel = isInversion
    ? 'VISUAL CONTEXT: BENIGN (Cute puppy picnic photo)'
    : 'VISUAL CONTEXT: HATE TEMPLATE (Terrorism/Islamophobic trope)';

  const resultColor = isInversion ? '#2dd4bf' : '#f43f5e';
  const resultVerdict = isInversion 
    ? 'CONTEXTUAL INVERSION: DEFUSED -> SAFE (P_cal = 0.298 < T*)'
    : 'EMERGENT TOXICITY: CAUGHT -> HIGH RISK (P_cal = 0.827 >= T*)';

  return (
    <group ref={groupRef} position={[-0.5, 0, 0]}>
      {/* Central Interactive Exhibit Display */}
      {/* 1. Left: Text Modality Node */}
      <group position={[-1.8, 0.8, 0]}>
        <mesh ref={textFieldRef}>
          <icosahedronGeometry args={[0.45, 1]} />
          <meshStandardMaterial
            color={textColor}
            emissive={textColor}
            emissiveIntensity={0.5}
            roughness={0.2}
            metalness={0.8}
            wireframe
          />
        </mesh>
        <Text
          position={[0, 0.7, 0]}
          fontSize={0.11}
          color={textColor}
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {textLabel}
        </Text>
      </group>

      {/* 2. Left-Bottom: Image Modality Node */}
      <group position={[-1.8, -0.8, 0]}>
        <mesh ref={imageFieldRef}>
          <octahedronGeometry args={[0.45, 1]} />
          <meshStandardMaterial
            color={imageColor}
            emissive={imageColor}
            emissiveIntensity={0.5}
            roughness={0.2}
            metalness={0.8}
            wireframe
          />
        </mesh>
        <Text
          position={[0, -0.7, 0]}
          fontSize={0.11}
          color={imageColor}
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {imageLabel}
        </Text>
      </group>

      {/* Interaction Energy Conduits */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([-1.35, 0.8, 0, 0.8, 0, 0]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={textColor} transparent opacity={0.6} />
      </line>
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([-1.35, -0.8, 0, 0.8, 0, 0]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={imageColor} transparent opacity={0.6} />
      </line>

      {/* 3. Right: Combined Emergent / Inverted State Node */}
      <group position={[1.5, 0, 0]}>
        <mesh ref={fusionRingRef}>
          <torusGeometry args={[0.9, 0.02, 16, 64]} />
          <meshBasicMaterial color={resultColor} transparent opacity={0.5} />
        </mesh>

        <mesh>
          <dodecahedronGeometry args={[0.55, 0]} />
          <meshStandardMaterial
            color={resultColor}
            emissive={resultColor}
            emissiveIntensity={0.7}
            roughness={0.1}
            metalness={0.9}
            wireframe
          />
        </mesh>

        <Text
          position={[0, 1.25, 0]}
          fontSize={0.12}
          color="#f8fafc"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {isInversion ? 'LEXICAL INTENT NEUTRALIZATION' : 'CROSS-MODAL HATE EMERGENCE'}
        </Text>

        <Text
          position={[0, -1.25, 0]}
          fontSize={0.11}
          color={resultColor}
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_ad063nRzCd_bE.woff"
          anchorX="center"
        >
          {resultVerdict}
        </Text>
      </group>
    </group>
  );
};
