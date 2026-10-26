import React, { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

export type SceneMode = 'LANDING' | 'OVERVIEW' | 'TEXT' | 'MEME' | 'INVERSION' | 'QUEUE' | 'RESEARCH' | 'SOCIAL' | 'PROFILE_WATCH';

interface CameraControllerProps {
  mode: SceneMode;
}

// Camera coordinate targets for each scene
const CAMERA_PRESETS: Record<SceneMode, { pos: [number, number, number]; target: [number, number, number]; fov: number }> = {
  LANDING: {
    pos: [0, 1.5, 14],
    target: [0, 0, 0],
    fov: 48,
  },
  OVERVIEW: {
    pos: [0, 4.5, 12],
    target: [0, 0.5, 0],
    fov: 50,
  },
  TEXT: {
    pos: [-2.2, 0.6, 7.5],
    target: [0, 0.2, 0],
    fov: 45,
  },
  MEME: {
    pos: [0, 0.5, 7.2],
    target: [0, 0.1, 0],
    fov: 46,
  },
  INVERSION: {
    pos: [1.8, 0.8, 8],
    target: [0, 0.2, 0],
    fov: 48,
  },
  QUEUE: {
    pos: [-3.2, 3.8, 10.5],
    target: [0, 1.2, 0],
    fov: 52,
  },
  RESEARCH: {
    pos: [0, 6.5, 13],
    target: [0, 1.8, 0],
    fov: 50,
  },
  SOCIAL: {
    pos: [-2.2, 0.6, 7.5],
    target: [0, 0.2, 0],
    fov: 45,
  },
  PROFILE_WATCH: {
    pos: [0, 1.8, 9.5],
    target: [0, 0.2, 0],
    fov: 47,
  },
};

export const CameraController: React.FC<CameraControllerProps> = ({ mode }) => {
  const { camera } = useThree();
  const currentTarget = useRef(new THREE.Vector3(0, 0, 0));
  const mousePos = useRef({ x: 0, y: 0 });

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mousePos.current.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mousePos.current.y = (e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useFrame((_, delta) => {
    const preset = CAMERA_PRESETS[mode] || CAMERA_PRESETS.LANDING;
    const targetVector = new THREE.Vector3(...preset.target);

    // Subtle parallax offset based on cursor position
    const parallaxX = mousePos.current.x * 0.45;
    const parallaxY = -mousePos.current.y * 0.3;

    const desiredPos = new THREE.Vector3(
      preset.pos[0] + parallaxX,
      preset.pos[1] + parallaxY,
      preset.pos[2]
    );

    // Smooth cinematic damping
    const dampSpeed = Math.min(delta * 2.8, 0.15);
    camera.position.lerp(desiredPos, dampSpeed);
    currentTarget.current.lerp(targetVector, dampSpeed);
    camera.lookAt(currentTarget.current);

    if (camera instanceof THREE.PerspectiveCamera && camera.fov !== preset.fov) {
      camera.fov = THREE.MathUtils.lerp(camera.fov, preset.fov, dampSpeed);
      camera.updateProjectionMatrix();
    }
  });

  return null;
};
