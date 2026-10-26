import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, Sphere, Text, Line } from '@react-three/drei';
import * as THREE from 'three';

// Particle Field surrounding the AI Network
function NeuralParticles({ count = 200 }) {
  const pointsRef = useRef<THREE.Points>(null!);

  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const colorCyan = new THREE.Color('#00f2fe');
    const colorPurple = new THREE.Color('#8b5cf6');

    for (let i = 0; i < count; i++) {
      const radius = 3.5 + Math.random() * 4.5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      pos[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = radius * Math.cos(phi);

      const mixed = Math.random() > 0.5 ? colorCyan : colorPurple;
      col[i * 3] = mixed.r;
      col[i * 3 + 1] = mixed.g;
      col[i * 3 + 2] = mixed.b;
    }
    return [pos, col];
  }, [count]);

  useFrame((state, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.05;
      pointsRef.current.rotation.x += delta * 0.02;
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
        size={0.045}
        vertexColors
        transparent
        opacity={0.65}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// Hub Node with pulsing glow rings
interface HubNodeProps {
  position: [number, number, number];
  color: string;
  label: string;
  sublabel: string;
  size?: number;
}

function HubNode({ position, color, label, sublabel, size = 0.35 }: HubNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const ringRef = useRef<THREE.Mesh>(null!);

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.4;
      meshRef.current.rotation.z += delta * 0.2;
    }
    if (ringRef.current) {
      ringRef.current.rotation.x += delta * 0.8;
      ringRef.current.rotation.y += delta * 0.5;
    }
  });

  return (
    <group position={position}>
      <Float speed={2} rotationIntensity={0.2} floatIntensity={0.3}>
        {/* Core Sphere */}
        <Sphere ref={meshRef} args={[size, 24, 24]}>
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.7}
            roughness={0.2}
            metalness={0.8}
            wireframe={true}
          />
        </Sphere>
        {/* Inner solid sphere */}
        <Sphere args={[size * 0.65, 16, 16]}>
          <meshBasicMaterial color={color} />
        </Sphere>
        {/* Pulsing Outer Wireframe Ring */}
        <mesh ref={ringRef}>
          <torusGeometry args={[size * 1.5, 0.015, 12, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.4} />
        </mesh>
        {/* Text Labels in 3D Space */}
        <Text
          position={[0, -size - 0.25, 0]}
          fontSize={0.16}
          color="#f8fafc"
          anchorX="center"
          anchorY="top"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_al061-VTx-q74.woff"
        >
          {label}
        </Text>
        <Text
          position={[0, -size - 0.45, 0]}
          fontSize={0.11}
          color="#94a3b8"
          anchorX="center"
          anchorY="top"
        >
          {sublabel}
        </Text>
      </Float>
    </group>
  );
}

// Main Network Graph
function NetworkGraph() {
  const groupRef = useRef<THREE.Group>(null!);

  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.12;
    }
  });

  // Hub Positions
  const centerPos: [number, number, number] = [0, 0, 0];
  const textPos: [number, number, number] = [-2.8, 1.2, 0];
  const imagePos: [number, number, number] = [2.8, 1.2, 0];
  const jevPos: [number, number, number] = [0, 2.6, 0.5];
  const decisionPos: [number, number, number] = [0, -2.4, 0];

  return (
    <group ref={groupRef}>
      {/* 1. Center Core: Multimodal Intelligence Fusion */}
      <HubNode
        position={centerPos}
        color="#00f2fe"
        label="FUSION ENGINE"
        sublabel="Interaction Matrix [t*v, |t-v|]"
        size={0.65}
      />

      {/* 2. Left Node: MuRIL Text Stream */}
      <HubNode
        position={textPos}
        color="#38bdf8"
        label="MuRIL ENCODER"
        sublabel="Multilingual Text (768-D)"
        size={0.4}
      />

      {/* 3. Right Node: CLIP Vision Stream */}
      <HubNode
        position={imagePos}
        color="#a855f7"
        label="CLIP ViT-B/32"
        sublabel="Visual Embeddings (512-D)"
        size={0.4}
      />

      {/* 4. Top Node: JEV Contextual Layer */}
      <HubNode
        position={jevPos}
        color="#f59e0b"
        label="JEV REASONING"
        sublabel="Context & Intent Gate"
        size={0.38}
      />

      {/* 5. Bottom Node: Tri-Tier Moderation Decision */}
      <HubNode
        position={decisionPos}
        color="#10b981"
        label="MODERATION CORE"
        sublabel="Safe / Review / High Risk"
        size={0.45}
      />

      {/* Dynamic Connection Lines with glowing gradient */}
      <Line points={[textPos, centerPos]} color="#38bdf8" lineWidth={2} transparent opacity={0.6} />
      <Line points={[imagePos, centerPos]} color="#a855f7" lineWidth={2} transparent opacity={0.6} />
      <Line points={[centerPos, jevPos]} color="#f59e0b" lineWidth={2} transparent opacity={0.5} />
      <Line points={[centerPos, decisionPos]} color="#00f2fe" lineWidth={2.5} transparent opacity={0.8} />
      <Line points={[jevPos, decisionPos]} color="#10b981" lineWidth={1.5} dashed dashScale={10} transparent opacity={0.4} />

      {/* Floating Particles */}
      <NeuralParticles count={300} />
    </group>
  );
}

export function NetworkScene() {
  return (
    <div className="relative w-full h-full min-h-[420px] rounded-2xl overflow-hidden glass-panel border border-cyan-500/20 shadow-glow-cyan/20">
      {/* Decorative Cyber Overlays */}
      <div className="absolute top-4 left-4 z-10 flex items-center space-x-2">
        <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
        <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
        <span className="text-xs font-mono uppercase tracking-wider text-cyan-300 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-500/30">
          LIVE NEURAL GRAPH // MULTIMODAL TOPOLOGY
        </span>
      </div>

      <div className="absolute bottom-4 right-4 z-10 text-[11px] font-mono text-slate-400 bg-dark-900/90 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-800 flex items-center space-x-3">
        <span>3D INTERACTION ACTIVE</span>
        <span className="text-slate-600">|</span>
        <span>DRAG TO ROTATE</span>
      </div>

      <Canvas
        camera={{ position: [0, 0, 7.8], fov: 48 }}
        className="cursor-grab active:cursor-grabbing"
      >
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1.2} color="#00f2fe" />
        <pointLight position={[-10, -10, -10]} intensity={0.8} color="#8b5cf6" />
        <pointLight position={[0, 5, 0]} intensity={0.9} color="#f59e0b" />
        
        <NetworkGraph />
        
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={(3 * Math.PI) / 4}
          dampingFactor={0.05}
        />
      </Canvas>
    </div>
  );
}
export default NetworkScene;
