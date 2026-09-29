import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Bounds, Line, OrbitControls } from "@react-three/drei";

function layoutExperts(experts, palette) {
  const maxActivation = Math.max(...experts.map((expert) => expert.activation), 1);
  return experts.map((expert, index) => {
    const total = experts.length || 1;
    const phi = Math.acos(1 - 2 * (index + 0.5) / total);
    const theta = Math.PI * (1 + Math.sqrt(5)) * index;
    const spread = 2.4;
    const wobble = ((expert.expert_id * 17) % 100) / 100 - 0.5;
    return {
      ...expert,
      x: spread * Math.sin(phi) * Math.cos(theta) + wobble * 0.18,
      y: spread * Math.sin(phi) * Math.sin(theta) + wobble * 0.14,
      z: spread * Math.cos(phi) + wobble * 0.16,
      size: 0.09 + (expert.activation / maxActivation) * 0.16,
      color: palette[index % palette.length],
    };
  });
}

function ambientNodes(count = 48) {
  return Array.from({ length: count }, (_, index) => {
    const phi = Math.acos(1 - 2 * (index + 0.5) / count);
    const theta = Math.PI * (1 + Math.sqrt(5)) * (index + 11);
    const radius = 11 + (index % 5) * 0.5;
    return {
      x: radius * Math.sin(phi) * Math.cos(theta),
      y: radius * Math.sin(phi) * Math.sin(theta),
      z: radius * Math.cos(phi),
    };
  });
}

function ExpertNode({ node, selected, onSelect }) {
  const ringRef = useRef(null);
  useFrame((_, delta) => {
    if (ringRef.current && selected) {
      ringRef.current.rotation.y += delta * 0.8;
      ringRef.current.rotation.x += delta * 0.35;
    }
  });
  return (
    <group position={[node.x, node.y, node.z]}>
      {selected && (
        <mesh ref={ringRef}>
          <torusGeometry args={[node.size * 1.55, 0.018, 12, 36]} />
          <meshBasicMaterial color={node.color} transparent opacity={0.75} />
        </mesh>
      )}
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          onSelect(node.expert_id);
        }}
        onPointerOver={(event) => {
          event.stopPropagation();
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "default";
        }}
      >
        <sphereGeometry args={[node.size, 28, 28]} />
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={selected ? 0.95 : 0.42}
          roughness={0.28}
          metalness={0.35}
        />
      </mesh>
    </group>
  );
}

function AmbientField() {
  const nodes = useMemo(() => ambientNodes(), []);
  const links = useMemo(() => nodes.slice(0, 28).map((node, index) => {
    const target = nodes[(index * 5 + 7) % nodes.length];
    return [node, target];
  }), [nodes]);
  return (
    <group>
      {nodes.map((node, index) => (
        <mesh key={index} position={[node.x, node.y, node.z]}>
          <sphereGeometry args={[0.035, 8, 8]} />
          <meshBasicMaterial color="#5f7f97" transparent opacity={0.22} />
        </mesh>
      ))}
      {links.map(([start, end], index) => (
        <Line
          key={index}
          points={[[start.x, start.y, start.z], [end.x, end.y, end.z]]}
          color="#3d566c"
          transparent
          opacity={0.12}
          lineWidth={0.5}
        />
      ))}
    </group>
  );
}

function GraphScene({ experts, selectedId, onSelect, palette }) {
  const nodes = useMemo(() => layoutExperts(experts, palette), [experts, palette]);
  const links = useMemo(() => {
    if (nodes.length < 2) return [];
    return nodes.map((node, index) => {
      const next = nodes[(index + 1) % nodes.length];
      return { from: node, to: next };
    });
  }, [nodes]);
  return (
    <>
      <ambientLight intensity={0.42} />
      <pointLight position={[9, 8, 7]} intensity={1.35} color="#dff8ff" />
      <pointLight position={[-8, -5, -6]} intensity={0.55} color="#48d3cf" />
      <AmbientField />
      <Bounds fit clip observe margin={1.65}>
        <group>
          {links.map((link, index) => (
            <Line
              key={`${link.from.expert_id}-${index}`}
              points={[[link.from.x, link.from.y, link.from.z], [link.to.x, link.to.y, link.to.z]]}
              color={link.from.color}
              transparent
              opacity={0.42}
              lineWidth={1.2}
            />
          ))}
          {nodes.map((node) => (
            <ExpertNode
              key={node.expert_id}
              node={node}
              selected={node.expert_id === selectedId}
              onSelect={onSelect}
            />
          ))}
        </group>
      </Bounds>
      <OrbitControls makeDefault enablePan enableZoom enableRotate minDistance={1.5} maxDistance={12} />
    </>
  );
}

export default function NeuronGraph3D({ experts, selectedId, onSelect, palette, expertCountLabel }) {
  return (
    <div className="neuron-field neuron-graph-3d">
      {experts.length ? (
        <Canvas className="neuron-graph-canvas" camera={{ position: [0, 0, 6], fov: 52 }} dpr={[1, 2]}>
          <color attach="background" args={["#07111f"]} />
          <fog attach="fog" args={["#07111f", 18, 30]} />
          <GraphScene experts={experts} selectedId={selectedId} onSelect={onSelect} palette={palette} />
        </Canvas>
      ) : (
        <div className="neuron-graph-empty">Run an inference to map active experts in 3D.</div>
      )}
      <div className="field-caption field-hint">Drag to orbit - scroll to zoom - shift-drag to pan - click a node to inspect</div>
      <div className="field-caption">{expertCountLabel}</div>
    </div>
  );
}
