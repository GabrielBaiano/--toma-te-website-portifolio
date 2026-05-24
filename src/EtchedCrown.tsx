import { useRef, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

const vertShader = `
  varying vec3 vNormal;
  varying vec2 vUv;
  varying vec4 vScreenPos;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    vScreenPos = gl_Position;
  }
`

const fragShader = `
  uniform vec3 uBase;
  uniform vec3 uLine;
  uniform vec2 uResolution;
  varying vec3 vNormal;
  varying vec4 vScreenPos;

  // 4x4 Bayer Matrix for Ordered Dithering (Float-only for cross-device compatibility)
  float bayer4x4(vec2 p) {
    vec2 pos = floor(mod(p, 4.0));
    float x = pos.x;
    float y = pos.y;
    
    if (y == 0.0) {
      if (x == 0.0) return 0.0625;
      if (x == 1.0) return 0.5625;
      if (x == 2.0) return 0.1875;
      if (x == 3.0) return 0.6875;
    }
    if (y == 1.0) {
      if (x == 0.0) return 0.8125;
      if (x == 1.0) return 0.3125;
      if (x == 2.0) return 0.9375;
      if (x == 3.0) return 0.4375;
    }
    if (y == 2.0) {
      if (x == 0.0) return 0.25;
      if (x == 1.0) return 0.75;
      if (x == 2.0) return 0.125;
      if (x == 3.0) return 0.625;
    }
    if (y == 3.0) {
      if (x == 0.0) return 1.0;
      if (x == 1.0) return 0.5;
      if (x == 2.0) return 0.875;
      if (x == 3.0) return 0.375;
    }
    return 0.0;
  }

  void main() {
    // Basic lighting calculation
    vec3 keyLight = normalize(vec3(1.5, 1.0, 1.0));
    vec3 fillLight = normalize(vec3(-1.0, -0.5, -0.5));
    
    float dotNL = dot(vNormal, keyLight);
    float key = max(dotNL, 0.0);
    float fill = max(dot(vNormal, fillLight), 0.0) * 0.1;
    
    // Smooth the lighting value a bit before dithering
    float lit = clamp(0.02 + key * 1.0 + fill, 0.0, 1.0);

    // Screen-space coordinates for dithering
    vec2 screenCoord = (vScreenPos.xy / vScreenPos.w * 0.5 + 0.5) * uResolution;
    
    // Pixelation factor: Divide screenCoord to make dither dots larger
    float pixelSize = 3.0; 
    float limit = bayer4x4(screenCoord / pixelSize);
    
    // Obra Dinn "Hard" 1-bit thresholding
    float ramp = lit > limit ? 1.0 : 0.0;

    // Rim/Outline effect
    // We use the view-space normal to create a dark border
    float rim = 1.0 - abs(vNormal.z);
    float outline = step(0.7, rim);
    
    // Combine lighting and outline
    float final = ramp;
    if (outline > 0.5) final = 0.0;

    vec3 col = mix(uLine, uBase, final);
    gl_FragColor = vec4(col, 1.0);
  }
`

// Single continuous mesh — no seam between band and peaks
function createCrown(): THREE.BufferGeometry {
  const peaks = 5, R = 2.2, thick = 0.35
  const bandH = 0.65, peakH = 1.05
  const totalH = bandH + peakH
  const Ri = R - thick
  const rS = 300, hS = 150
  const yOff = -totalH * 0.3

  function maxH(theta: number): number {
    const pw = (2 * Math.PI) / peaks
    const seg = ((theta % (2 * Math.PI)) + 2 * Math.PI) % pw
    const c = pw / 2
    return bandH + peakH * Math.pow(1.0 - Math.abs(seg - c) / c, 0.6)
  }

  const pos: number[] = [], nrm: number[] = [], uvs: number[] = [], idx: number[] = []
  let vi = 0

  function buildSurf(radius: number, nSign: number): number[][] {
    const grid: number[][] = []
    for (let j = 0; j <= hS; j++) {
      const row: number[] = []
      for (let i = 0; i <= rS; i++) {
        const u = i / rS, theta = u * 2 * Math.PI
        const y = Math.min((j / hS) * totalH, maxH(theta))
        pos.push(Math.cos(theta) * radius, y + yOff, Math.sin(theta) * radius)
        nrm.push(Math.cos(theta) * nSign, 0.1 * nSign, Math.sin(theta) * nSign)
        uvs.push(u, j / hS)
        row.push(vi++)
      }
      grid.push(row)
    }
    return grid
  }

  const outer = buildSurf(R, 1)
  const inner = buildSurf(Ri, -1)

  // Outer faces
  for (let j = 0; j < hS; j++)
    for (let i = 0; i < rS; i++) {
      const a = outer[j][i], b = outer[j][i+1], c = outer[j+1][i], d = outer[j+1][i+1]
      idx.push(a, c, b); idx.push(b, c, d)
    }
  // Inner faces
  for (let j = 0; j < hS; j++)
    for (let i = 0; i < rS; i++) {
      const a = inner[j][i], b = inner[j][i+1], c = inner[j+1][i], d = inner[j+1][i+1]
      idx.push(a, b, c); idx.push(b, d, c)
    }
  // Bottom cap
  for (let i = 0; i < rS; i++) {
    idx.push(outer[0][i], outer[0][i+1], inner[0][i])
    idx.push(outer[0][i+1], inner[0][i+1], inner[0][i])
  }
  // Top cap (zigzag)
  for (let i = 0; i < rS; i++) {
    idx.push(outer[hS][i], inner[hS][i], outer[hS][i+1])
    idx.push(inner[hS][i], inner[hS][i+1], outer[hS][i+1])
  }

  const geo = new THREE.BufferGeometry()
  geo.setIndex(idx)
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3))
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  geo.computeVertexNormals()
  return geo
}

export default function EtchedCrown() {
  const ref = useRef<THREE.Mesh>(null)
  const { size } = useThree()
  const geo = useMemo(() => createCrown(), [])
  const uniforms = useMemo(() => ({
    uBase: { value: new THREE.Color('#f2d80a') },
    uLine: { value: new THREE.Color('#0a0a00') },
    uResolution: { value: new THREE.Vector2(size.width, size.height) }
  }), [])

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    
    // Update resolution uniform in case window resizes
    uniforms.uResolution.value.set(size.width, size.height)

    if (ref.current) {
      ref.current.rotation.y = t * 0.2
      ref.current.rotation.x = 0.3 + Math.sin(t * 0.3) * 0.03
      ref.current.rotation.z = Math.sin(t * 0.2) * 0.02 - 0.1
      ref.current.position.y = Math.sin(t * 0.5) * 0.05
    }
  })

  return (
    <mesh ref={ref} geometry={geo} scale={0.6}>
      <shaderMaterial
        vertexShader={vertShader}
        fragmentShader={fragShader}
        uniforms={uniforms}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}
