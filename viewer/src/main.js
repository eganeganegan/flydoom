import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import "./styles.css";

const ui = Object.fromEntries(
  [
    "viewport", "session-title", "node-count", "edge-count", "synapse-count", "frame-label",
    "peak-type", "peak-meta", "activity-fill", "doom-frame", "action-label", "reward-label",
    "play", "timeline", "time-current", "time-total", "speed", "gain", "orbit",
    "morphology", "fullscreen", "loading", "loading-message",
  ].map((id) => [id, document.getElementById(id)]),
);

const sessionRoot = (new URLSearchParams(window.location.search).get("session") || "/session")
  .replace(/\/$/, "");

const fetchBinary = async (filename, Type) => {
  const response = await fetch(`${sessionRoot}/${filename}`);
  if (!response.ok) throw new Error(`Could not load ${filename} (${response.status})`);
  return new Type(await response.arrayBuffer());
};

const formatNumber = (value) => new Intl.NumberFormat("en-US", { notation: "compact" }).format(value);
const formatTime = (seconds) => {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = (seconds % 60).toFixed(2).padStart(5, "0");
  return `${minutes}:${remainder}`;
};

const lineVertexShader = `
  attribute float owner;
  uniform sampler2D activityMap;
  uniform float nodeCount;
  uniform float gain;
  varying vec3 colorOut;
  varying float alphaOut;
  void main() {
    float u = (owner + 0.5) / nodeCount;
    float activity = clamp(texture2D(activityMap, vec2(u, 0.5)).r * gain, 0.0, 1.0);
    float firing = smoothstep(0.62, 1.0, activity);
    colorOut = mix(vec3(0.055, 0.25, 0.27), vec3(0.58, 1.0, 0.98), smoothstep(0.06, 0.72, activity));
    colorOut = mix(colorOut, vec3(1.0, 0.56, 0.28), firing);
    alphaOut = 0.025 + activity * 0.90;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const lineFragmentShader = `
  varying vec3 colorOut;
  varying float alphaOut;
  void main() { gl_FragColor = vec4(colorOut, alphaOut); }
`;

const pointVertexShader = `
  attribute float owner;
  attribute float kind;
  uniform sampler2D activityMap;
  uniform float nodeCount;
  uniform float gain;
  uniform float pixelRatio;
  varying vec3 colorOut;
  varying float alphaOut;
  void main() {
    float activity = clamp(texture2D(activityMap, vec2((owner + 0.5) / nodeCount, 0.5)).r * gain, 0.0, 1.0);
    vec3 base = kind > 1.5 ? vec3(0.33, 0.23, 0.16) : (kind > 0.5 ? vec3(0.08, 0.32, 0.39) : vec3(0.045, 0.19, 0.21));
    colorOut = mix(base, vec3(0.62, 1.0, 0.98), smoothstep(0.04, 0.68, activity));
    colorOut = mix(colorOut, vec3(1.0, 0.54, 0.25), smoothstep(0.72, 1.0, activity));
    alphaOut = 0.22 + activity * 0.76;
    vec4 view = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * view;
    gl_PointSize = pixelRatio * (2.3 + 8.5 * activity) * (2.4 / max(1.0, -view.z));
  }
`;

const pointFragmentShader = `
  varying vec3 colorOut;
  varying float alphaOut;
  void main() {
    float radius = length(gl_PointCoord - vec2(0.5));
    if (radius > 0.5) discard;
    float core = 1.0 - smoothstep(0.04, 0.5, radius);
    float glow = 1.0 - smoothstep(0.18, 0.5, radius);
    gl_FragColor = vec4(colorOut * (0.75 + core), alphaOut * (0.25 + glow * 0.75));
  }
`;

function addAtmosphere(scene) {
  const count = 900;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < positions.length; i += 3) {
    const radius = 2.5 + Math.random() * 4.5;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i] = radius * Math.sin(phi) * Math.cos(theta);
    positions[i + 1] = radius * Math.cos(phi) * 0.62;
    positions[i + 2] = radius * Math.sin(phi) * Math.sin(theta);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0x4cb8b3, size: 0.007, transparent: true, opacity: 0.12,
    depthWrite: false, blending: THREE.AdditiveBlending,
  });
  scene.add(new THREE.Points(geometry, material));
}

async function loadSession() {
  const manifestResponse = await fetch(`${sessionRoot}/manifest.json`);
  if (!manifestResponse.ok) {
    throw new Error(
      "No exported session found. Run scripts/export_web_viewer.py, then reload this page.",
    );
  }
  const manifest = await manifestResponse.json();
  const files = manifest.files;
  const loads = [
    fetchBinary(files.positions, Float32Array),
    fetchBinary(files.activity, Float32Array),
    fetchBinary(files.frames, Uint8Array),
    fetchBinary(files.node_kind, Uint8Array),
    fetchBinary(files.edges, Uint32Array),
    fetchBinary(files.edge_strength, Float32Array),
  ];
  if (files.skeleton_segments) {
    loads.push(fetchBinary(files.skeleton_segments, Float32Array));
    loads.push(fetchBinary(files.skeleton_owners, Uint32Array));
  }
  const loaded = await Promise.all(loads);
  return {
    manifest,
    positions: loaded[0], activity: loaded[1], frames: loaded[2], kinds: loaded[3],
    edges: loaded[4], edgeStrength: loaded[5],
    skeletonSegments: loaded[6] || null, skeletonOwners: loaded[7] || null,
  };
}

function buildViewer(data) {
  const { manifest } = data;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x031315);
  scene.fog = new THREE.FogExp2(0x031315, 0.115);
  const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.01, 100);
  camera.position.set(0.15, 0.2, 3.25);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  ui.viewport.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.045;
  controls.minDistance = 1.1;
  controls.maxDistance = 8;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.42;

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.88, 0.48, 0.36));

  const activityValues = new Float32Array(manifest.node_count);
  const activityMap = new THREE.DataTexture(
    activityValues, manifest.node_count, 1, THREE.RedFormat, THREE.FloatType,
  );
  activityMap.minFilter = THREE.NearestFilter;
  activityMap.magFilter = THREE.NearestFilter;
  activityMap.needsUpdate = true;
  const sharedUniforms = {
    activityMap: { value: activityMap },
    nodeCount: { value: manifest.node_count },
    gain: { value: Number(ui.gain.value) },
    pixelRatio: { value: Math.min(devicePixelRatio, 2) },
  };

  const pointGeometry = new THREE.BufferGeometry();
  pointGeometry.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
  pointGeometry.setAttribute(
    "owner", new THREE.BufferAttribute(Float32Array.from({ length: manifest.node_count }, (_, i) => i), 1),
  );
  pointGeometry.setAttribute("kind", new THREE.BufferAttribute(Float32Array.from(data.kinds), 1));
  const pointMaterial = new THREE.ShaderMaterial({
    uniforms: sharedUniforms,
    vertexShader: pointVertexShader,
    fragmentShader: pointFragmentShader,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const nodes = new THREE.Points(pointGeometry, pointMaterial);
  scene.add(nodes);

  const edgePositions = new Float32Array(manifest.edge_count * 6);
  const edgeColors = new Float32Array(manifest.edge_count * 6);
  for (let edge = 0; edge < manifest.edge_count; edge += 1) {
    const source = data.edges[edge * 2];
    const target = data.edges[edge * 2 + 1];
    edgePositions.set(data.positions.subarray(source * 3, source * 3 + 3), edge * 6);
    edgePositions.set(data.positions.subarray(target * 3, target * 3 + 3), edge * 6 + 3);
    const strength = 0.035 + data.edgeStrength[edge] * 0.10;
    edgeColors.set([0.035 * strength, 0.42 * strength, 0.40 * strength], edge * 6);
    edgeColors.set([0.035 * strength, 0.42 * strength, 0.40 * strength], edge * 6 + 3);
  }
  const edgeGeometry = new THREE.BufferGeometry();
  edgeGeometry.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
  edgeGeometry.setAttribute("color", new THREE.BufferAttribute(edgeColors, 3));
  const edgeLines = new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.14, depthWrite: false,
    blending: THREE.AdditiveBlending,
  }));
  scene.add(edgeLines);

  let morphology = null;
  if (data.skeletonSegments) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(data.skeletonSegments, 3));
    geometry.setAttribute("owner", new THREE.BufferAttribute(Float32Array.from(data.skeletonOwners), 1));
    const material = new THREE.ShaderMaterial({
      uniforms: sharedUniforms,
      vertexShader: lineVertexShader,
      fragmentShader: lineFragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    morphology = new THREE.LineSegments(geometry, material);
    scene.add(morphology);
  } else {
    ui.morphology.textContent = "NO SWC DATA";
    ui.morphology.classList.remove("active");
    ui.morphology.disabled = true;
  }
  addAtmosphere(scene);

  const doomContext = ui["doom-frame"].getContext("2d");
  const [frameHeight, frameWidth, channels] = manifest.frame_shape;
  if (channels !== 3) throw new Error("Viewer expects RGB trace frames");
  ui["doom-frame"].width = frameWidth;
  ui["doom-frame"].height = frameHeight;
  const rgba = new Uint8ClampedArray(frameWidth * frameHeight * 4);
  const image = new ImageData(rgba, frameWidth, frameHeight);

  let frame = 0;
  let playing = true;
  let speed = 1;
  let accumulated = 0;
  let previousTime = performance.now();
  ui.timeline.max = String(manifest.frame_count - 1);
  ui["time-total"].textContent = formatTime(manifest.frame_count / manifest.fps);
  ui["session-title"].textContent = manifest.title;
  ui["node-count"].textContent = formatNumber(manifest.node_count);
  ui["edge-count"].textContent = formatNumber(manifest.edge_count);
  ui["synapse-count"].textContent = formatNumber(manifest.total_synapses);

  function setFrame(nextFrame) {
    frame = ((nextFrame % manifest.frame_count) + manifest.frame_count) % manifest.frame_count;
    const activityOffset = frame * manifest.node_count;
    let peak = 0;
    let peakIndex = 0;
    for (let index = 0; index < manifest.node_count; index += 1) {
      const value = Math.min(Math.abs(data.activity[activityOffset + index]) / manifest.activity_clip, 1);
      activityValues[index] = value;
      if (value > peak) { peak = value; peakIndex = index; }
    }
    activityMap.needsUpdate = true;
    const frameOffset = frame * frameWidth * frameHeight * 3;
    for (let source = 0, target = 0; source < frameWidth * frameHeight * 3; source += 3, target += 4) {
      rgba[target] = data.frames[frameOffset + source];
      rgba[target + 1] = data.frames[frameOffset + source + 1];
      rgba[target + 2] = data.frames[frameOffset + source + 2];
      rgba[target + 3] = 255;
    }
    doomContext.putImageData(image, 0, 0);
    const action = manifest.actions[frame];
    ui["action-label"].textContent = manifest.action_names[action] || `action ${action}`;
    const reward = manifest.rewards[frame];
    ui["reward-label"].textContent = `${reward >= 0 ? "+" : ""}${reward.toFixed(2)}`;
    ui["reward-label"].style.color = reward > 0 ? "#9dffff" : reward < 0 ? "#ff9b78" : "#eaffff";
    ui["peak-type"].textContent = manifest.types[peakIndex] || "untyped neuron";
    ui["peak-meta"].textContent = `${manifest.body_ids[peakIndex]} · ${manifest.regions[peakIndex]}`;
    ui["activity-fill"].style.width = `${Math.round(peak * 100)}%`;
    ui["frame-label"].textContent = `${frame + 1} / ${manifest.frame_count}`;
    ui["time-current"].textContent = formatTime(frame / manifest.fps);
    ui.timeline.value = String(frame);
  }

  ui.play.addEventListener("click", () => {
    playing = !playing;
    ui.play.textContent = playing ? "❚❚" : "▶";
  });
  ui.timeline.addEventListener("input", () => { accumulated = 0; setFrame(Number(ui.timeline.value)); });
  ui.speed.addEventListener("change", () => { speed = Number(ui.speed.value); });
  ui.gain.addEventListener("input", () => { sharedUniforms.gain.value = Number(ui.gain.value); });
  ui.orbit.addEventListener("click", () => {
    controls.autoRotate = !controls.autoRotate;
    ui.orbit.classList.toggle("active", controls.autoRotate);
  });
  ui.morphology.addEventListener("click", () => {
    if (!morphology) return;
    morphology.visible = !morphology.visible;
    ui.morphology.classList.toggle("active", morphology.visible);
  });
  ui.fullscreen.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  });
  addEventListener("keydown", (event) => {
    if (event.code === "Space") { event.preventDefault(); ui.play.click(); }
    if (event.code === "ArrowRight") setFrame(frame + 1);
    if (event.code === "ArrowLeft") setFrame(frame - 1);
  });
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    composer.setSize(innerWidth, innerHeight);
    sharedUniforms.pixelRatio.value = Math.min(devicePixelRatio, 2);
  });

  setFrame(0);
  ui.play.textContent = "❚❚";
  renderer.setAnimationLoop((now) => {
    const elapsed = Math.min((now - previousTime) / 1000, 0.1);
    previousTime = now;
    if (playing) {
      accumulated += elapsed * manifest.fps * speed;
      if (accumulated >= 1) {
        const advance = Math.floor(accumulated);
        accumulated -= advance;
        setFrame(frame + advance);
      }
    }
    controls.update();
    composer.render();
  });
}

async function main() {
  try {
    const data = await loadSession();
    buildViewer(data);
    ui.loading.classList.add("hidden");
  } catch (error) {
    console.error(error);
    ui["loading-message"].textContent = error.message;
    ui["loading-message"].style.color = "#ff9b78";
  }
}

main();
