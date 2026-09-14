import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import "./styles.css";

const ids = [
  "viewport", "session-title", "node-count", "edge-count", "synapse-count", "frame-label",
  "peak-type", "peak-meta", "doom-frame", "sensory-map", "activity-raster", "active-count",
  "action-label", "reward-label", "health-value", "kills-value", "ammo-value", "alive-value",
  "dopamine-total", "dopamine-negative", "dopamine-positive", "reward-breakdown", "action-grid",
  "play", "timeline", "time-current", "time-total", "speed", "gain", "orbit", "morphology",
  "fullscreen", "loading", "loading-message",
  "system-state", "learning-mode",
];
const ui = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const sessionRoot = (new URLSearchParams(window.location.search).get("session") || "/session")
  .replace(/\/$/, "");

const fetchBinary = async (filename, Type) => {
  const response = await fetch(`${sessionRoot}/${filename}`);
  if (!response.ok) throw new Error(`Could not load ${filename} (${response.status})`);
  return new Type(await response.arrayBuffer());
};

const optionalBinary = (files, name, Type) => (
  files[name] ? fetchBinary(files[name], Type) : Promise.resolve(null)
);

const formatNumber = (value) => new Intl.NumberFormat(
  "en-US", { notation: "compact", maximumFractionDigits: 1 },
).format(value);
const formatSigned = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
const formatTime = (seconds) => {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = (seconds % 60).toFixed(2).padStart(5, "0");
  return `${minutes}:${remainder}`;
};
const formatTelemetry = (value) => Number.isFinite(value) ? String(Math.round(value)) : "—";

const lineVertexShader = `
  attribute float owner;
  uniform sampler2D activityMap;
  uniform float nodeCount;
  uniform float gain;
  varying vec3 colorOut;
  varying float alphaOut;
  void main() {
    float activity = clamp(texture2D(activityMap, vec2((owner + 0.5) / nodeCount, 0.5)).r * gain, 0.0, 1.0);
    colorOut = mix(vec3(0.10), vec3(0.62, 1.0, 0.98), smoothstep(0.08, 0.7, activity));
    colorOut = mix(colorOut, vec3(1.0, 0.54, 0.25), smoothstep(0.72, 1.0, activity));
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
    vec3 base = kind > 1.5 ? vec3(0.28, 0.20, 0.14) : vec3(0.07, 0.16, 0.17);
    colorOut = mix(base, vec3(0.68, 1.0, 0.98), smoothstep(0.04, 0.68, activity));
    colorOut = mix(colorOut, vec3(1.0, 0.54, 0.25), smoothstep(0.72, 1.0, activity));
    alphaOut = 0.20 + activity * 0.78;
    vec4 view = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * view;
    gl_PointSize = pixelRatio * (2.0 + 8.0 * activity) * (2.4 / max(1.0, -view.z));
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

async function loadSession() {
  const manifestResponse = await fetch(`${sessionRoot}/manifest.json`);
  if (!manifestResponse.ok) {
    throw new Error("No session found. Run scripts/export_web_viewer.py, then reload.");
  }
  const manifest = await manifestResponse.json();
  const files = manifest.files;
  const loaded = await Promise.all([
    fetchBinary(files.positions, Float32Array),
    fetchBinary(files.activity, Float32Array),
    fetchBinary(files.frames, Uint8Array),
    fetchBinary(files.node_kind, Uint8Array),
    fetchBinary(files.edges, Uint32Array),
    fetchBinary(files.edge_strength, Float32Array),
    optionalBinary(files, "skeleton_segments", Float32Array),
    optionalBinary(files, "skeleton_owners", Uint32Array),
    optionalBinary(files, "telemetry", Float32Array),
    optionalBinary(files, "action_values", Float32Array),
    optionalBinary(files, "reward_components", Float32Array),
  ]);
  const telemetry = loaded[8] || new Float32Array(manifest.frame_count * 3).fill(Number.NaN);
  const actionCount = (manifest.action_names || []).length;
  const rewardCount = (manifest.reward_component_names || []).length;
  return {
    manifest,
    positions: loaded[0],
    activity: loaded[1],
    frames: loaded[2],
    kinds: loaded[3],
    edges: loaded[4],
    edgeStrength: loaded[5],
    skeletonSegments: loaded[6],
    skeletonOwners: loaded[7],
    telemetry,
    actionValues: loaded[9] || new Float32Array(manifest.frame_count * actionCount),
    rewardComponents: loaded[10] || new Float32Array(manifest.frame_count * rewardCount),
  };
}

function buildActionGrid(names) {
  ui["action-grid"].replaceChildren();
  names.forEach((name, index) => {
    const cell = document.createElement("div");
    cell.className = "action-cell";
    cell.dataset.index = String(index);
    const label = document.createElement("span");
    label.textContent = name.replaceAll("_", " ");
    const value = document.createElement("strong");
    value.textContent = "0.00";
    cell.append(label, value);
    ui["action-grid"].append(cell);
  });
}

function buildRewardBreakdown(names) {
  ui["reward-breakdown"].replaceChildren();
  names.forEach((name, index) => {
    const row = document.createElement("div");
    row.dataset.index = String(index);
    const label = document.createElement("span");
    label.textContent = name.replaceAll("_", " ");
    const value = document.createElement("strong");
    value.textContent = "+0.00";
    row.append(label, value);
    ui["reward-breakdown"].append(row);
  });
}

function drawSensory(data, frame, frameWidth, frameHeight) {
  const canvas = ui["sensory-map"];
  const width = 96;
  const height = 54;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  const frameSize = frameWidth * frameHeight * 3;
  const currentOffset = frame * frameSize;
  const previousOffset = Math.max(frame - 1, 0) * frameSize;
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.floor(y * frameHeight / height);
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.floor(x * frameWidth / width);
      const source = (sourceY * frameWidth + sourceX) * 3;
      const leftSource = (sourceY * frameWidth + Math.max(sourceX - 1, 0)) * 3;
      const gray = (
        data.frames[currentOffset + source] * 0.30
        + data.frames[currentOffset + source + 1] * 0.59
        + data.frames[currentOffset + source + 2] * 0.11
      );
      const left = (
        data.frames[currentOffset + leftSource] * 0.30
        + data.frames[currentOffset + leftSource + 1] * 0.59
        + data.frames[currentOffset + leftSource + 2] * 0.11
      );
      const previous = (
        data.frames[previousOffset + source] * 0.30
        + data.frames[previousOffset + source + 1] * 0.59
        + data.frames[previousOffset + source + 2] * 0.11
      );
      const drive = Math.min(Math.abs(gray - left) * 1.7 + Math.abs(gray - previous) + gray * 0.12, 255);
      const visible = drive > 28 ? Math.round(65 + drive * 0.74) : Math.round(drive * 0.16);
      const target = (y * width + x) * 4;
      image.data[target] = Math.round(visible * 0.82);
      image.data[target + 1] = visible;
      image.data[target + 2] = visible;
      image.data[target + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

function drawActivityRaster(data, frame) {
  const canvas = ui["activity-raster"];
  const width = 220;
  const height = 88;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  const { manifest } = data;
  const start = Math.max(0, frame - width + 1);
  let active = 0;
  const currentOffset = frame * manifest.node_count;
  for (let neuron = 0; neuron < manifest.node_count; neuron += 1) {
    if (Math.abs(data.activity[currentOffset + neuron]) / manifest.activity_clip > 0.55) active += 1;
  }
  for (let x = 0; x < width; x += 1) {
    const sourceFrame = start + x - (width - (frame - start + 1));
    if (sourceFrame < 0 || sourceFrame > frame) continue;
    for (let y = 0; y < height; y += 1) {
      const first = Math.floor(y * manifest.node_count / height);
      const last = Math.max(first + 1, Math.floor((y + 1) * manifest.node_count / height));
      let rate = 0;
      for (let neuron = first; neuron < last; neuron += 1) {
        rate = Math.max(
          rate,
          Math.min(Math.abs(data.activity[sourceFrame * manifest.node_count + neuron])
            / manifest.activity_clip, 1),
        );
      }
      const target = ((height - y - 1) * width + x) * 4;
      if (rate > 0.72) {
        image.data[target] = Math.round(180 + 75 * rate);
        image.data[target + 1] = Math.round(95 + 105 * rate);
        image.data[target + 2] = Math.round(42 + 38 * rate);
      } else {
        const level = Math.round(18 + 220 * rate);
        image.data[target] = Math.round(level * 0.72);
        image.data[target + 1] = level;
        image.data[target + 2] = level;
      }
      image.data[target + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  ui["active-count"].textContent = `${formatNumber(active)} RATE EVENTS`;
}

function addAtmosphere(scene) {
  const positions = new Float32Array(450 * 3);
  for (let i = 0; i < positions.length; i += 3) {
    const radius = 2.4 + Math.random() * 3.2;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i] = radius * Math.sin(phi) * Math.cos(theta);
    positions[i + 1] = radius * Math.cos(phi) * 0.62;
    positions[i + 2] = radius * Math.sin(phi) * Math.sin(theta);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0x8ac8c5,
    size: 0.006,
    transparent: true,
    opacity: 0.12,
    depthWrite: false,
  })));
}

function buildViewer(data) {
  const { manifest } = data;
  const actionNames = manifest.action_names || [];
  const rewardNames = manifest.reward_component_names || [];
  buildActionGrid(actionNames);
  buildRewardBreakdown(rewardNames.length ? rewardNames : ["recorded total"]);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020808);
  scene.fog = new THREE.FogExp2(0x020808, 0.12);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  camera.position.set(0.15, 0.2, 3.25);
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.18;
  ui.viewport.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.045;
  controls.minDistance = 1.1;
  controls.maxDistance = 8;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.36;
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 0.78, 0.44, 0.38));

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
    "owner",
    new THREE.BufferAttribute(Float32Array.from({ length: manifest.node_count }, (_, i) => i), 1),
  );
  pointGeometry.setAttribute("kind", new THREE.BufferAttribute(Float32Array.from(data.kinds), 1));
  scene.add(new THREE.Points(pointGeometry, new THREE.ShaderMaterial({
    uniforms: sharedUniforms,
    vertexShader: pointVertexShader,
    fragmentShader: pointFragmentShader,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })));

  const edgePositions = new Float32Array(manifest.edge_count * 6);
  const edgeColors = new Float32Array(manifest.edge_count * 6);
  for (let edge = 0; edge < manifest.edge_count; edge += 1) {
    const source = data.edges[edge * 2];
    const target = data.edges[edge * 2 + 1];
    edgePositions.set(data.positions.subarray(source * 3, source * 3 + 3), edge * 6);
    edgePositions.set(data.positions.subarray(target * 3, target * 3 + 3), edge * 6 + 3);
    const strength = 0.035 + data.edgeStrength[edge] * 0.10;
    edgeColors.set([0.12 * strength, 0.48 * strength, 0.46 * strength], edge * 6);
    edgeColors.set([0.12 * strength, 0.48 * strength, 0.46 * strength], edge * 6 + 3);
  }
  const edgeGeometry = new THREE.BufferGeometry();
  edgeGeometry.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
  edgeGeometry.setAttribute("color", new THREE.BufferAttribute(edgeColors, 3));
  scene.add(new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.15,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })));

  let morphology = null;
  if (data.skeletonSegments) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(data.skeletonSegments, 3));
    geometry.setAttribute("owner", new THREE.BufferAttribute(Float32Array.from(data.skeletonOwners), 1));
    morphology = new THREE.LineSegments(geometry, new THREE.ShaderMaterial({
      uniforms: sharedUniforms,
      vertexShader: lineVertexShader,
      fragmentShader: lineFragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    scene.add(morphology);
  } else {
    ui.morphology.textContent = "NO SWC";
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
  const doomImage = new ImageData(rgba, frameWidth, frameHeight);
  const cumulativeDopamine = new Float32Array(manifest.frame_count);
  let runningReward = 0;
  for (let index = 0; index < manifest.frame_count; index += 1) {
    runningReward += manifest.rewards[index];
    cumulativeDopamine[index] = runningReward;
  }

  let frame = 0;
  let playing = true;
  let speed = 1;
  let accumulated = 0;
  let previousTime = performance.now();
  ui.timeline.max = String(manifest.frame_count - 1);
  ui["time-total"].textContent = formatTime(manifest.frame_count / manifest.fps);
  ui["session-title"].textContent = manifest.title.toUpperCase();
  const learningMode = (manifest.learning_mode || "unknown").replaceAll("_", " ");
  ui["system-state"].textContent = learningMode.toUpperCase();
  ui["learning-mode"].textContent = learningMode === "three factor"
    ? "LOCAL PLASTICITY / NO BACKPROP"
    : learningMode.toUpperCase();
  ui["node-count"].textContent = formatNumber(manifest.node_count);
  ui["edge-count"].textContent = formatNumber(manifest.edge_count);
  ui["synapse-count"].textContent = formatNumber(manifest.total_synapses);

  function setRewardColor(element, value) {
    element.classList.toggle("positive", value > 0);
    element.classList.toggle("negative", value < 0);
  }

  function setFrame(nextFrame) {
    frame = ((nextFrame % manifest.frame_count) + manifest.frame_count) % manifest.frame_count;
    const activityOffset = frame * manifest.node_count;
    let peak = 0;
    let peakIndex = 0;
    for (let index = 0; index < manifest.node_count; index += 1) {
      const value = Math.min(Math.abs(data.activity[activityOffset + index]) / manifest.activity_clip, 1);
      activityValues[index] = value;
      if (value > peak) {
        peak = value;
        peakIndex = index;
      }
    }
    activityMap.needsUpdate = true;

    const frameOffset = frame * frameWidth * frameHeight * 3;
    for (let source = 0, target = 0; source < frameWidth * frameHeight * 3; source += 3, target += 4) {
      rgba[target] = data.frames[frameOffset + source];
      rgba[target + 1] = data.frames[frameOffset + source + 1];
      rgba[target + 2] = data.frames[frameOffset + source + 2];
      rgba[target + 3] = 255;
    }
    doomContext.putImageData(doomImage, 0, 0);
    drawSensory(data, frame, frameWidth, frameHeight);
    drawActivityRaster(data, frame);

    const action = manifest.actions[frame];
    const actionName = actionNames[action] || `action ${action}`;
    ui["action-label"].textContent = `ACTION / ${actionName.replaceAll("_", " ").toUpperCase()}`;
    const reward = manifest.rewards[frame];
    ui["reward-label"].textContent = `DOPAMINE ${formatSigned(reward)}`;
    setRewardColor(ui["reward-label"], reward);
    ui["dopamine-total"].textContent = formatSigned(cumulativeDopamine[frame]);
    setRewardColor(ui["dopamine-total"], cumulativeDopamine[frame]);
    const rewardWidth = Math.min(Math.abs(reward) / 10, 1) * 100;
    ui["dopamine-positive"].style.width = reward > 0 ? `${rewardWidth}%` : "0";
    ui["dopamine-negative"].style.width = reward < 0 ? `${rewardWidth}%` : "0";

    const telemetryOffset = frame * 3;
    ui["health-value"].textContent = formatTelemetry(data.telemetry[telemetryOffset]);
    ui["ammo-value"].textContent = formatTelemetry(data.telemetry[telemetryOffset + 1]);
    ui["kills-value"].textContent = formatTelemetry(data.telemetry[telemetryOffset + 2]);
    ui["alive-value"].textContent = `${(frame / manifest.fps).toFixed(1)} s`;

    const actionOffset = frame * actionNames.length;
    ui["action-grid"].querySelectorAll(".action-cell").forEach((cell, index) => {
      cell.classList.toggle("selected", index === action);
      cell.querySelector("strong").textContent = data.actionValues[actionOffset + index].toFixed(2);
    });

    const rewardOffset = frame * rewardNames.length;
    ui["reward-breakdown"].querySelectorAll("div").forEach((row, index) => {
      const value = rewardNames.length ? data.rewardComponents[rewardOffset + index] : reward;
      const valueElement = row.querySelector("strong");
      valueElement.textContent = formatSigned(value);
      setRewardColor(valueElement, value);
    });

    ui["peak-type"].textContent = manifest.types[peakIndex] || "UNTYPED";
    ui["peak-meta"].textContent = `${manifest.body_ids[peakIndex]} · ${manifest.regions[peakIndex]}`;
    ui["frame-label"].textContent = `FRAME ${frame + 1} / ${manifest.frame_count}`;
    ui["time-current"].textContent = formatTime(frame / manifest.fps);
    ui.timeline.value = String(frame);
  }

  ui.play.addEventListener("click", () => {
    playing = !playing;
    ui.play.textContent = playing ? "PAUSE" : "PLAY";
  });
  ui.timeline.addEventListener("input", () => {
    accumulated = 0;
    setFrame(Number(ui.timeline.value));
  });
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
    if (event.code === "Space") {
      event.preventDefault();
      ui.play.click();
    } else if (event.code === "ArrowRight") {
      setFrame(frame + 1);
    } else if (event.code === "ArrowLeft") {
      setFrame(frame - 1);
    }
  });

  function resizeRenderer() {
    const width = Math.max(ui.viewport.clientWidth, 1);
    const height = Math.max(ui.viewport.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    composer.setSize(width, height);
    sharedUniforms.pixelRatio.value = Math.min(devicePixelRatio, 2);
  }
  new ResizeObserver(resizeRenderer).observe(ui.viewport);
  resizeRenderer();
  setFrame(0);
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
    buildViewer(await loadSession());
    ui.loading.classList.add("hidden");
  } catch (error) {
    console.error(error);
    ui["loading-message"].textContent = error.message;
    ui["loading-message"].classList.add("negative");
  }
}

main();
