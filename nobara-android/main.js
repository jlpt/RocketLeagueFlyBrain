import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const MODEL_URL = './model/scene.gltf';
const CONFIG_URL = './scene.js';
const CONFIG_POLL_MS = 1500;
const DEG = Math.PI / 180;
const TYPE_SPEED_MS = 22;

// Mood presets the director can switch between. Colors lerp when the preset changes.
const PRESETS = {
  showroom: { bg: 0x07090e, fog: 0x07090e, fogDensity: 0.03, key: 0xffffff, keyI: 2.4, rimA: 0x00e5ff, rimB: 0xff4fa3, accent: 0x00e5ff, floor: 0x0d1117, ambient: 0.5, rain: false },
  neon:      { bg: 0x0a0414, fog: 0x0a0414, fogDensity: 0.045, key: 0xffb0e0, keyI: 1.6, rimA: 0x7b5cff, rimB: 0xff2e88, accent: 0xff2e88, floor: 0x120818, ambient: 0.3, rain: false },
  rain:      { bg: 0x0b1118, fog: 0x0b1118, fogDensity: 0.07, key: 0xcfe6ff, keyI: 1.2, rimA: 0x3fa9ff, rimB: 0x9fb8c9, accent: 0x3fa9ff, floor: 0x0a0f14, ambient: 0.3, rain: true },
  sunset:    { bg: 0x2a1020, fog: 0x3a1a25, fogDensity: 0.03, key: 0xffb27a, keyI: 2.8, rimA: 0xff7a59, rimB: 0xffd166, accent: 0xffb35c, floor: 0x1a0f14, ambient: 0.4, rain: false },
};

// Named poses. Values are [x, y, z] Euler degrees applied on top of the rest pose.
// Bone keys: upperarmr, forearmr, upperarml, forearml, spine, spine001, spine002, face, handr, handl.
const POSES = {
  idle: {},
};

// Bone keys we drive. The rig names look like "upper_arm.R_139"; we normalize them to "upperarmr".
const RIG_KEYS = ['upperarmr', 'forearmr', 'upperarml', 'forearml', 'spine', 'spine001', 'spine002', 'face', 'handr', 'handl'];
const boneKey = (name) => name.replace(/_\d+$/, '').replace(/[^a-z0-9]/gi, '').toLowerCase();

// ---------- renderer / scene ----------
const canvas = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(PRESETS.showroom.bg);
scene.fog = new THREE.FogExp2(PRESETS.showroom.fog, PRESETS.showroom.fogDensity);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.35, 4.4);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;
controls.minDistance = 1.6;
controls.maxDistance = 7;
controls.maxPolarAngle = Math.PI * 0.55;
controls.update();

// ---------- lights ----------
const hemi = new THREE.HemisphereLight(0xdfefff, 0x0a0c12, PRESETS.showroom.ambient);
scene.add(hemi);

const key = new THREE.DirectionalLight(PRESETS.showroom.key, PRESETS.showroom.keyI);
key.position.set(2.5, 4, 3);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -2, right: 2, top: 3, bottom: -1, near: 1, far: 12 });
key.shadow.bias = -0.0005;
key.shadow.normalBias = 0.02;
scene.add(key);

const rimA = new THREE.PointLight(PRESETS.showroom.rimA, 12, 10);
rimA.position.set(-2.5, 2.6, -1.5);
scene.add(rimA);
const rimB = new THREE.PointLight(PRESETS.showroom.rimB, 12, 10);
rimB.position.set(2.5, 2.2, -1.5);
scene.add(rimB);

// ---------- showroom set ----------
const floorMat = new THREE.MeshStandardMaterial({ color: PRESETS.showroom.floor, roughness: 0.4, metalness: 0.5 });
const floor = new THREE.Mesh(new THREE.CircleGeometry(14, 80), floorMat);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(28, 56, 0x00e5ff, 0x12303f);
grid.position.y = 0.003;
grid.material.transparent = true;
grid.material.opacity = 0.35;
scene.add(grid);

const wall = new THREE.Mesh(
  new THREE.PlaneGeometry(20, 7),
  new THREE.MeshStandardMaterial({ color: 0x0b0f16, roughness: 0.9 }),
);
wall.position.set(0, 3, -3.2);
wall.receiveShadow = true;
scene.add(wall);

const stripMat = new THREE.MeshBasicMaterial({ color: PRESETS.showroom.accent });
for (const x of [-3.2, -1.6, 1.6, 3.2]) {
  const strip = new THREE.Mesh(new THREE.BoxGeometry(0.04, 4.5, 0.02), stripMat);
  strip.position.set(x, 2.5, -3.19);
  scene.add(strip);
}

const pedestal = new THREE.Mesh(
  new THREE.CylinderGeometry(0.9, 1.0, 0.16, 96),
  new THREE.MeshStandardMaterial({ color: 0xc9d3de, roughness: 0.25, metalness: 0.8 }),
);
pedestal.position.y = 0.08;
pedestal.castShadow = true;
pedestal.receiveShadow = true;
scene.add(pedestal);

const ringMat = new THREE.MeshBasicMaterial({ color: PRESETS.showroom.accent });
const ring = new THREE.Mesh(new THREE.TorusGeometry(0.95, 0.012, 12, 128), ringMat);
ring.rotation.x = Math.PI / 2;
ring.position.y = 0.165;
scene.add(ring);

// Holographic price tag beside the unit
const labelCanvas = document.createElement('canvas');
labelCanvas.width = 512;
labelCanvas.height = 160;
const labelCtx = labelCanvas.getContext('2d');
const labelTex = new THREE.CanvasTexture(labelCanvas);
labelTex.colorSpace = THREE.SRGBColorSpace;
const labelSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTex, transparent: true, depthWrite: false }));
labelSprite.scale.set(1.2, 0.36, 1);
labelSprite.position.set(1.2, 1.5, -0.6);
scene.add(labelSprite);

function drawLabel({ accent = '#00e5ff', lines = [] } = {}) {
  labelCtx.clearRect(0, 0, 512, 160);
  labelCtx.fillStyle = 'rgba(6, 12, 20, 0.82)';
  labelCtx.fillRect(0, 0, 512, 160);
  labelCtx.strokeStyle = accent;
  labelCtx.lineWidth = 4;
  labelCtx.strokeRect(4, 4, 504, 152);
  labelCtx.fillStyle = accent;
  labelCtx.font = '600 24px monospace';
  labelCtx.fillText(lines[0] ?? '', 24, 46);
  labelCtx.fillStyle = '#e6f7ff';
  labelCtx.font = '700 40px system-ui, sans-serif';
  labelCtx.fillText(lines[1] ?? '', 24, 102);
  labelCtx.fillStyle = '#9fb8c9';
  labelCtx.font = '400 24px monospace';
  labelCtx.fillText(lines[2] ?? '', 24, 140);
  labelTex.needsUpdate = true;
}
drawLabel();

// Rain (only visible in the rain preset)
const RAIN_COUNT = 1800;
const rainPos = new Float32Array(RAIN_COUNT * 3);
for (let i = 0; i < RAIN_COUNT; i++) {
  rainPos[i * 3] = (Math.random() - 0.5) * 16;
  rainPos[i * 3 + 1] = Math.random() * 9;
  rainPos[i * 3 + 2] = (Math.random() - 0.5) * 10 - 1;
}
const rainGeo = new THREE.BufferGeometry();
rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPos, 3));
const rain = new THREE.Points(
  rainGeo,
  new THREE.PointsMaterial({ color: 0x9fd4ff, size: 0.03, transparent: true, opacity: 0.6, depthWrite: false }),
);
rain.visible = false;
scene.add(rain);

// ---------- the unit ----------
const model = new THREE.Group(); // yaw (turn) lives here
scene.add(model);

const materials = {}; // name -> material, with the original color stashed in userData.base
const bones = {};     // rig key -> bone, with the rest rotation stashed in userData.base

const loader = new GLTFLoader();
loader.load(
  MODEL_URL,
  (gltf) => {
    const root = gltf.scene;

    root.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true;
        o.receiveShadow = true;
        o.frustumCulled = false;
        for (const m of [].concat(o.material)) {
          if (m && !materials[m.name]) {
            m.userData.base = m.color.clone();
            materials[m.name] = m;
          }
        }
      }
      const k = o.name && boneKey(o.name);
      if (k && RIG_KEYS.includes(k) && !bones[k]) {
        bones[k] = o;
        o.userData.base = o.quaternion.clone();
      }
    });

    // Fit the unit to ~1.7m and stand it on the pedestal, centred on the origin.
    const rawHeight = new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3()).y;
    root.scale.setScalar(1.7 / rawHeight);
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    const center = box.getCenter(new THREE.Vector3());
    root.position.x -= center.x;
    root.position.z -= center.z;
    root.position.y -= box.min.y;

    model.add(root);
    model.position.y = 0.165;

    modelReady = true;
    applyColors(currentColors);
    document.getElementById('loading').classList.add('done');
    window.nobara = { scene, camera, model, materials, bones }; // handy for debugging in the console
  },
  undefined,
  (err) => {
    const el = document.getElementById('loading');
    el.classList.add('error');
    el.textContent = 'Could not load ./model/scene.gltf. Serve this folder over HTTP (see README) instead of opening the file directly.';
    console.error(err);
  },
);

// ---------- mood / pose / colors driven by scene.js ----------
const MOOD = {
  bg: new THREE.Color(), fog: new THREE.Color(), key: new THREE.Color(),
  rimA: new THREE.Color(), rimB: new THREE.Color(), floor: new THREE.Color(), accent: new THREE.Color(),
  fogDensity: 0.03, keyI: 2.4, ambient: 0.5, rain: false,
};

function setMood(name, accentHex) {
  const p = PRESETS[name] ?? PRESETS.showroom;
  MOOD.bg.set(p.bg);
  MOOD.fog.set(p.fog);
  MOOD.key.set(p.key);
  MOOD.rimA.set(p.rimA);
  MOOD.rimB.set(p.rimB);
  MOOD.floor.set(p.floor);
  MOOD.accent.set(accentHex ?? p.accent);
  MOOD.fogDensity = p.fogDensity;
  MOOD.keyI = p.keyI;
  MOOD.ambient = p.ambient;
  MOOD.rain = p.rain;
}
setMood('showroom');

let poseTarget = {};
let waving = false;
let turnTarget = 0;
const poseCur = {};
const tmpEuler = new THREE.Euler();
const tmpQuat = new THREE.Quaternion();

let modelReady = false;
let currentColors = {};

function applyColors(colors = {}) {
  currentColors = colors;
  if (!modelReady) return; // applied again once the model finishes loading
  for (const [name, mat] of Object.entries(materials)) {
    mat.color.copy(mat.userData.base); // reset, then apply overrides
  }
  for (const [name, hex] of Object.entries(colors)) {
    if (materials[name]) materials[name].color.set(hex);
    else console.warn(`scene.js: no material named "${name}"`);
  }
}

// ---------- typewriter dialogue ----------
const speakerEl = document.getElementById('speaker');
const textEl = document.getElementById('text');
const dialogueEl = document.getElementById('dialogue');
let typeTimer = null;
let fullText = '';

function say(speaker, text, color) {
  clearInterval(typeTimer);
  speakerEl.textContent = speaker;
  speakerEl.style.color = color || '';
  fullText = text;
  let i = 0;
  textEl.textContent = '';
  typeTimer = setInterval(() => {
    i++;
    textEl.textContent = fullText.slice(0, i);
    if (i >= fullText.length) clearInterval(typeTimer);
  }, TYPE_SPEED_MS);
}
dialogueEl.addEventListener('click', () => {
  clearInterval(typeTimer);
  textEl.textContent = fullText;
});

// ---------- live config from scene.js ----------
let lastJSON = '';
let lastSayId;
let lastLabelJSON = '';

function applyConfig(cfg) {
  setMood(cfg.preset ?? 'showroom', cfg.accent);
  applyColors(cfg.colors);

  const labelJSON = JSON.stringify(cfg.label ?? {});
  if (labelJSON !== lastLabelJSON) {
    lastLabelJSON = labelJSON;
    drawLabel(cfg.label);
  }

  poseTarget = typeof cfg.pose === 'string' ? (POSES[cfg.pose] ?? {}) : (cfg.pose ?? {});
  waving = !!cfg.wave;
  turnTarget = (cfg.turn ?? 0) * DEG;

  if (cfg.say && cfg.say.id !== lastSayId) {
    lastSayId = cfg.say.id;
    say(cfg.say.speaker ?? '', cfg.say.text ?? '', cfg.say.color);
  }
}

async function pollConfig() {
  try {
    const mod = await import(`${CONFIG_URL}?t=${Date.now()}`);
    const cfg = mod.default;
    const json = JSON.stringify(cfg);
    if (json !== lastJSON) {
      lastJSON = json;
      applyConfig(cfg);
    }
  } catch (err) {
    console.warn('Could not load scene.js (is the folder served over HTTP?)', err);
  }
  setTimeout(pollConfig, CONFIG_POLL_MS);
}
pollConfig();

// ---------- animation loop ----------
const clock = new THREE.Clock();

function applyPose(dt, t) {
  const k = 1 - Math.exp(-dt * 10);
  for (const key of RIG_KEYS) {
    const bone = bones[key];
    if (!bone) continue;

    const want = [...(poseTarget[key] ?? [0, 0, 0])];
    if (key === 'spine') want[0] += Math.sin(t * 1.5) * 1.2; // breathing
    // Wave with the free left arm; the right arm keeps holding the hammer.
    if (waving && key === 'upperarml') want[2] += 55;
    if (waving && key === 'forearml') want[2] += 60 + Math.sin(t * 5) * 15;

    const cur = (poseCur[key] ??= [0, 0, 0]);
    for (let i = 0; i < 3; i++) cur[i] += (want[i] - cur[i]) * k;

    tmpEuler.set(cur[0] * DEG, cur[1] * DEG, cur[2] * DEG);
    tmpQuat.setFromEuler(tmpEuler);
    bone.quaternion.copy(bone.userData.base).multiply(tmpQuat);
  }
}

function frame() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.elapsedTime;
  const ease = 1 - Math.exp(-dt * 2.5);

  scene.background.lerp(MOOD.bg, ease);
  scene.fog.color.lerp(MOOD.fog, ease);
  scene.fog.density += (MOOD.fogDensity - scene.fog.density) * ease;
  key.color.lerp(MOOD.key, ease);
  key.intensity += (MOOD.keyI - key.intensity) * ease;
  hemi.intensity += (MOOD.ambient - hemi.intensity) * ease;
  rimA.color.lerp(MOOD.rimA, ease);
  rimB.color.lerp(MOOD.rimB, ease);
  floorMat.color.lerp(MOOD.floor, ease);
  ringMat.color.lerp(MOOD.accent, ease);
  stripMat.color.lerp(MOOD.accent, ease);
  rain.visible = MOOD.rain;

  if (rain.visible) {
    const pos = rainGeo.attributes.position;
    for (let i = 0; i < RAIN_COUNT; i++) {
      let y = pos.getY(i) - dt * 4;
      if (y < 0) y += 9;
      pos.setY(i, y);
    }
    pos.needsUpdate = true;
  }

  model.rotation.y += (turnTarget - model.rotation.y) * (1 - Math.exp(-dt * 4));
  applyPose(dt, t);

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
frame();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
