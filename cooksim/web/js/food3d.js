/* CookSim REAL-mode food library (2026-09-09).
 *
 * The kitchen's food used to be flat-coloured primitives, and every COOKED item was the
 * same primitive tinted 80% toward dark brown with a "crust" blob on top — so cooked rice,
 * fries and seared steak were indistinguishable brown lumps on the plate. This module
 * builds each food the way it actually looks, per preparation state:
 *
 *   - organic geometry: spheres/boxes/discs displaced by smooth value noise (no perfect
 *     primitives), rounded-edge fries, hand-formed patties, ragged fried-egg whites;
 *   - generated PBR textures: colour + normal (from a painted height map) + roughness,
 *     e.g. seared crust with grill marks, batter crumb, marbled raw beef, fried-egg lace,
 *     tomato flesh with seed gel, lettuce veins, toasted tortilla spots;
 *   - physical materials: clearcoat on wet/glossy surfaces (yolk, tomato skin, seared
 *     juices), sheen on rice, translucency on onion;
 *   - a CC0 photoscanned hamburger bun (Poly Haven `hamburger_buns`) for the bun, cut into
 *     heel and crown for the plated stack.
 *
 * Every builder returns a THREE.Group whose origin is the food's BASE (y = 0), sized to
 * the same footprint the renderer always used (tile = 1 unit; plate radius 0.34), so
 * crates, boards, pans, hands and the plated stack place it unchanged. Looks are
 * deterministic (hash noise), so a rebuilt item never flickers between frames — the
 * assistant's observation frames stay stable.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const ASSET_BASE = "/static/assets/real/";

// ---------------------------------------------------------------- deterministic noise
export const rnd = (i, j = 0, k = 0) => {
  const v = Math.sin(i * 12.9898 + j * 78.233 + k * 37.719) * 43758.5453;
  return v - Math.floor(v);
};
function lat(x, y, z) { return rnd(x * 1.31 + 17, y * 2.17 + 3, z * 3.07 + 5); }
function smooth(t) { return t * t * (3 - 2 * t); }
// 3-D value noise in [0,1]
export function vnoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const fx = smooth(x - xi), fy = smooth(y - yi), fz = smooth(z - zi);
  const L = (a, b, c) => lat(xi + a, yi + b, zi + c);
  const x00 = L(0, 0, 0) + (L(1, 0, 0) - L(0, 0, 0)) * fx, x10 = L(0, 1, 0) + (L(1, 1, 0) - L(0, 1, 0)) * fx;
  const x01 = L(0, 0, 1) + (L(1, 0, 1) - L(0, 0, 1)) * fx, x11 = L(0, 1, 1) + (L(1, 1, 1) - L(0, 1, 1)) * fx;
  const y0 = x00 + (x10 - x00) * fy, y1 = x01 + (x11 - x01) * fy;
  return y0 + (y1 - y0) * fz;
}
// 2-D fractal noise for painting textures
function fbm(x, y, oct = 4) {
  let a = 0, w = 0.5, s = 0;
  for (let o = 0; o < oct; o++) { a += vnoise(x, y, o * 3.1) * w; s += w; x *= 2.03; y *= 1.97; w *= 0.5; }
  return a / s;
}

// ---------------------------------------------------------------- geometry helpers
const GEO = {};
function geo(key, make) { return GEO[key] || (GEO[key] = make()); }

// displace a geometry's vertices by value noise. radial=true pushes along the direction
// from the origin (safe for shared edge vertices such as a box's), else along normals.
export function displace(src, amp, freq, seed = 0, radial = true, yScale = 1) {
  const g = src.clone();
  const p = g.attributes.position, n = g.attributes.normal;
  const v = new THREE.Vector3(), d = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const k = (vnoise(v.x * freq + seed, v.y * freq * yScale + seed * 0.7, v.z * freq + seed * 1.3) - 0.5) * 2 * amp;
    if (radial) d.copy(v).normalize(); else d.fromBufferAttribute(n, i);
    v.addScaledVector(d, k);
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}
function scaled(g, sx, sy, sz) { const c = g.clone(); c.applyMatrix4(new THREE.Matrix4().makeScale(sx, sy, sz)); c.computeVertexNormals(); return c; }
function roundedRect(w, h, r) {
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2); s.lineTo(w / 2 - r, -h / 2); s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r); s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + r, h / 2); s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  s.lineTo(-w / 2, -h / 2 + r); s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  return s;
}
// rounded-edge stick along z, centred
function stick(w, h, len, r) {
  return geo(`stick${w}_${h}_${len}_${r}`, () => {
    const g = new THREE.ExtrudeGeometry(roundedRect(w, h, r), { depth: len - 2 * r, bevelEnabled: true, bevelSize: r, bevelThickness: r, bevelSegments: 3, curveSegments: 5 });
    g.translate(0, 0, -(len - 2 * r) / 2);
    return g;
  });
}

// ---------------------------------------------------------------- texture helpers
const TEX = {};
function canvas(size) { const c = document.createElement("canvas"); c.width = c.height = size; return c; }
const hex = (h) => "#" + h.toString(16).padStart(6, "0");
function tex(key, size, draw, srgb = true) {
  if (TEX[key]) return TEX[key];
  const c = canvas(size); draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4;
  TEX[key] = t; return t;
}
// normal map from a painted height canvas (draw grayscale: white = high)
function normalTex(key, size, draw, strength = 2) {
  if (TEX[key]) return TEX[key];
  const hc = canvas(size); draw(hc.getContext("2d"), size);
  const src = hc.getContext("2d").getImageData(0, 0, size, size).data;
  const out = canvas(size), octx = out.getContext("2d"), img = octx.createImageData(size, size), d = img.data;
  const H = (x, y) => src[(((y + size) % size) * size + ((x + size) % size)) * 4] / 255;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const dx = (H(x + 1, y) - H(x - 1, y)) * strength, dy = (H(x, y + 1) - H(x, y - 1)) * strength;
    const l = Math.hypot(dx, dy, 1);
    const i = (y * size + x) * 4;
    d[i] = (-dx / l * 0.5 + 0.5) * 255; d[i + 1] = (dy / l * 0.5 + 0.5) * 255; d[i + 2] = (1 / l * 0.5 + 0.5) * 255; d[i + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(out); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4;
  TEX[key] = t; return t;
}
// paint helpers
function fill(ctx, size, color) { ctx.fillStyle = color; ctx.fillRect(0, 0, size, size); }
function blobs(ctx, size, n, seed, rmin, rmax, colorFn) {
  for (let i = 0; i < n; i++) {
    const x = rnd(i, seed) * size, y = rnd(i, seed + 1) * size, r = rmin + rnd(i, seed + 2) * (rmax - rmin);
    ctx.fillStyle = colorFn(i, rnd(i, seed + 3)); ctx.beginPath(); ctx.arc(x, y, r, 0, 6.283); ctx.fill();
  }
}
function noiseShade(ctx, size, scale, amp, seed, base) {
  // multiply-ish mottling from fbm, painted as translucent dark/light blocks
  const step = 4;
  for (let y = 0; y < size; y += step) for (let x = 0; x < size; x += step) {
    const v = fbm(x / size * scale + seed, y / size * scale + seed * 2) - 0.5;
    ctx.fillStyle = v > 0 ? `rgba(255,255,255,${Math.min(1, v * amp)})` : `rgba(${base},${Math.min(1, -v * amp)})`;
    ctx.fillRect(x, y, step, step);
  }
}
function heightNoise(ctx, size, scale, seed, oct = 4, gain = 1, bias = 0) {
  const img = ctx.createImageData(size, size), d = img.data;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const v = Math.max(0, Math.min(1, (fbm(x / size * scale + seed, y / size * scale + seed, oct) - 0.5) * gain + 0.5 + bias));
    const i = (y * size + x) * 4; d[i] = d[i + 1] = d[i + 2] = v * 255; d[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------- materials
function phys(o) { return new THREE.MeshPhysicalMaterial(Object.assign({ roughness: 0.6, metalness: 0 }, o)); }

const MAT = {};
function material(key, make) { return MAT[key] || (MAT[key] = make()); }

// -- beef: raw ground / seared crust / raw marbled / cut face
const mRawGround = () => material("rawground", () => phys({
  map: tex("rawground", 256, (c, s) => {
    fill(c, s, "#b5483a");
    blobs(c, s, 900, 11, 2, 6, (i, r) => i % 6 ? `rgba(${200 + r * 40 | 0},${90 + r * 60 | 0},${80 + r * 40 | 0},.75)` : "rgba(240,225,215,.8)");
    noiseShade(c, s, 6, 0.5, 3, "80,20,15");
  }),
  normalMap: normalTex("rawground_n", 256, (c, s) => { fill(c, s, "#808080"); blobs(c, s, 900, 11, 2, 6, (i, r) => `rgba(${120 + r * 135 | 0},${120 + r * 135 | 0},${120 + r * 135 | 0},.9)`); }, 1.4),
  roughness: 0.62 }));
const mSearCrust = () => material("searcrust", () => phys({
  map: tex("searcrust", 256, (c, s) => {
    fill(c, s, "#47220e");
    blobs(c, s, 700, 21, 2, 7, (i, r) => i % 4 ? `rgba(${90 + r * 50 | 0},${45 + r * 25 | 0},${18 + r * 12 | 0},.7)` : "rgba(28,12,4,.75)");
    noiseShade(c, s, 5, 0.45, 7, "30,10,3");
    c.strokeStyle = "rgba(24,10,3,.9)"; c.lineWidth = 13;
    for (let k = -1; k < 6; k++) { c.beginPath(); c.moveTo(-20, k * 62); c.lineTo(s + 20, k * 62 + 70); c.stroke(); }
    c.strokeStyle = "rgba(150,90,40,.35)"; c.lineWidth = 3;
    for (let k = -1; k < 6; k++) { c.beginPath(); c.moveTo(-20, k * 62 - 10); c.lineTo(s + 20, k * 62 + 60); c.stroke(); }
  }),
  normalMap: normalTex("searcrust_n", 256, (c, s) => {
    heightNoise(c, s, 9, 4, 4, 1.4);
    c.strokeStyle = "rgba(0,0,0,.9)"; c.lineWidth = 13;
    for (let k = -1; k < 6; k++) { c.beginPath(); c.moveTo(-20, k * 62); c.lineTo(s + 20, k * 62 + 70); c.stroke(); }
  }, 2.2),
  roughness: 0.45, clearcoat: 0.12, clearcoatRoughness: 0.6 }));
const mRawSteakTop = () => material("rawsteak", () => phys({
  map: tex("rawsteak", 256, (c, s) => {
    fill(c, s, "#b2302a");
    noiseShade(c, s, 4, 0.5, 9, "90,20,25");
    c.strokeStyle = "rgba(245,235,225,.85)"; c.lineCap = "round";
    for (let k = 0; k < 14; k++) {                       // fat marbling veins
      c.lineWidth = 2 + rnd(k, 5) * 4; c.beginPath();
      let x = rnd(k, 1) * s, y = rnd(k, 2) * s; c.moveTo(x, y);
      for (let t = 0; t < 6; t++) { x += (rnd(k, t + 3) - 0.5) * 60; y += (rnd(k, t + 9) - 0.5) * 60; c.lineTo(x, y); }
      c.stroke();
    }
    blobs(c, s, 200, 33, 1, 3, () => "rgba(250,240,235,.6)");
  }),
  normalMap: normalTex("rawsteak_n", 256, (c, s) => heightNoise(c, s, 7, 6, 3, 0.8), 1.2),
  roughness: 0.42, clearcoat: 0.35, clearcoatRoughness: 0.45 }));
const mCutFace = () => material("cutface", () => phys({
  map: tex("cutface", 128, (c, s) => {
    const g = c.createLinearGradient(0, 0, 0, s);          // v: bottom crust -> rosy centre -> top crust
    g.addColorStop(0, "#5a2c14"); g.addColorStop(0.18, "#8c4a36"); g.addColorStop(0.42, "#c9766a"); g.addColorStop(0.58, "#c9766a"); g.addColorStop(0.82, "#8c4a36"); g.addColorStop(1, "#5a2c14");
    c.fillStyle = g; c.fillRect(0, 0, s, s);
    c.strokeStyle = "rgba(255,225,215,.35)";
    for (let k = 0; k < 40; k++) { c.lineWidth = 1; c.beginPath(); const x = rnd(k, 8) * s; c.moveTo(x, s * 0.2); c.lineTo(x + (rnd(k, 9) - 0.5) * 8, s * 0.8); c.stroke(); }
  }), roughness: 0.5 }));
const mFlesh = () => material("flesh", () => phys({ color: 0xb8362c, roughness: 0.45, clearcoat: 0.3, clearcoatRoughness: 0.5,
  map: tex("flesh", 128, (c, s) => { fill(c, s, "#b8362c"); noiseShade(c, s, 5, 0.5, 12, "90,15,20"); blobs(c, s, 120, 41, 1, 3, () => "rgba(245,235,225,.7)"); }) }));

// -- fries
const mFries = (cooked) => material(cooked ? "fries" : "friesraw", () => phys({
  map: tex(cooked ? "fries" : "friesraw", 256, (c, s) => {
    fill(c, s, cooked ? "#e6b24e" : "#f3e6b6");
    noiseShade(c, s, 5, 0.55, 15, cooked ? "150,80,20" : "190,170,110");
    if (cooked) blobs(c, s, 260, 51, 2, 9, (i, r) => i % 3 ? `rgba(${170 + r * 40 | 0},${95 + r * 40 | 0},25,.45)` : "rgba(250,225,150,.5)");
    else blobs(c, s, 120, 51, 2, 7, () => "rgba(255,250,225,.4)");
  }),
  normalMap: normalTex(cooked ? "fries_n" : "friesraw_n", 256, (c, s) => heightNoise(c, s, 10, 8, 4, 1.1), cooked ? 1.6 : 0.8),
  roughness: cooked ? 0.42 : 0.62, clearcoat: cooked ? 0.2 : 0, clearcoatRoughness: 0.6 }));

// -- fish: raw flesh / batter
const mFishRaw = () => material("fishraw", () => phys({
  map: tex("fishraw", 256, (c, s) => {
    fill(c, s, "#f0c0b2"); noiseShade(c, s, 4, 0.35, 21, "210,120,110");
    c.strokeStyle = "rgba(255,248,244,.85)"; c.lineWidth = 4;
    for (let k = 0; k < 9; k++) { c.beginPath(); c.moveTo(k * 32 - 10, 0); c.quadraticCurveTo(k * 32 + 40, s / 2, k * 32 - 10, s); c.stroke(); }
  }),
  normalMap: normalTex("fishraw_n", 256, (c, s) => { fill(c, s, "#808080"); c.strokeStyle = "#b0b0b0"; c.lineWidth = 4; for (let k = 0; k < 9; k++) { c.beginPath(); c.moveTo(k * 32 - 10, 0); c.quadraticCurveTo(k * 32 + 40, s / 2, k * 32 - 10, s); c.stroke(); } }, 1.2),
  roughness: 0.28, clearcoat: 0.6, clearcoatRoughness: 0.3 }));
const mBatter = () => material("batter", () => phys({
  map: tex("batter", 256, (c, s) => {
    fill(c, s, "#d79a44"); noiseShade(c, s, 6, 0.6, 31, "140,70,20");
    blobs(c, s, 500, 61, 2, 8, (i, r) => i % 3 ? `rgba(${160 + r * 60 | 0},${90 + r * 40 | 0},30,.55)` : "rgba(250,225,160,.55)");
  }),
  normalMap: normalTex("batter_n", 256, (c, s) => { heightNoise(c, s, 12, 14, 4, 1.4); blobs(c, s, 400, 62, 3, 9, (i, r) => `rgba(255,255,255,${0.3 + r * 0.6})`); }, 2.6),
  roughness: 0.48 }));

// -- egg
const mEggWhite = () => material("eggwhite", () => phys({
  map: tex("eggwhite", 256, (c, s) => {
    const g = c.createRadialGradient(s / 2, s / 2, s * 0.05, s / 2, s / 2, s * 0.5);
    g.addColorStop(0, "#fbfaf6"); g.addColorStop(0.6, "#f7f4ec"); g.addColorStop(0.82, "#ecdfbf"); g.addColorStop(0.93, "#c9903e"); g.addColorStop(1, "#8a5722");
    c.fillStyle = g; c.fillRect(0, 0, s, s);
    blobs(c, s, 90, 71, 3, 10, () => "rgba(255,255,255,.35)");
  }),
  normalMap: normalTex("eggwhite_n", 256, (c, s) => heightNoise(c, s, 6, 20, 3, 0.7), 1.0),
  roughness: 0.42, clearcoat: 0.15, clearcoatRoughness: 0.5 }));
const mYolk = () => material("yolk", () => phys({ color: 0xf6a71a, roughness: 0.18, clearcoat: 1, clearcoatRoughness: 0.08 }));

// -- rice
const mRiceMound = () => material("ricemound", () => phys({ color: 0xf2ede2, roughness: 0.62, sheen: 0.5, sheenColor: new THREE.Color(0xffffff), sheenRoughness: 0.6,
  normalMap: normalTex("rice_n", 256, (c, s) => { fill(c, s, "#808080"); blobs(c, s, 900, 81, 2, 4, (i, r) => `rgba(${150 + r * 100 | 0},${150 + r * 100 | 0},${150 + r * 100 | 0},.9)`); }, 1.6) }));
const mRiceGrain = () => material("ricegrain", () => phys({ color: 0xfaf7ef, roughness: 0.32, clearcoat: 0.3, clearcoatRoughness: 0.4, sheen: 0.3, sheenColor: new THREE.Color(0xffffff) }));
const mRiceRaw = () => material("riceraw", () => phys({ color: 0xf4efe4, roughness: 0.5 }));

// -- tomato
const mTomatoSkin = () => material("tomatoskin", () => phys({ color: 0xd63a2c, roughness: 0.22, clearcoat: 0.8, clearcoatRoughness: 0.15,
  map: tex("tomatoskin", 128, (c, s) => { fill(c, s, "#d63a2c"); noiseShade(c, s, 3, 0.25, 41, "150,25,20"); }) }));
const mTomatoFlesh = () => material("tomatoflesh", () => phys({ roughness: 0.3, clearcoat: 0.5, clearcoatRoughness: 0.3,
  map: tex("tomatoflesh", 128, (c, s) => {
    fill(c, s, "#e3564a");
    c.fillStyle = "#f2a89a"; c.beginPath(); c.arc(s / 2, s / 2, s * 0.33, 0, 6.283); c.fill();          // seed gel
    c.fillStyle = "#c73a2d"; c.beginPath(); c.arc(s / 2, s / 2, s * 0.12, 0, 6.283); c.fill();          // core
    blobs(c, s, 26, 43, 3, 5, () => "#f3e08a");                                                        // seeds
  }) }));
const mSauce = () => material("sauce", () => phys({ color: 0xc23c26, roughness: 0.22, clearcoat: 0.6, clearcoatRoughness: 0.25 }));

// -- lettuce
const mLeaf = () => material("leaf", () => phys({ side: THREE.DoubleSide, roughness: 0.48, clearcoat: 0.2, clearcoatRoughness: 0.5,
  map: tex("leaf", 256, (c, s) => {
    fill(c, s, "#8ccf5b"); noiseShade(c, s, 5, 0.4, 51, "60,120,40");
    c.strokeStyle = "rgba(236,246,210,.9)"; c.lineWidth = 5; c.beginPath(); c.moveTo(s / 2, s); c.lineTo(s / 2, 0); c.stroke();
    c.lineWidth = 2;
    for (let k = 0; k < 10; k++) { const y = k / 10 * s; c.beginPath(); c.moveTo(s / 2, y + 20); c.quadraticCurveTo(s * 0.75, y, s, y - 30); c.stroke(); c.beginPath(); c.moveTo(s / 2, y + 20); c.quadraticCurveTo(s * 0.25, y, 0, y - 30); c.stroke(); }
  }),
  normalMap: normalTex("leaf_n", 256, (c, s) => heightNoise(c, s, 8, 24, 4, 1.2), 1.4) }));

// -- onion
const mOnionSkin = () => material("onionskin", () => phys({ roughness: 0.4, clearcoat: 0.3, clearcoatRoughness: 0.4,
  map: tex("onionskin", 256, (c, s) => {
    fill(c, s, "#d9a24f"); noiseShade(c, s, 3, 0.4, 61, "120,70,20");
    c.strokeStyle = "rgba(110,60,15,.35)"; for (let k = 0; k < 40; k++) { c.lineWidth = 1 + rnd(k, 3) * 2; c.beginPath(); c.moveTo(k / 40 * s, 0); c.lineTo(k / 40 * s + (rnd(k, 4) - 0.5) * 10, s); c.stroke(); }
  }) }));
const mOnionFlesh = (cooked) => material(cooked ? "onioncooked" : "onionflesh", () => phys({ roughness: cooked ? 0.25 : 0.35, transparent: true, opacity: cooked ? 0.92 : 0.88, clearcoat: cooked ? 0.6 : 0.3,
  map: tex(cooked ? "onioncooked_t" : "onionflesh_t", 128, (c, s) => {
    fill(c, s, cooked ? "#e6c07a" : "#f6f1dc");
    c.strokeStyle = cooked ? "rgba(190,130,50,.5)" : "rgba(200,190,150,.55)"; c.lineWidth = 3;
    for (let r = 8; r < s; r += 12) { c.beginPath(); c.arc(s / 2, s / 2, r, 0, 6.283); c.stroke(); }
  }) }));

// -- mushroom
const mMushCap = () => material("mushcap", () => phys({ roughness: 0.6,
  map: tex("mushcap", 128, (c, s) => { fill(c, s, "#cdb59a"); noiseShade(c, s, 4, 0.35, 71, "110,80,55"); blobs(c, s, 60, 72, 2, 6, () => "rgba(120,85,55,.4)"); }) }));
const mMushCooked = () => material("mushcooked", () => phys({ color: 0x7a5636, roughness: 0.25, clearcoat: 0.5, clearcoatRoughness: 0.3 }));

// -- tortilla / bread
const mTortilla = () => material("tortilla", () => phys({ side: THREE.DoubleSide, roughness: 0.75,
  map: tex("tortilla", 256, (c, s) => {
    fill(c, s, "#f0e2c0"); noiseShade(c, s, 4, 0.25, 81, "170,140,90");
    blobs(c, s, 40, 82, 4, 16, (i, r) => `rgba(${150 + r * 40 | 0},${95 + r * 30 | 0},40,${0.35 + r * 0.4})`);
    blobs(c, s, 120, 83, 1, 4, () => "rgba(120,70,30,.5)");
  }),
  normalMap: normalTex("tortilla_n", 256, (c, s) => { heightNoise(c, s, 8, 26, 4, 0.9); blobs(c, s, 40, 82, 4, 16, () => "rgba(255,255,255,.6)"); }, 1.5) }));
const mCrumb = () => material("crumb", () => phys({ roughness: 0.9,
  map: tex("crumb", 128, (c, s) => { fill(c, s, "#f4ead3"); blobs(c, s, 400, 91, 1, 4, (i, r) => `rgba(${200 + r * 40 | 0},${180 + r * 40 | 0},${140 + r * 40 | 0},.8)`); }),
  normalMap: normalTex("crumb_n", 128, (c, s) => { fill(c, s, "#808080"); blobs(c, s, 400, 91, 1, 4, (i, r) => `rgba(${60 + r * 160 | 0},${60 + r * 160 | 0},${60 + r * 160 | 0},.9)`); }, 1.8) }));

// -- potato / cheese / misc
const mPotatoSkin = () => material("potatoskin", () => phys({ roughness: 0.85,
  map: tex("potatoskin", 256, (c, s) => { fill(c, s, "#c9a267"); noiseShade(c, s, 4, 0.4, 101, "120,85,40"); blobs(c, s, 40, 102, 2, 5, () => "rgba(90,60,30,.6)"); blobs(c, s, 200, 103, 1, 2, () => "rgba(80,55,30,.4)"); }),
  normalMap: normalTex("potatoskin_n", 256, (c, s) => { heightNoise(c, s, 7, 30, 4, 1.0); blobs(c, s, 40, 102, 2, 5, () => "#222"); }, 1.6) }));
const mCheese = () => material("cheese2", () => phys({ roughness: 0.38, clearcoat: 0.25, clearcoatRoughness: 0.45, sheen: 0.2, sheenColor: new THREE.Color(0xfff2a0),
  map: tex("cheese2", 128, (c, s) => {
    fill(c, s, "#f4c94e"); noiseShade(c, s, 3, 0.15, 111, "200,150,40");
    const holes = [[24, 30, 9], [70, 22, 6], [102, 58, 10], [44, 78, 7], [88, 100, 8], [18, 108, 5], [62, 52, 4]];
    for (const [x, y, r] of holes) { c.fillStyle = "#d9a92c"; c.beginPath(); c.arc(x, y, r, 0, 6.283); c.fill(); c.fillStyle = "#c2941f"; c.beginPath(); c.arc(x + r * 0.2, y + r * 0.25, r * 0.72, 0, 6.283); c.fill(); }
  }) }));
const mShell = () => material("shell", () => phys({ roughness: 0.55, map: tex("shell", 128, (c, s) => { fill(c, s, "#f1e6d2"); blobs(c, s, 160, 121, 1, 3, () => "rgba(170,140,110,.5)"); }) }));

// ---------------------------------------------------------------- builders
function G() { return new THREE.Group(); }
function put(grp, mesh, x = 0, y = 0, z = 0) { mesh.position.set(x, y, z); mesh.castShadow = true; mesh.receiveShadow = true; grp.add(mesh); return mesh; }
function box(w, h, d, sx = 4, sy = 2, sz = 3) { return geo(`bx${w}_${h}_${d}_${sx}${sy}${sz}`, () => new THREE.BoxGeometry(w, h, d, sx, sy, sz)); }
function sph(r, w = 32, h = 20) { return geo(`sp${r}_${w}_${h}`, () => new THREE.SphereGeometry(r, w, h)); }
function disp(key, g, amp, freq, seed, radial = true) { return geo(key, () => displace(g, amp, freq, seed, radial)); }

// per-face material arrays for BoxGeometry: [+x, -x, +y(top), -y, +z, -z]
const faces = (top, bottom, side, end) => [end || side, end || side, top, bottom, side, side];

function riceCooked() {
  const g = G();
  const mound = disp("ricemound", scaled(sph(0.2, 36, 20), 1, 0.52, 1), 0.012, 16, 3);
  put(g, new THREE.Mesh(mound, mRiceMound()), 0, 0.06);
  const grain = geo("grain", () => new THREE.SphereGeometry(1, 7, 5));
  const N = 260, inst = new THREE.InstancedMesh(grain, mRiceGrain(), N);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), p = new THREE.Vector3(), s = new THREE.Vector3();
  for (let i = 0; i < N; i++) {
    const u = rnd(i, 1), v = rnd(i, 2);
    const th = Math.acos(1 - u * 0.97), ph = v * 6.283;               // upper hemisphere of the mound
    const r = i < N - 24 ? 1 : 1.15 + rnd(i, 7) * 0.25;               // a few grains spilled at the base
    p.set(Math.sin(th) * Math.cos(ph) * 0.2 * r, i < N - 24 ? 0.06 + Math.cos(th) * 0.104 + 0.004 : 0.008, Math.sin(th) * Math.sin(ph) * 0.2 * r);
    e.set(rnd(i, 3) * 3.14, rnd(i, 4) * 6.28, rnd(i, 5) * 3.14); q.setFromEuler(e);
    s.set(0.013, 0.0085, 0.03); m.compose(p, q, s); inst.setMatrixAt(i, m);
  }
  inst.instanceMatrix.needsUpdate = true; inst.castShadow = true; g.add(inst);
  return g;
}
function riceRaw() {                                                 // a measured cup of dry grains
  const g = G();
  put(g, new THREE.Mesh(geo("cup", () => new THREE.CylinderGeometry(0.15, 0.12, 0.13, 24, 1, true)), phys({ color: 0xd8d0c0, roughness: 0.7, side: THREE.DoubleSide })), 0, 0.065);
  put(g, new THREE.Mesh(disp("ricerawtop", sph(0.14, 20, 12), 0.006, 30, 5), mRiceRaw()), 0, 0.1).scale.set(1, 0.3, 1);
  const grain = geo("grain", () => new THREE.SphereGeometry(1, 7, 5));
  const N = 90, inst = new THREE.InstancedMesh(grain, mRiceRaw(), N);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), p = new THREE.Vector3(), s = new THREE.Vector3(0.011, 0.007, 0.024);
  for (let i = 0; i < N; i++) { const a = rnd(i, 1) * 6.283, r = Math.sqrt(rnd(i, 2)) * 0.13; p.set(Math.cos(a) * r, 0.14 + rnd(i, 6) * 0.01, Math.sin(a) * r); e.set(0, rnd(i, 4) * 6.28, 0); q.setFromEuler(e); m.compose(p, q, s); inst.setMatrixAt(i, m); }
  inst.instanceMatrix.needsUpdate = true; g.add(inst);
  return g;
}

function fries(cooked) {
  const g = G(); const M = mFries(cooked);
  for (let i = 0; i < 12; i++) {
    const L = 0.2 + rnd(i, 1) * 0.1, layer = Math.floor(i / 4);
    const f = put(g, new THREE.Mesh(stick(0.05, 0.05, L, 0.011), M), (rnd(i, 2) - 0.5) * 0.2, 0.026 + layer * 0.048, (rnd(i, 3) - 0.5) * 0.14);
    f.rotation.set((rnd(i, 4) - 0.5) * 0.3, (layer % 2 ? 1.25 : 0.1) + (rnd(i, 5) - 0.5) * 1.1, (rnd(i, 6) - 0.5) * 0.25);
  }
  return g;
}

function steakRaw() {
  const g = G();
  const slab = put(g, new THREE.Mesh(disp("slab", box(0.34, 0.075, 0.24, 8, 2, 6), 0.006, 14, 8), faces(mRawSteakTop(), mRawSteakTop(), mFlesh())), 0, 0.04);
  slab.rotation.y = 0.35;
  const fat = put(g, new THREE.Mesh(box(0.34, 0.06, 0.03, 2, 1, 1), phys({ color: 0xf2e6d6, roughness: 0.5 })), -0.04, 0.045, 0.115); fat.rotation.y = 0.35;
  return g;
}
function steakStrips(cooked) {
  const g = G();
  const M = cooked ? faces(mSearCrust(), mSearCrust(), mCutFace()) : faces(mRawSteakTop(), mRawSteakTop(), mFlesh());
  for (let i = 0; i < 6; i++) {
    const L = 0.2 + rnd(i, 1) * 0.08, top = i >= 4;
    const st = put(g, new THREE.Mesh(disp("strip" + i, box(L, 0.045, 0.06, 6, 2, 3), 0.005, 18, 20 + i), M),
                   (rnd(i, 2) - 0.5) * 0.12, 0.024 + (top ? 0.046 : 0), top ? (i - 4.5) * 0.09 : (i - 1.5) * 0.072);
    st.rotation.set(0, (rnd(i, 3) - 0.5) * 0.6 + (top ? 0.6 : 0), (rnd(i, 4) - 0.5) * 0.12);
    if (!cooked && i % 2) { const fat = new THREE.Mesh(box(L * 0.7, 0.006, 0.014, 1, 1, 1), phys({ color: 0xf2e6d6, roughness: 0.5 })); fat.position.set(0, 0.024, 0.012); st.add(fat); }
  }
  return g;
}

function fishRaw() {
  const g = G();
  const f = put(g, new THREE.Mesh(disp("fillet", scaled(sph(0.15, 36, 22), 1.7, 0.42, 0.85), 0.01, 12, 30), mFishRaw()), 0, 0.06);
  f.rotation.y = 0.2;
  const tail = put(g, new THREE.Mesh(geo("tail", () => new THREE.CylinderGeometry(0, 0.1, 0.02, 3)), mFishRaw()), -0.27, 0.05, 0.05); tail.rotation.z = Math.PI / 2; tail.rotation.y = 0.2;
  return g;
}
function fishSlices() {
  const g = G();
  for (let i = 0; i < 5; i++) {
    const sl = put(g, new THREE.Mesh(disp("fslice", box(0.16, 0.026, 0.09, 4, 1, 3), 0.003, 20, 40), mFishRaw()), (i - 2) * 0.05, 0.014 + i * 0.011, (i % 2) * 0.05 - 0.025);
    sl.rotation.y = 0.3 + (rnd(i, 1) - 0.5) * 0.2;
  }
  return g;
}
function fishFried() {
  const g = G();
  const f = put(g, new THREE.Mesh(disp("battered", scaled(sph(0.16, 40, 24), 1.45, 0.72, 0.85), 0.022, 18, 50), mBatter()), 0, 0.1);
  f.rotation.y = 0.15;
  for (let i = 0; i < 14; i++) put(g, new THREE.Mesh(sph(0.03 + rnd(i, 9) * 0.02, 10, 8), mBatter()), (rnd(i, 1) - 0.5) * 0.4, 0.16 + rnd(i, 2) * 0.04, (rnd(i, 3) - 0.5) * 0.22).scale.set(1, 0.7, 1);
  return g;
}

function eggFried() {
  const g = G();
  put(g, new THREE.Mesh(disp("eggwhite", scaled(sph(0.17, 40, 20), 1.2, 0.15, 1.0), 0.018, 9, 60), mEggWhite()), 0, 0.02);
  put(g, new THREE.Mesh(disp("eggwhite2", scaled(sph(0.12, 32, 16), 1.1, 0.18, 1.0), 0.012, 11, 61), mEggWhite()), 0.07, 0.022, 0.05);
  put(g, new THREE.Mesh(sph(0.072, 28, 18), mYolk()), 0.0, 0.045, -0.01).scale.set(1, 0.72, 1);
  return g;
}
function eggRaw() {
  const g = G();
  put(g, new THREE.Mesh(disp("eggshell", sph(0.13, 28, 18), 0.002, 20, 70), mShell()), 0, 0.15).scale.set(0.85, 1.15, 0.85);
  return g;
}

function tomatoRaw() {
  const g = G();
  put(g, new THREE.Mesh(disp("tomato", sph(0.15, 36, 24), 0.007, 5, 80), mTomatoSkin()), 0, 0.15).scale.set(1, 0.92, 1);
  const green = phys({ color: 0x4f8f3a, roughness: 0.6 });
  put(g, new THREE.Mesh(geo("stem", () => new THREE.CylinderGeometry(0.008, 0.012, 0.05, 6)), green), 0, 0.3);
  for (let k = 0; k < 5; k++) { const lf = put(g, new THREE.Mesh(box(0.06, 0.006, 0.018, 1, 1, 1), green), Math.cos(k * 1.257) * 0.03, 0.283, Math.sin(k * 1.257) * 0.03); lf.rotation.y = -k * 1.257; lf.rotation.z = -0.35; }
  return g;
}
function tomatoDice() {
  const g = G();
  for (let i = 0; i < 8; i++) {
    const s = 0.07 + rnd(i, 1) * 0.03;
    const d = put(g, new THREE.Mesh(box(s, s * 0.8, s, 2, 2, 2), i % 2 ? faces(mTomatoFlesh(), mTomatoSkin(), mTomatoFlesh(), mTomatoSkin()) : faces(mTomatoSkin(), mTomatoFlesh(), mTomatoFlesh())),
                  (rnd(i, 2) - 0.5) * 0.24, s * 0.4 + (i > 5 ? 0.06 : 0), (rnd(i, 3) - 0.5) * 0.24);
    d.rotation.set((rnd(i, 4) - 0.5) * 0.4, rnd(i, 5) * 6.28, (rnd(i, 6) - 0.5) * 0.4);
  }
  return g;
}
function tomatoCooked() {
  const g = G();
  put(g, new THREE.Mesh(disp("sauce", scaled(sph(0.19, 36, 20), 1, 0.35, 1), 0.01, 10, 90), mSauce()), 0, 0.03);
  for (let i = 0; i < 6; i++) put(g, new THREE.Mesh(box(0.06, 0.03, 0.06, 2, 1, 2), mTomatoFlesh()), (rnd(i, 1) - 0.5) * 0.26, 0.07, (rnd(i, 2) - 0.5) * 0.26).rotation.y = rnd(i, 3) * 3;
  return g;
}

function ribbonGeo() {
  return geo("ribbon", () => {
    const pg = new THREE.PlaneGeometry(0.2, 0.05, 12, 2);
    const p = pg.attributes.position;
    for (let i = 0; i < p.count; i++) { const x = p.getX(i), y = p.getY(i); p.setZ(i, Math.sin(x * 28) * 0.014 + Math.sin(y * 60) * 0.004); }
    pg.computeVertexNormals(); return pg;
  });
}
function lettuceShreds() {
  const g = G(); const M = mLeaf();
  for (let i = 0; i < 16; i++) {
    const rb = put(g, new THREE.Mesh(ribbonGeo(), M), (rnd(i, 1) - 0.5) * 0.28, 0.012 + (i % 4) * 0.02, (rnd(i, 2) - 0.5) * 0.28);
    rb.rotation.set(-1.2 + (rnd(i, 3) - 0.5) * 1.2, rnd(i, 4) * 6.28, (rnd(i, 5) - 0.5) * 0.8);
  }
  return g;
}
function lettuceHead() {
  const g = G(); const M = mLeaf();
  for (let i = 0; i < 10; i++) {
    const a = i / 10 * 6.283, rr = i % 2 ? 0.1 : 0.05;
    const leaf = put(g, new THREE.Mesh(disp("leafblob" + (i % 3), sph(0.11, 20, 12), 0.012, 16, 100 + i), M), Math.cos(a) * rr, 0.06 + (i % 3) * 0.022, Math.sin(a) * rr);
    leaf.scale.set(1.15, 0.38, 0.8); leaf.rotation.y = a + 0.4;
  }
  return g;
}

function onionRaw() {
  const g = G();
  put(g, new THREE.Mesh(disp("onion", sph(0.16, 36, 24), 0.004, 8, 110), mOnionSkin()), 0, 0.16).scale.set(1, 1.08, 1);
  put(g, new THREE.Mesh(geo("onionstem", () => new THREE.CylinderGeometry(0.0, 0.035, 0.09, 8)), phys({ color: 0xb8894a, roughness: 0.8 })), 0, 0.36);
  return g;
}
function onionPieces(cooked) {
  const g = G(); const M = mOnionFlesh(cooked);
  for (let i = 0; i < 9; i++) {
    const s = 0.06 + rnd(i, 1) * 0.03;
    const d = put(g, new THREE.Mesh(box(s, cooked ? 0.02 : s * 0.5, s, 1, 1, 1), M), (rnd(i, 2) - 0.5) * 0.24, (cooked ? 0.012 : s * 0.25) + (i > 6 ? 0.04 : 0), (rnd(i, 3) - 0.5) * 0.24);
    d.rotation.set((rnd(i, 4) - 0.5) * 0.3, rnd(i, 5) * 6.28, 0);
  }
  return g;
}

function mushroomRaw() {
  const g = G();
  put(g, new THREE.Mesh(geo("mstem", () => new THREE.CylinderGeometry(0.055, 0.07, 0.16, 12)), phys({ color: 0xefe6d8, roughness: 0.7 })), 0, 0.08);
  put(g, new THREE.Mesh(disp("mcap", sph(0.15, 28, 16), 0.005, 10, 120), mMushCap()), 0, 0.17).scale.set(1, 0.6, 1);
  return g;
}
function mushroomSlices() {
  const g = G();
  for (let i = 0; i < 6; i++) {
    const sl = put(g, new THREE.Mesh(box(0.12, 0.014, 0.1, 2, 1, 2), faces(mMushCap(), mMushCap(), phys({ color: 0xefe6d8, roughness: 0.7 }))), (rnd(i, 1) - 0.5) * 0.2, 0.008 + i * 0.012, (rnd(i, 2) - 0.5) * 0.2);
    sl.rotation.y = rnd(i, 3) * 3;
  }
  return g;
}
function mushroomCooked() {
  const g = G(); const M = mMushCooked();
  for (let i = 0; i < 7; i++) put(g, new THREE.Mesh(disp("mpiece", sph(0.06, 14, 10), 0.008, 30, 130), M), (rnd(i, 1) - 0.5) * 0.24, 0.045, (rnd(i, 2) - 0.5) * 0.22).scale.set(1, 0.6, 1.3);
  return g;
}

function tortilla(stack = 2) {
  const g = G();
  for (let k = 0; k < stack; k++) {
    const d = put(g, new THREE.Mesh(disp("tortilla", geo("disc", () => new THREE.CircleGeometry(0.2, 48)), 0.006, 12, 140 + k, false), mTortilla()), 0, 0.006 + k * 0.012, 0);
    d.rotation.x = -Math.PI / 2; d.rotation.z = k * 0.4;
  }
  return g;
}

function potatoRaw() {
  const g = G();
  put(g, new THREE.Mesh(disp("potato", sph(0.15, 32, 20), 0.012, 7, 150), mPotatoSkin()), 0, 0.12).scale.set(1.25, 0.85, 0.95);
  return g;
}

function patty(state) {
  const g = G();
  const cooked = state === "cooked", M = cooked ? mSearCrust() : mRawGround();
  const cyl = geo("patty", () => new THREE.CylinderGeometry(0.18, 0.176, 0.08, 40, 3));
  const p = put(g, new THREE.Mesh(disp("pattyd", cyl, 0.007, 14, 160), M), 0, 0.04);
  p.scale.set(1, cooked ? 0.9 : 1, 1);
  return g;
}

function cheeseSlices() {
  const g = G(); const M = mCheese();
  const s1 = put(g, new THREE.Mesh(box(0.3, 0.018, 0.3, 1, 1, 1), M), 0, 0.01); s1.rotation.y = 0.25;
  const s2 = put(g, new THREE.Mesh(box(0.3, 0.018, 0.3, 1, 1, 1), M), 0.012, 0.03, -0.01); s2.rotation.y = 0.6;
  return g;
}

// ---------------------------------------------------------------- the hamburger bun (photoscan)
let bunScene = null; const bunWaiters = []; let bunLoading = false;
function loadBun(cb) {
  if (bunScene) return cb(bunScene);
  bunWaiters.push(cb);
  if (bunLoading) return;
  bunLoading = true;
  new GLTFLoader().load(`${ASSET_BASE}models/hamburger_buns/hamburger_buns.gltf`, (gl) => {
    bunScene = gl.scene;
    for (const w of bunWaiters) w(bunScene);
    bunWaiters.length = 0;
  }, undefined, () => { bunWaiters.length = 0; });
}
const BUN_PARTS = {};
// the middle bun of the photoscan, normalised to footprint radius R and base y=0; `part`
// = "whole" | "heel" (bottom 42%) | "crown" (top 58%), cut by dropping triangles and
// capping the cut with a bread-crumb disc
function bunPart(scene, part, R) {
  const key = part + "_" + R;
  if (BUN_PARTS[key]) return BUN_PARTS[key].clone();
  let mesh = null;
  const node = scene.getObjectByName("hamburger_buns_02") || scene;
  node.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
  if (!mesh) return new THREE.Group();
  let g = mesh.geometry.clone().toNonIndexed();
  g.applyMatrix4(mesh.matrixWorld);
  g.computeBoundingBox();
  const bb = g.boundingBox, size = bb.getSize(new THREE.Vector3());
  const s = (2 * R) / Math.max(size.x, size.z);
  g.translate(-(bb.min.x + bb.max.x) / 2, -bb.min.y, -(bb.min.z + bb.max.z) / 2);
  g.scale(s, s, s);
  g.computeBoundingBox();
  const H = g.boundingBox.max.y, cut = H * 0.42;
  const out = new THREE.Group();
  if (part !== "whole") {
    const pos = g.attributes.position, nor = g.attributes.normal, uv = g.attributes.uv;
    const keepP = [], keepN = [], keepU = [];
    for (let t = 0; t < pos.count; t += 3) {
      const ys = [pos.getY(t), pos.getY(t + 1), pos.getY(t + 2)];
      const keep = part === "heel" ? ys.every((y) => y <= cut + 0.004) : ys.every((y) => y >= cut - 0.004);
      if (!keep) continue;
      for (let k = 0; k < 3; k++) { keepP.push(pos.getX(t + k), pos.getY(t + k), pos.getZ(t + k)); keepN.push(nor.getX(t + k), nor.getY(t + k), nor.getZ(t + k)); if (uv) keepU.push(uv.getX(t + k), uv.getY(t + k)); }
    }
    const ng = new THREE.BufferGeometry();
    ng.setAttribute("position", new THREE.Float32BufferAttribute(keepP, 3));
    ng.setAttribute("normal", new THREE.Float32BufferAttribute(keepN, 3));
    if (uv) ng.setAttribute("uv", new THREE.Float32BufferAttribute(keepU, 2));
    if (part === "crown") ng.translate(0, -cut, 0);
    g = ng;
    // cap: a crumb disc closes the cut (heel: on top; crown: underneath)
    const capR = R * 0.93, cap = new THREE.Mesh(geo("buncap" + capR, () => new THREE.CircleGeometry(capR, 40)), mCrumb());
    cap.rotation.x = part === "heel" ? -Math.PI / 2 : Math.PI / 2;
    cap.position.y = part === "heel" ? cut - 0.002 : 0.002;
    out.add(cap);
  }
  const m = new THREE.Mesh(g, mesh.material);
  m.material.side = THREE.DoubleSide;
  m.castShadow = true; m.receiveShadow = true;
  out.add(m);
  out.userData.height = part === "heel" ? cut : part === "crown" ? H - cut : H;
  BUN_PARTS[key] = out;
  return out.clone();
}
// holder that fills in when the scan is loaded; `fallback` (a Group) shows until then
export function bun(part, R, fallback) {
  const holder = new THREE.Group();
  if (fallback) holder.add(fallback);
  loadBun((scene) => {
    const p = bunPart(scene, part, R);
    holder.clear(); holder.add(p);
    holder.userData.height = p.userData.height;
  });
  return holder;
}
export const BUN_HEIGHT = { whole: 0.16, heel: 0.06, crown: 0.1 };   // approximations used by the plated stack

// ---------------------------------------------------------------- dispatch
/** A photoreal mesh for (ingredient, state) or null when the generic renderer look applies. */
export function foodMesh(name, state) {
  const s = state || "raw";
  if (s === "burnt") return null;
  const cooked = s === "cooked", chopped = s === "chopped";
  switch (name) {
    case "rice": return cooked ? riceCooked() : chopped ? null : riceRaw();
    case "potato": return cooked || chopped ? fries(cooked) : potatoRaw();
    case "steak": return cooked || chopped ? steakStrips(cooked) : steakRaw();
    case "meat": return chopped ? null : patty(s);
    case "fish": return cooked ? fishFried() : chopped ? fishSlices() : fishRaw();
    case "egg": return cooked ? eggFried() : chopped ? null : eggRaw();
    case "tomato": return cooked ? tomatoCooked() : chopped ? tomatoDice() : tomatoRaw();
    case "lettuce": return cooked ? null : chopped ? lettuceShreds() : lettuceHead();
    case "onion": return cooked ? onionPieces(true) : chopped ? onionPieces(false) : onionRaw();
    case "mushroom": return cooked ? mushroomCooked() : chopped ? mushroomSlices() : mushroomRaw();
    case "tortilla": return s === "raw" ? tortilla(2) : null;
    case "cheese": return s === "raw" ? cheeseSlices() : null;
    case "bun": return s === "raw" ? bun("whole", 0.165, null) : null;
    default: return null;
  }
}
// the plated-layer height for the bespoke looks (the stack raises the next layer by this)
export const LAYER_H = {
  "rice/cooked": 0.19, "potato/cooked": 0.14, "potato/chopped": 0.14, "steak/cooked": 0.1, "steak/chopped": 0.1,
  "steak/raw": 0.09, "fish/cooked": 0.2, "fish/chopped": 0.08, "fish/raw": 0.1, "egg/cooked": 0.075, "egg/raw": 0.2,
  "tomato/chopped": 0.1, "tomato/cooked": 0.1, "lettuce/chopped": 0.08, "onion/chopped": 0.08, "onion/cooked": 0.05,
  "mushroom/chopped": 0.08, "mushroom/cooked": 0.09, "tortilla/raw": 0.03, "cheese/raw": 0.045, "meat/raw": 0.085,
  "meat/cooked": 0.075, "potato/raw": 0.2, "rice/raw": 0.16,
};
