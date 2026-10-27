/* CookSim true-3D renderer (Three.js, WebGL2).
 *
 * Renders the kitchen as a real 3D scene: extruded counters, modelled
 * stations, capsule chefs and food meshes under physically-based materials,
 * a key directional light with soft shadow maps, a hemisphere fill, ACES tone
 * mapping and Unreal bloom for glow (oven, sparks, progress rings). Cook motion
 * is interpolated between server ticks.
 *
 * Exposes the same view API as the old renderer (so game.js is unchanged):
 *   new KitchenRenderer(canvas) · setState(state) · update(dt) · resize(W,H)
 */
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { foodMesh, bun as bunModel, LAYER_H, BUN_HEIGHT } from "./food3d.js?v=20260909e";

// REAL mode (photoreal commercial kitchen): opt-in per INSTANCE via window.__style at
// construction time. The benchmark pages never set it, so scripted/e2e/detection frames
// are byte-for-byte the classic look; the study page sets it ONLY for the player canvas
// (its hidden benchmark-observation renderer restores classic during construction).
const ASSET_BASE = "/static/assets/real/";
// asset-load bookkeeping for the headless observation page: frames must not be captured
// while PBR textures / models are still arriving (they render black until then)
window.__loadsPending = 0;
{
  const lm = THREE.DefaultLoadingManager;
  lm.onStart = () => { window.__loadsPending = Math.max(1, window.__loadsPending); };
  lm.onProgress = (_url, loaded, total) => { window.__loadsPending = Math.max(0, total - loaded); };
  lm.onLoad = () => { window.__loadsPending = 0; };
  lm.onError = () => { /* a missing file must not block capture forever: counted by onProgress */ };
}

const CH = 0.62;           // counter height (world units, tile = 1)
const WH = 1.05;           // wall height

const ING = {
  onion:    { c: 0xe8c14a, c2: 0x9ec64a },
  tomato:   { c: 0xe1452f, c2: 0x3f8f3a },
  lettuce:  { c: 0x66c244, c2: 0x4e9a36 },
  mushroom: { c: 0xb79a82, c2: 0xefe6d8 },
  meat:     { c: 0xb0563f, c2: 0x6e3526 },
  fish:     { c: 0x9cbcd6, c2: 0x6f93b0 },
  bun:      { c: 0xd9a566, c2: 0xfff2d0 },
  cheese:   { c: 0xf2c94c, c2: 0xd0a728 },
  rice:     { c: 0xf6f1e6, c2: 0xe0d8c4 },
  dough:    { c: 0xeed9a8, c2: 0xd5bd86 },
  egg:      { c: 0xfaf3e2, c2: 0xf6b73c },
  potato:   { c: 0xcb9e60, c2: 0xa67d44 },
  steak:    { c: 0xc23b2e, c2: 0xf2e8dc },
  tortilla: { c: 0xf0dfae, c2: 0xd8bf85 },
  noodles:  { c: 0xf3e3b2, c2: 0xe2c584 },
  shrimp:   { c: 0xf2997a, c2: 0xe8613f },
  lemon:    { c: 0xf5d93b, c2: 0xd9b30a },
  peas:     { c: 0x6fbf4a, c2: 0x4c9430 },
  butter:   { c: 0xf7e8b0, c2: 0xe3c96e },
  pepper:   { c: 0x3d3128, c2: 0x8a7862 },
  avocado:  { c: 0x3f6b2f, c2: 0xc8d98a },
  corn:     { c: 0xf2ce3a, c2: 0x7fae4a },
};
const COOK_COLORS = [0x4a90d9, 0xe0533d, 0x52b35a, 0xe0a52b, 0x9b59b6, 0x16a59a];
const DIR_YAW = { north: Math.PI, south: 0, east: Math.PI / 2, west: -Math.PI / 2 };
// facing offset in world (x, z): grid north = -z, south = +z, east = +x, west = -x
const DIR_VEC = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] };

function lerp(a, b, t) { return a + (b - a) * t; }
function mat(color, opts = {}) {
  return new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.72, metalness: 0.0 }, opts));
}
function tintFor(state, base) {
  if (state === "burnt") return 0x140f0c;                 // clearly charred black
  if (state === "cooked") {                                // strongly browned/seared
    const c = new THREE.Color(base); c.lerp(new THREE.Color(0x5a2606), 0.8); return c.getHex();
  }
  return base;
}

class KitchenRenderer3D {
  constructor(view) {
    this.real = window.__style === "real";      // photoreal commercial-kitchen mode
    const r = new THREE.WebGLRenderer({ canvas: view, antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: !!window.__capture });
    r.shadowMap.enabled = !window.__noshadow;   // disabled for GPU (ANGLE/Vulkan) headless recording
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = this.real ? 1.08 : 0.86;
    this.renderer = r;

    const scene = new THREE.Scene();
    if (this.real) {
      scene.background = new THREE.Color(0x2a2c30);   // replaced by the HDRI when it loads
    } else {
      scene.background = new THREE.Color(0x20222b);
      scene.fog = new THREE.Fog(0x20222b, 18, 38);    // no fog indoors in real mode
    }
    this.scene = scene;

    // image-based lighting so metal/rough materials get real reflections
    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    if (this.real) {
      // real kitchen HDRI: correct indoor reflections + a believable room outside the
      // playfield. Falls back silently to the RoomEnvironment set above if missing.
      // window.__envReady flips true once the HDRI is applied (or has failed). Headless
      // capture MUST wait for it: RGBELoader is async, so a frame rendered before it
      // lands is lit only by the dim fallback RoomEnvironment, and REAL-mode PBR
      // surfaces come out near-black. That is exactly how a full day of "photoreal"
      // captures shipped looking like a dark, flat kitchen.
      window.__envReady = false;
      new RGBELoader().load(ASSET_BASE + "env.hdr", (hdr) => {
        hdr.mapping = THREE.EquirectangularReflectionMapping;
        scene.environment = pmrem.fromEquirectangular(hdr).texture;
        scene.environmentIntensity = 1.25;
        window.__envReady = true;
      }, undefined, () => { window.__envReady = true; });   // fallback: never hang
      // visible background stays a near-black warm grey: the restaurant walls
      // enclose every play view, so anything above them should read as the dark
      // ceiling void of a real venue, not a glowing blurred skybox
      scene.background = new THREE.Color(0x141516);
      this._texLoader = new THREE.TextureLoader();
      this._pbrCache = {};
      this._gltfCache = {};
      this._foodTexCache = {};
    }

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);

    // ---- EGOCENTRIC mode (press V): first-person camera on cook 0 -------
    this.ego = null;                          // player id or null
    this._egoDir = new THREE.Vector3(0, 0, 1);
    this._egoHeld = {};                       // physical key -> synthesized arrow key
    window.__ck3d = this;
    const EGO_TURN = { w: 0, d: 1, s: 2, a: 3 };            // relative: fwd/right/back/left
    const RING = ["north", "east", "south", "west"];
    const ARROW = { north: "ArrowUp", east: "ArrowRight", south: "ArrowDown", west: "ArrowLeft" };
    const remap = (k) => {
      const c = this.cooks[this.ego];
      const dir = (c && c.dir) || "south";
      const abs = RING[(RING.indexOf(dir) + EGO_TURN[k]) % 4];
      return ARROW[abs];
    };
    window.addEventListener("keydown", (e) => {
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (k === "v") { this.toggleEgo(); return; }
      if (this.ego === null || !(k in EGO_TURN) || e.repeat) return;
      e.stopImmediatePropagation(); e.preventDefault();
      const a = remap(k); this._egoHeld[k] = a;
      window.dispatchEvent(new KeyboardEvent("keydown", { key: a }));
    }, true);
    window.addEventListener("keyup", (e) => {
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      // release even if ego was toggled off mid-hold, so no arrow gets stuck
      if (!(k in this._egoHeld)) return;
      e.stopImmediatePropagation(); e.preventDefault();
      window.dispatchEvent(new KeyboardEvent("keyup", { key: this._egoHeld[k] }));
      delete this._egoHeld[k];
    }, true);

    // lighting (IBL provides ambient, so direct lights stay modest)
    const hemi = new THREE.HemisphereLight(this.real ? 0xfff8ef : 0xfff4e2,
                                           this.real ? 0x3a3632 : 0x2a2230,
                                           this.real ? 0.35 : 0.28);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(this.real ? 0xfff6e8 : 0xfff0d6, this.real ? 1.8 : 1.7);
    key.position.set(6, 12, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);   // 4096 halved real-mode fps under SwiftShader
    key.shadow.bias = -0.0006;
    key.shadow.normalBias = 0.02;
    const sc = key.shadow.camera;
    sc.near = 1; sc.far = 50; sc.left = -14; sc.right = 14; sc.top = 14; sc.bottom = -14;
    this.key = key; scene.add(key); scene.add(key.target);
    if (!this.real) {           // real mode: hemi+IBL already fill; every extra
      const fill = new THREE.DirectionalLight(0x9fb6e0, 0.22);   // light costs ~7%
      fill.position.set(-7, 6, -4); scene.add(fill);             // fps in SwiftShader
    }
    if (this.real) {
      // warm recessed ceiling downlights over the line — the signature look of a
      // working commercial kitchen (pools of warm light on stainless)
      for (const [lx, lz] of [[-2.2, -1.2], [2.2, -1.2], [0, 1.6]]) {
        const dl = new THREE.PointLight(0xffe3c0, 4.1, 9, 2);
        dl.position.set(lx, 2.95, lz);
        scene.add(dl);
      }
    }

    // postprocessing — bloom only on genuinely bright (emissive) things
    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), this.real ? 0.07 : 0.14, 0.5, 0.96);  // gentle — don't wash items into "magic light"
    this.composer.addPass(this.bloom);

    this.staticGroup = new THREE.Group(); scene.add(this.staticGroup);
    this.dynGroup = new THREE.Group(); scene.add(this.dynGroup);

    this.state = null; this._sig = ""; this.center = new THREE.Vector3();
    this.cooks = {}; this.stationDyn = {}; this.items = {}; this.particles = []; this.sprites = [];
    this._pulse = {};   // per-cook interact pulse, triggered once per state with action=interact
    this._flyers = [];  // transient meshes flying hand<->station to show transfers
    this._lastScore = 0; this._t = 0;
    this._steamTex = this._softTex(0xffffff);
    this._smokeTex = this._softTex(0x555555);
    if (this.real) this._smokeTexReal = this._softTex(0x9a9a9a);   // paler, less "floating orb"
    this._fireTex = this._softTex(0xffd27f);
    this._sparkTex = this._softTex(0xffe14a);
    this._ringTex = this._ringTexture();
    this._geo = {};   // shared geometry cache
  }

  // ---- shared geometry helpers ----------------------------------------
  g(key, make) { return this._geo[key] || (this._geo[key] = make()); }
  box(w, h, d) { return this.g(`b${w}_${h}_${d}`, () => new THREE.BoxGeometry(w, h, d)); }
  cyl(rt, rb, h, s = 20) { return this.g(`c${rt}_${rb}_${h}_${s}`, () => new THREE.CylinderGeometry(rt, rb, h, s)); }
  sph(rad, s = 16) { return this.g(`s${rad}_${s}`, () => new THREE.SphereGeometry(rad, s, s)); }

  iso() {}                       // (compat no-op)
  toWorld(x, y) { return new THREE.Vector3(x - this.center.x, 0, y - this.center.z); }

  // ---- REAL-mode PBR material library ----------------------------------
  // Each set is diffuse+normal+roughness(+ao) JPGs under /static/assets/real/.
  // Missing files degrade gracefully to the tinted base colour (loader errors are
  // swallowed), so the renderer never breaks if an asset set is absent.
  pbr(name, { repeat = 1, color = 0xffffff, roughness = 1.0, metalness = 0.0,
              rotation = 0, envI = 1.0, noRoughMap = false, normScale = 1 } = {}) {
    const key = `${name}_${repeat}_${color}_${rotation}_${envI}_${noRoughMap ? 1 : 0}_${normScale}`;
    if (this._pbrCache[key]) return this._pbrCache[key];
    // NB: a texture whose file 404s NEVER loads, and an unloaded texture samples
    // BLACK — leaving it attached as e.g. an aoMap would kill all indirect light
    // on that material (steel has no _ao file and was rendering near-black). So
    // every slot detaches itself from the material if its file fails to load.
    const ld = (suffix, srgb, slot, m) => {
      const t = this._texLoader.load(ASSET_BASE + name + "_" + suffix + ".jpg",
                                     undefined, undefined,
                                     () => { if (m()) { m()[slot] = null; m().needsUpdate = true; } });
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(repeat, repeat);
      t.rotation = rotation;
      t.anisotropy = 8;
      if (srgb) t.colorSpace = THREE.SRGBColorSpace;
      return t;
    };
    let mref = null; const m = () => mref;
    mref = new THREE.MeshStandardMaterial({
      color, roughness, metalness,
      map: ld("diff", true, "map", m), normalMap: ld("nor", false, "normalMap", m),
      // steel's rough map is near-black (≈0.03): multiplied in it turns every
      // surface into a chrome mirror, so that set uses a flat roughness value
      roughnessMap: noRoughMap ? null : ld("rough", false, "roughnessMap", m),
      aoMap: ld("ao", false, "aoMap", m), aoMapIntensity: 0.85,
    });
    mref.envMapIntensity = envI;
    mref.normalScale = new THREE.Vector2(normScale, normScale);
    this._pbrCache[key] = mref;
    return mref;
  }
  // brushed stainless — the workhorse of every commercial kitchen surface
  steel(opts = {}) { return this.pbr("steel", Object.assign({ metalness: 1.0, roughness: 0.27, repeat: 1.6, envI: 1.25, noRoughMap: true, color: 0xcfd4d8, normScale: 1.5 }, opts)); }

  // Photoreal GLTF prop (Poly Haven, textures load automatically). Returns a
  // holder group immediately; the model is normalised (footprint diameter =
  // 2*targetR, base at y) and dropped in asynchronously. A load failure just
  // leaves the holder empty, so callers keep primitive fallbacks visible until
  // the model arrives (or forever, if the asset is missing).
  model(name, { targetR = 0.3, y = 0, rot = null, centerBowl = false, onReady = null } = {}) {
    const holder = new THREE.Group();
    const place = (scene) => {
      const inst = scene.clone(true);
      const pivot = new THREE.Group(); pivot.add(inst);
      if (rot) pivot.rotation.set(rot[0], rot[1], rot[2]);   // e.g. lay a tall model flat
      const bb = new THREE.Box3().setFromObject(pivot);
      const size = bb.getSize(new THREE.Vector3());
      // normalise by the FOOTPRINT after rotation, but never let a thin/tall
      // model blow up: cap the scale using the largest overall dimension too
      const s = Math.min((2 * targetR) / Math.max(size.x, size.z),
                         (2.6 * targetR) / Math.max(size.x, size.y, size.z));
      pivot.scale.setScalar(s);
      bb.setFromObject(pivot);
      pivot.position.set(-(bb.min.x + bb.max.x) / 2, y - bb.min.y, -(bb.min.z + bb.max.z) / 2);
      if (centerBowl) {
        // pan-style prop: centre the round BOWL (short horizontal extent) on the
        // origin instead of the bbox midpoint, so the handle overhangs naturally
        bb.setFromObject(pivot);
        const sz = bb.getSize(new THREE.Vector3());
        if (sz.z >= sz.x) pivot.position.z -= (bb.max.z - sz.x / 2);
        else pivot.position.x -= (bb.max.x - sz.z / 2);
      }
      pivot.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      holder.add(pivot);
      if (onReady) onReady(holder);
    };
    const c = this._gltfCache;
    if (c[name]) {
      if (c[name].scene) place(c[name].scene); else c[name].cbs.push(place);
    } else {
      c[name] = { scene: null, cbs: [place] };
      new GLTFLoader().load(`${ASSET_BASE}models/${name}/${name}.gltf`, (g) => {
        c[name].scene = g.scene;
        for (const cb of c[name].cbs) cb(g.scene);
        c[name].cbs = [];
      }, undefined, () => { c[name].cbs = []; });
    }
    return holder;
  }

  // ---- camera ----------------------------------------------------------
  // A follow-camera keeps the cook large & centred so every action (hands,
  // station, held item, state change) is readable — far more legible than a
  // fit-the-whole-kitchen shot where the cook is a few pixels tall.
  // Render the first-person view with NO temporal easing, for callers that produce one
  // independent frame per game tick (the headless capture the benchmark observes and
  // films). Interactive callers must leave this off: there the easing is what stops the
  // view from snapping around the player's head.
  setEgoSnap(on = true) {
    this.egoSnap = !!on;
    if (on && this.ego !== null && this.cooks[this.ego]) {
      const v = DIR_VEC[this.cooks[this.ego].dir || "south"] || [0, 1];
      this._egoDir.set(v[0], 0, v[1]);
    }
  }

  // FILM MODE — explicit sub-tick interpolation, for callers that render SEVERAL frames
  // per game tick (presentation video) rather than one (the benchmark observation).
  //
  // The default cook interpolation is exponential: `lerp(c, p, min(1, dt*12))`. That is
  // right for an interactive session, where the target updates every ~1.2 s and the cook
  // eases in over ~0.2 s. It CANNOT be retuned to span a tick: to make one tick's motion
  // take a whole tick you need a per-frame factor near 1/N, and an exponential with that
  // factor only covers ~64% of the distance in N frames — so the rendered cook falls
  // permanently behind the simulation, and the error compounds every tick.
  //
  // So film mode interpolates LINEARLY from where the cook was rendered when the target
  // last changed to the new target, by an alpha the caller advances 0 -> 1 across the
  // tick. Position is then exact at alpha=1, with no accumulated lag. Pass null to
  // restore the interactive/observation behaviour.
  setFilmLerp(alpha = null) {
    this.filmLerp = (alpha === null || alpha === undefined) ? null : Math.max(0, Math.min(1, alpha));
  }

  toggleEgo(pid = 0) {
    this.ego = (this.ego === null) ? pid : null;
    this.camera.fov = this.ego !== null ? 72 : 38;
    this.camera.updateProjectionMatrix();
    for (const id in this.cooks) this.cooks[id].grp.visible = !(this.ego !== null && +id === this.ego);
    if (this.ego !== null) {
      // HUD pill: what the ego cook is holding (updated every setState)
      if (!this._egoHud) {
        const el = document.createElement("div");
        el.id = "ck3dEgoHud";
        el.style.cssText = "position:fixed;left:50%;bottom:18px;transform:translateX(-50%);" +
          "background:rgba(15,16,22,.9);color:#f6f4ee;padding:9px 18px;border-radius:999px;" +
          "font:600 15px system-ui,sans-serif;z-index:40;pointer-events:none;white-space:nowrap;" +
          "box-shadow:0 2px 12px rgba(0,0,0,.5);border:1px solid rgba(255,255,255,.14)";
        document.body.appendChild(el);
        this._egoHud = el;
      }
      this._updateEgoHud();
    } else {
      // flush any still-held synthesized movement keys so the cook stops
      for (const k in this._egoHeld) {
        window.dispatchEvent(new KeyboardEvent("keyup", { key: this._egoHeld[k] }));
        delete this._egoHeld[k];
      }
      if (this._egoHud) { this._egoHud.remove(); this._egoHud = null; }
      if (this._egoMarker) this._egoMarker.visible = false;
      if (this._egoHeldMesh) { this.scene.remove(this._egoHeldMesh); this._egoHeldMesh = null; }
      this._egoHeldSig = "0";
      // restore other cooks' badge size/opacity altered by the ego camera
      for (const id in this.cooks) {
        const b = this.cooks[id].grp.userData.badge;
        if (b) { b.scale.set(0.4, 0.4, 1); b.material.opacity = 1; }
      }
    }
    window.dispatchEvent(new CustomEvent("ck3d-ego", { detail: { ego: this.ego } }));
  }

  _egoItemLabel(it) { return ((it.state && it.state !== "raw") ? it.state + " " : "") + it.name; }
  _updateEgoHud() {
    if (!this._egoHud || this.ego === null || !this.state) return;
    const p = (this.state.players || []).find(pp => pp.id === this.ego);
    const h = p && p.holding;
    let txt;
    if (!h) txt = "\u{1F91A} (empty hands)";
    else if (h.kind === "plate") {
      const c = (h.contents || []).map(it => this._egoItemLabel(it));
      txt = "\u{1F91A} holding: plate" + (c.length ? " — " + c.join(", ") : " (empty)");
    } else txt = "\u{1F91A} holding: " + this._egoItemLabel(h);
    this._egoHud.textContent = txt;
  }

  fit(W, H) {
    if (!this.state) return;
    const w = this.state.width, h = this.state.height;
    this.center.set((w - 1) / 2, 0, (h - 1) / 2);
    const cam = this.camera;
    cam.aspect = W / H;
    // view radius: cook stays large & readable, but enough kitchen shows that
    // the stations it's working (and their progress rings) stay on screen
    // real mode frames slightly wider/lower so the room (hood, walls, dining)
    // gives context; classic keeps the tight legacy framing untouched
    this._viewR = this.real ? 4.1 : 3.7;
    this._camDist = this._viewR / Math.tan((cam.fov * Math.PI / 180) / 2);
    cam.updateProjectionMatrix();
    if (!this._camFocus) this._camFocus = new THREE.Vector3(0, 0, 0);
  }

  _updateCamera(dt) {
    if (!this._camDist) return;
    if (this.ego !== null && this.cooks[this.ego]) {
      const c = this.cooks[this.ego];
      c.grp.visible = false;
      const wp = this.toWorld(c.x, c.y);
      const v = DIR_VEC[c.dir || "south"] || [0, 1];
      // egoSnap: no easing. The smoothing below is right for an interactive session at
      // 60 fps (a turn lands in ~0.3 s) and catastrophically wrong for a headless capture
      // that renders ONE frame per game tick: there dt is 0.01, so k is 0.06 and the
      // first-person view completes 6% of a turn per tick — ~48 ticks for a 90-degree
      // turn the simulation performs in exactly one. See snapEgo().
      const k = this.egoSnap ? 1 : Math.min(1, dt * 6);
      this._egoDir.x += (v[0] - this._egoDir.x) * k;
      this._egoDir.z += (v[1] - this._egoDir.z) * k;
      const d = this._egoDir.clone().normalize();
      // eye clearly above counter/crate tops with a downward pitch so counter
      // TOPS (and what sits on them) are visible, not just cabinet fronts
      this.camera.position.set(wp.x - d.x * 0.22, 1.32, wp.z - d.z * 0.22);
      this.camera.lookAt(wp.x + d.x * 1.8, 0.24, wp.z + d.z * 1.8);
      this.key.target.position.set(wp.x, 0, wp.z);
      // first-person viewmodel: the held item floats bottom-right "in hand",
      // always drawn on top so the player never has to read the HUD to know
      const p0h = (this.state && this.state.players || []).find(pp => pp.id === this.ego);
      const hsig = this._itemSig(p0h && p0h.holding);
      if (hsig !== this._egoHeldSig) {
        if (this._egoHeldMesh) { this.scene.remove(this._egoHeldMesh); this._egoHeldMesh = null; }
        if (p0h && p0h.holding) {
          const m = this._itemMesh(p0h.holding);
          m.scale.setScalar(0.5);
          // a stylized glove under the item grounds it as "in MY hand" (not a
          // loose object on a nearby counter). Real mode: neutral white chef
          // sleeve/hand, not a cartoon-coloured ball.
          const glove = new THREE.Mesh(this.sph(0.13),
            mat(this.real ? 0xf4f1ea : COOK_COLORS[this.ego % COOK_COLORS.length], { roughness: this.real ? 0.8 : 0.6 }));
          glove.scale.set(1.15, 0.55, 1.15); glove.position.y = -0.045;
          m.add(glove);
          // depth testing stays ON (2026-09-09): the viewmodel sits inside the cook's own
          // tile at hand height, so no counter can occlude it, while depthTest=false made
          // every closed mesh paint its back faces over its front (seared strips showed
          // their undersides, rice showed the far side of the mound)
          m.traverse(o => { if (o.isMesh) { o.castShadow = false; o.renderOrder = 999; } });
          glove.renderOrder = 998;                 // glove behind the item it cups
          this.scene.add(m);
          this._egoHeldMesh = m;
        }
        this._egoHeldSig = hsig;
      }
      if (this._egoHeldMesh) {
        const right = d.clone().cross(new THREE.Vector3(0, 1, 0));
        // low and near bottom-centre: reads as "in my hands", and cannot be
        // mistaken for a loose world object sitting on a nearby counter
        this._egoHeldMesh.position.set(
          this.camera.position.x + d.x * 0.5 + right.x * 0.11,
          this.camera.position.y - 0.5 + Math.sin(this._t * 1.7) * 0.008,
          this.camera.position.z + d.z * 0.5 + right.z * 0.11);
      }
      // faced-tile marker: subtle ring on the tile Space will interact with
      const p0 = (this.state && this.state.players || []).find(pp => pp.id === this.ego);
      if (p0) {
        if (!this._egoMarker) {
          const m = new THREE.Mesh(
            new THREE.RingGeometry(0.26, 0.4, 32),
            new THREE.MeshBasicMaterial({ color: 0xffe14a, transparent: true, opacity: 0.55,
                                          depthWrite: false, depthTest: false, side: THREE.DoubleSide }));
          m.rotation.x = -Math.PI / 2;
          m.renderOrder = 997;                       // never hidden behind crates/counters
          this.scene.add(m);
          this._egoMarker = m;
        }
        const fv = DIR_VEC[p0.dir_name] || [0, 1];
        const tx = p0.x + fv[0], ty = p0.y + fv[1];
        const mp = this.toWorld(tx, ty);
        const trow = this.state.terrain[ty];
        const tt = (trow && trow[tx]) || "floor";
        const onCounter = tt !== "floor";
        // sit the ring on the TOP of whatever occupies the faced tile, so it
        // never slices through a crate/oven front when seen edge-on
        let my = 0.03;
        if (onCounter) {
          my = CH + 0.09;
          if (tt.endsWith("_source") && tt !== "plate_source") my = CH + 0.5;   // crate top
          else if (tt === "oven") my = CH + 0.7;
          else if (tt === "fryer") my = CH + 0.56;
          else if (tt === "pot") my = CH + 0.44;
          else if (tt === "trash" && this.real) my = 0.93;   // floor-standing bin
          else if (tt === "wall") my = WH + 0.02;
        }
        this._egoMarker.position.set(mp.x, my, mp.z);
        // bright yellow on interactable tiles; subdued cool ring on plain floor
        // (a bold yellow ring on the floor wrongly suggested "drop it here")
        this._egoMarker.material.opacity = onCounter ? 0.6 : 0.28;
        this._egoMarker.material.color.setHex(onCounter ? 0xffe14a : 0x9fb6e0);
        this._egoMarker.scale.setScalar(onCounter ? 1 : 0.8);
        this._egoMarker.visible = true;
      } else if (this._egoMarker) this._egoMarker.visible = false;
      // other cooks' overhead badges: cap on-screen size & fade when very close
      for (const id in this.cooks) {
        if (+id === this.ego) continue;
        const b = this.cooks[id].grp.userData.badge; if (!b) continue;
        const dc = this.camera.position.distanceTo(this.cooks[id].grp.position);
        const s = 0.4 * Math.min(1, dc / 3.5);      // sprite scale ∝ distance ⇒ capped screen size
        b.scale.set(s, s, 1);
        b.material.opacity = dc < 1.5 ? Math.max(0, (dc - 0.6) / 0.9) : 1;
      }
      return;
    }
    let fx = 0, fz = 0, n = 0;
    for (const id in this.cooks) { const c = this.cooks[id]; const wp = this.toWorld(c.x, c.y); fx += wp.x; fz += wp.z; n++; }
    if (n) {
      const k = Math.min(1, dt * 3.5);
      this._camFocus.x += (fx / n - this._camFocus.x) * k;
      this._camFocus.z += (fz / n - this._camFocus.z) * k;
    }
    // clamp focus so the view never drifts off the kitchen into empty space
    const halfW = (this.state.width - 1) / 2, halfH = (this.state.height - 1) / 2;
    const mx = Math.max(0, halfW - (this._viewR - 2)), mz = Math.max(0, halfH - (this._viewR - 2));
    const f = this._camFocus;
    f.x = Math.max(-mx, Math.min(mx, f.x));
    f.z = Math.max(-mz, Math.min(mz, f.z));
    const d = this._camDist;
    if (this.real) this.camera.position.set(f.x, d * 0.76, f.z + d * 0.71);
    else this.camera.position.set(f.x, d * 0.82, f.z + d * 0.66);
    this.camera.lookAt(f.x, 0.3, f.z);
    this.key.target.position.set(f.x, 0, f.z);
  }

  setState(state) {
    this.state = state;
    const sig = state.width + "x" + state.height + ":" + JSON.stringify(state.terrain);
    if (sig !== this._sig) { this._sig = sig; this._buildStatic(); this._resetDyn(); }
    const dScore = state.score - (this._lastScore || 0);
    if (this._lastScore !== undefined && dScore > 0.5) this._delivery(Math.round(dScore));
    this._lastScore = state.score;
    const fails = (state.stats && state.stats.failed_deliveries) || 0;
    if (this._lastFails !== undefined && fails > this._lastFails) this._failFeedback();
    this._lastFails = fails;
    // each state where a cook is interacting fires one fresh action pulse
    for (const p of state.players) if (p.action === "interact") this._pulse[p.id] = true;
    this._updateEgoHud();
    this.fit(this.renderer.domElement.width / this.renderer.getPixelRatio(),
             this.renderer.domElement.height / this.renderer.getPixelRatio());
  }

  _resetDyn() {
    this.dynGroup.clear();
    this.cooks = {}; this.stationDyn = {}; this.items = {};
    for (const s of this.sprites) this.scene.remove(s.spr);
    this.sprites = []; this.particles = [];
  }

  // ---- static geometry -------------------------------------------------
  _buildStatic() {
    this.staticGroup.clear();
    this._ovenWins = {};                       // rebuilt with the fixtures below
    const st = this.state, w = st.width, h = st.height;
    this.center.set((w - 1) / 2, 0, (h - 1) / 2);

    // floor: real mode = ceramic kitchen tiles (PBR); classic = painted checker
    const floorMat = this.real
      ? this.pbr("tile", { repeat: Math.max(w, h) * 0.9, roughness: 0.55, normScale: 1.4 })
      : new THREE.MeshStandardMaterial({ map: this._floorTex(w, h), roughness: 0.92 });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(w, h), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set((w - 1) / 2 - this.center.x, 0, (h - 1) / 2 - this.center.z);
    floor.receiveShadow = true;
    this.staticGroup.add(floor);
    // skirting / base platform under everything
    // real mode: dark recessed curb — reads as a shadow gap under the equipment
    // line rather than a pale display plinth
    const baseM = this.real ? mat(0x3a3b3d, { roughness: 0.92 })
                            : mat(0x2c2733, { roughness: 0.95 });
    const base = new THREE.Mesh(new THREE.BoxGeometry(w + 0.4, 0.3, h + 0.4), baseM);
    base.position.set((w - 1) / 2 - this.center.x, -0.16, (h - 1) / 2 - this.center.z);
    base.receiveShadow = true; this.staticGroup.add(base);
    if (this.real) this._buildSurround(st);   // full restaurant around the kitchen block

    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const t = st.terrain[y][x];
      if (t === "floor") continue;
      const p = this.toWorld(x, y);
      const yaw = this._floorYaw(st, x, y);
      if (t === "wall") { this.staticGroup.add(this._wall(p)); continue; }
      // real mode: the trash tile is a floor-standing pedal bin, no cabinet under it
      if (!(this.real && t === "trash")) this.staticGroup.add(this._counter(p, yaw, t === "counter"));
      const fx = this._fixture(t, p, yaw);
      if (fx) this.staticGroup.add(fx);
    }
    if (this.real) {
      // contact-shadow strips where the floor meets the counter line (real rooms
      // are darkest in exactly these crevices) + a center floor drain
      if (!this._edgeShTex) {
        const C = document.createElement("canvas"); C.width = 8; C.height = 32;
        const g2 = C.getContext("2d");
        const gr = g2.createLinearGradient(0, 0, 0, 32);
        gr.addColorStop(0, "rgba(0,0,0,.6)"); gr.addColorStop(1, "rgba(0,0,0,0)");
        g2.fillStyle = gr; g2.fillRect(0, 0, 8, 32);
        this._edgeShTex = new THREE.CanvasTexture(C);
      }
      const eM = new THREE.MeshBasicMaterial({ map: this._edgeShTex, transparent: true, opacity: 0.42, depthWrite: false });
      const DIRS = [[0, -1, 0], [0, 1, Math.PI], [1, 0, -Math.PI / 2], [-1, 0, Math.PI / 2]];
      let fx0 = 0, fy0 = 0, nf = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        if (st.terrain[y][x] !== "floor") continue;
        fx0 += x; fy0 += y; nf++;
        for (const [dx, dy, rz] of DIRS) {
          const row = st.terrain[y + dy];
          if (row === undefined || row[x + dx] === undefined || row[x + dx] === "floor") continue;
          const p = this.toWorld(x, y);
          const s = new THREE.Mesh(this.g("edgesh", () => new THREE.PlaneGeometry(1.0, 0.26)), eM);
          s.rotation.set(-Math.PI / 2, 0, rz);
          s.position.set(p.x + dx * 0.37, 0.004, p.z + dy * 0.37);
          s.renderOrder = 1;
          this.staticGroup.add(s);
        }
      }
      if (nf) {          // round steel drain in the middle of the work aisle
        const dx2 = Math.round(fx0 / nf), dy2 = Math.round(fy0 / nf);
        if ((st.terrain[dy2] || [])[dx2] === "floor") {
          const dp = this.toWorld(dx2, dy2);
          const drainG = new THREE.Group(); drainG.position.set(dp.x + 0.02, 0, dp.z + 0.03);
          const ring = new THREE.Mesh(this.cyl(0.17, 0.17, 0.012, 20), mat(0x54585c, { metalness: 0.8, roughness: 0.4 }));
          ring.position.y = 0.005; drainG.add(ring);
          const plate = new THREE.Mesh(this.cyl(0.14, 0.14, 0.014, 20), mat(0x2b2e31, { metalness: 0.7, roughness: 0.5 }));
          plate.position.y = 0.007; drainG.add(plate);
          for (let i = 0; i < 4; i++) {
            const slot = new THREE.Mesh(this.box(0.2, 0.004, 0.016), mat(0x0d0e10, { roughness: 0.9 }));
            slot.position.y = 0.016; slot.rotation.y = i * Math.PI / 4; drainG.add(slot);
          }
          this.staticGroup.add(drainG);
        }
      }
    }
  }

  // yaw that turns a fixture's local +z ("front") toward the nearest walkable
  // floor tile, so labels/handles/knobs face the aisle a cook works from
  _floorYaw(st, x, y) {
    const opts = [[0, 1, 0], [0, -1, Math.PI], [1, 0, Math.PI / 2], [-1, 0, -Math.PI / 2]];
    for (const [dx, dy, yaw] of opts) {
      const row = st.terrain[y + dy];
      if (row && row[x + dx] === "floor") return yaw;
    }
    return 0;
  }

  _counter(p, yaw = 0, allowClutter = false) {
    const grp = new THREE.Group();
    if (this.real) {
      // commercial stainless prep table: brushed steel cabinet, seamless steel top
      // with a rolled front edge, dark recessed kick at the bottom. Door seams +
      // a pull handle live on the side that faces the aisle (yaw).
      if (!this._steelBody) {
        this._steelBody = this.steel({ roughness: 0.38 });
        this._steelTop = this.steel({ roughness: 0.38, repeat: 1.1, rotation: Math.PI / 2 });
        this._kickMat = mat(0x1b1d20, { roughness: 0.9, metalness: 0.2 });
        this._seamMat = mat(0x2e3134, { roughness: 0.6, metalness: 0.6 });
      }
      // bodies nearly fill the tile: adjacent units sit flush like a real line
      // (0.94 left lit floor slivers showing through every unit-to-unit gap)
      const kick = new THREE.Mesh(this.box(0.95, 0.09, 0.95), this._kickMat);
      kick.position.set(p.x, 0.045, p.z); grp.add(kick);
      const body = new THREE.Mesh(this.box(0.985, CH - 0.13, 0.985), this._steelBody);
      body.position.set(p.x, 0.09 + (CH - 0.13) / 2, p.z);
      body.castShadow = true; body.receiveShadow = true; grp.add(body);
      const top = new THREE.Mesh(this.box(1.0, 0.05, 1.0), this._steelTop);
      top.position.set(p.x, CH + 0.025, p.z); top.castShadow = true; top.receiveShadow = true;
      grp.add(top);
      // faint wipe-marks / smudges so the steel doesn't read factory-fresh; the
      // shared texture is rotated per tile so no two counters match
      if (!this._smudgeM) {
        const C = document.createElement("canvas"); C.width = C.height = 128;
        const cx2 = C.getContext("2d");
        for (let i = 0; i < 14; i++) {          // barely-there wipe arcs
          cx2.strokeStyle = `rgba(255,255,255,${0.02 + (i % 3) * 0.012})`;
          cx2.lineWidth = 3 + (i % 4) * 2;
          cx2.beginPath();
          cx2.arc((i * 47) % 128, (i * 71 + 30) % 128, 10 + (i * 13) % 22, i, i + 1.6);
          cx2.stroke();
        }
        for (let i = 0; i < 6; i++) {
          cx2.fillStyle = "rgba(30,32,34,.025)";
          cx2.beginPath();
          cx2.ellipse((i * 53 + 20) % 128, (i * 89 + 50) % 128, 12, 5, i * 1.3, 0, 6.28);
          cx2.fill();
        }
        const st = new THREE.CanvasTexture(C);
        this._smudgeM = new THREE.MeshStandardMaterial({ map: st, transparent: true, opacity: 0.22,
          roughness: 0.24, metalness: 0.5, depthWrite: false });
      }
      const sm = new THREE.Mesh(this.g("smpl", () => new THREE.PlaneGeometry(0.96, 0.96)), this._smudgeM);
      sm.rotation.x = -Math.PI / 2;
      sm.rotation.z = (Math.abs(Math.round(p.x * 3 + p.z * 5)) % 4) * Math.PI / 2;
      sm.position.set(p.x, CH + 0.0505, p.z); sm.renderOrder = 1;
      grp.add(sm);
      const front = new THREE.Group(); front.position.set(p.x, 0, p.z); front.rotation.y = yaw;
      const lip = new THREE.Mesh(this.g("clip", () => new THREE.CylinderGeometry(0.028, 0.028, 1.0, 10)), this._steelTop);
      lip.rotation.z = Math.PI / 2; lip.position.set(0, CH + 0.045, 0.5); front.add(lip);
      // recessed double-door seam lines + a horizontal pull bar
      const seamV = new THREE.Mesh(this.box(0.012, CH - 0.17, 0.012), this._seamMat);
      seamV.position.set(0, 0.09 + (CH - 0.13) / 2, 0.494); front.add(seamV);
      const seamH = new THREE.Mesh(this.box(0.9, 0.012, 0.012), this._seamMat);
      seamH.position.set(0, 0.115, 0.494); front.add(seamH);
      for (const sx of [-0.21, 0.21]) {
        const hdl = new THREE.Mesh(this.g("chdl", () => new THREE.CylinderGeometry(0.014, 0.014, 0.3, 8)), this._steelTop);
        hdl.rotation.x = Math.PI / 2; hdl.rotation.z = Math.PI / 2;
        hdl.position.set(sx, CH - 0.19, 0.507); front.add(hdl);
      }
      grp.add(front);
      // working clutter along the BACK lip of plain counters (centre stays free
      // for placed items): folded towel / salt+pepper+oil / bain with a spoon.
      // Deterministic per tile so the kitchen looks stocked but stable.
      if (allowClutter) {
        const hsh = Math.abs(Math.round(p.x * 37.1 + p.z * 91.7 + p.x * p.z * 13.3)) % 5;
        if (hsh < 3) {
          const cl = new THREE.Group(); cl.position.set(p.x, CH + 0.05, p.z); cl.rotation.y = yaw;
          const off = ((Math.abs(Math.round(p.x * 53 + p.z * 29)) % 3) - 1) * 0.24;
          if (hsh === 0) {                     // folded side towel, blue stripe
            const tw = new THREE.Mesh(this.box(0.24, 0.024, 0.16), mat(0xf3f1e9, { roughness: 0.95 }));
            tw.position.set(off, 0.012, -0.33); tw.rotation.y = 0.35; cl.add(tw);
            const st = new THREE.Mesh(this.box(0.245, 0.006, 0.03), mat(0x4a6a92, { roughness: 0.95 }));
            st.position.set(off, 0.027, -0.33); st.rotation.y = 0.35; cl.add(st);
          } else if (hsh === 1) {              // salt & pepper + oil squeeze bottle
            const salt = new THREE.Mesh(this.cyl(0.032, 0.035, 0.1, 10), mat(0xf0efe9, { roughness: 0.5 }));
            salt.position.set(off - 0.05, 0.05, -0.34); cl.add(salt);
            const pep = new THREE.Mesh(this.cyl(0.032, 0.035, 0.1, 10), mat(0x3c3e42, { roughness: 0.5 }));
            pep.position.set(off + 0.04, 0.05, -0.31); cl.add(pep);
            const oil = new THREE.Mesh(this.cyl(0.038, 0.042, 0.16, 10), mat(0xc23b28, { roughness: 0.45 }));
            oil.position.set(off + 0.14, 0.08, -0.36); cl.add(oil);
            const tip = new THREE.Mesh(this.g("oiltip", () => new THREE.ConeGeometry(0.02, 0.06, 10)), mat(0xc23b28, { roughness: 0.45 }));
            tip.position.set(off + 0.14, 0.19, -0.36); cl.add(tip);
          } else {                             // steel bain with a spoon standing in it
            const bain = new THREE.Mesh(this.cyl(0.085, 0.075, 0.12, 14), this.steel({ roughness: 0.35 }));
            bain.position.set(off, 0.06, -0.33); cl.add(bain);
            const spoon = new THREE.Mesh(this.cyl(0.012, 0.012, 0.3, 6), mat(0xb9bec3, { metalness: 0.85, roughness: 0.3 }));
            spoon.position.set(off + 0.03, 0.2, -0.35); spoon.rotation.z = 0.4; cl.add(spoon);
          }
          cl.traverse(o => { if (o.isMesh) o.castShadow = true; });
          grp.add(cl);
        }
      }
      return grp;
    }
    if (!this._woodBody) {
      this._woodBody = new THREE.MeshStandardMaterial({ map: this._woodTex(0xa8794a, 0x8a6038), roughness: 0.62, metalness: 0.04 });
      this._woodTop = new THREE.MeshStandardMaterial({ map: this._woodTex(0xd9b683, 0xc09a64), roughness: 0.4, metalness: 0.04 });
    }
    const body = new THREE.Mesh(this.box(0.94, CH, 0.94), this._woodBody);
    body.position.set(p.x, CH / 2, p.z); body.castShadow = true; body.receiveShadow = true;
    const top = new THREE.Mesh(this.box(0.98, 0.1, 0.98), this._woodTop);
    top.position.set(p.x, CH + 0.02, p.z); top.castShadow = true; top.receiveShadow = true;
    grp.add(body, top); return grp;
  }
  _wall(p) {
    if (this.real) {
      // painted plaster wall with a white subway-tile splashback band — what you
      // actually see behind a commercial line
      const grp = new THREE.Group();
      if (!this._plasterMat) {
        this._plasterMat = this.pbr("plaster", { repeat: 1.2, roughness: 0.9, color: 0xe8e2d8 });
        this._splashMat = this.pbr("tile", { repeat: 2.2, roughness: 0.35, color: 0xf2f2ee });
      }
      const m = new THREE.Mesh(this.box(0.98, WH, 0.98), this._plasterMat);
      m.position.set(p.x, WH / 2, p.z); m.castShadow = true; m.receiveShadow = true; grp.add(m);
      const band = new THREE.Mesh(this.box(1.0, 0.42, 1.0), this._splashMat);
      band.position.set(p.x, CH + 0.24, p.z); band.receiveShadow = true; grp.add(band);
      return grp;
    }
    const m = new THREE.Mesh(this.box(0.98, WH, 0.98), mat(0x6f5c52, { roughness: 0.85 }));
    m.position.set(p.x, WH / 2, p.z); m.castShadow = true; m.receiveShadow = true; return m;
  }

  // station fixtures (static parts) -------------------------------------
  _fixture(t, p, yaw = 0) {
    const grp = new THREE.Group(); grp.position.set(p.x, CH + 0.07, p.z);
    if (this.real && t !== "counter") {
      this._fixYaw = yaw;                           // aisle-facing yaw for labels/knobs
      const fx = this._fixtureReal(t, grp);
      if (fx !== undefined) {                       // real handler took the station
        grp.traverse(o => { if (o.isMesh) o.castShadow = true; });
        return grp;
      }
    }
    const top = 0;
    if (t === "counter") return null;
    if (t === "pot") {
      // tall cylindrical STOCKPOT on a visible burner: two side handles + a lid
      // sitting ajar — instantly reads as "pot", not a bin or a pan
      const burner = new THREE.Mesh(this.cyl(0.4, 0.42, 0.07, 24), mat(0x24262c, { metalness: 0.5, roughness: 0.55 }));
      burner.position.y = 0.035; grp.add(burner);
      const coil = new THREE.Mesh(this.g("burncoil", () => new THREE.TorusGeometry(0.3, 0.02, 8, 28)),
        new THREE.MeshStandardMaterial({ color: 0x91342a, emissive: 0xb2402e, emissiveIntensity: 0.6, roughness: 0.5 }));
      coil.rotation.x = Math.PI / 2; coil.position.y = 0.072; grp.add(coil);
      const body = new THREE.Mesh(this.cyl(0.3, 0.29, 0.5, 24), mat(0xc2c7cc, { metalness: 0.85, roughness: 0.3 }));
      body.position.y = 0.33; body.castShadow = true; grp.add(body);
      const rim = new THREE.Mesh(this.g("potrim2", () => new THREE.TorusGeometry(0.295, 0.022, 10, 24)), mat(0xe6eaee, { metalness: 0.9, roughness: 0.25 }));
      rim.rotation.x = Math.PI / 2; rim.position.y = 0.58; grp.add(rim);
      for (const sx of [-1, 1]) {              // two stockpot side handles
        const h = new THREE.Mesh(this.g("pothandle", () => new THREE.TorusGeometry(0.055, 0.016, 8, 14, Math.PI)), mat(0x8f959c, { metalness: 0.8, roughness: 0.35 }));
        h.position.set(sx * 0.31, 0.48, 0); h.rotation.z = sx > 0 ? -Math.PI / 2 : Math.PI / 2; grp.add(h);
      }
      const lid = new THREE.Mesh(this.cyl(0.3, 0.3, 0.035, 24), mat(0xd7dce0, { metalness: 0.88, roughness: 0.28 }));
      lid.position.set(0.17, 0.63, 0); lid.rotation.z = 0.5; grp.add(lid);   // ajar, leaning on the rim
      const knob = new THREE.Mesh(this.sph(0.035), mat(0x2c2e34, { roughness: 0.5 }));
      knob.position.set(0.155, 0.665, 0); grp.add(knob);
    } else if (t === "pan") {
      const pan = new THREE.Mesh(this.cyl(0.32, 0.3, 0.08, 24), mat(0x33333a, { metalness: 0.6, roughness: 0.4 }));
      pan.position.y = 0.06; pan.castShadow = true; grp.add(pan);
      const handle = new THREE.Mesh(this.box(0.45, 0.05, 0.08), mat(0x222228));
      handle.position.set(0.42, 0.06, 0); grp.add(handle);
    } else if (t === "oven") {
      // boxy APPLIANCE: steel shell, framed front door with a dark window (its
      // inner glow tracks the oven's status via _stationFx), handle bar + knobs
      const front = new THREE.Group(); front.rotation.y = yaw; grp.add(front);
      const o = new THREE.Mesh(this.box(0.82, 0.68, 0.66), mat(0x84898f, { metalness: 0.65, roughness: 0.35 }));
      o.position.y = 0.34; o.castShadow = true; front.add(o);
      const band = new THREE.Mesh(this.box(0.82, 0.1, 0.68), mat(0x3a3d44, { metalness: 0.5, roughness: 0.45 }));
      band.position.y = 0.63; front.add(band);
      for (const kx of [-0.22, 0, 0.22]) {     // control knobs on the top band
        const kn = new THREE.Mesh(this.cyl(0.035, 0.04, 0.03, 12), mat(0x14151a, { roughness: 0.4 }));
        kn.rotation.x = Math.PI / 2; kn.position.set(kx, 0.63, 0.345); front.add(kn);
      }
      const doorFrame = new THREE.Mesh(this.box(0.66, 0.46, 0.05), mat(0x4a4e55, { metalness: 0.6, roughness: 0.35 }));
      doorFrame.position.set(0, 0.3, 0.335); front.add(doorFrame);
      const win = new THREE.Mesh(this.box(0.52, 0.32, 0.03),
        new THREE.MeshStandardMaterial({ color: 0x0d0c10, emissive: 0xff7a1e, emissiveIntensity: 0.18, roughness: 0.25 }));
      win.position.set(0, 0.3, 0.355); front.add(win);
      this._ovenWins = this._ovenWins || {};   // status-driven glow (see _stationFx)
      this._ovenWins[p.x.toFixed(2) + "," + p.z.toFixed(2)] = win.material;
      const bar = new THREE.Mesh(this.cyl(0.022, 0.022, 0.62, 10), mat(0xd6dade, { metalness: 0.8, roughness: 0.3 }));
      bar.rotation.z = Math.PI / 2; bar.position.set(0, 0.56, 0.4); front.add(bar);
      for (const sx of [-0.26, 0.26]) {
        const post = new THREE.Mesh(this.box(0.04, 0.04, 0.07), mat(0x9b9fa5, { metalness: 0.7, roughness: 0.35 }));
        post.position.set(sx, 0.56, 0.37); front.add(post);
      }
    } else if (t === "fryer") {
      // deep fryer: stainless body, dark oil well, wire basket with a long handle
      const body = new THREE.Mesh(this.box(0.72, 0.34, 0.62), mat(0xb9bfc6, { metalness: 0.75, roughness: 0.32 }));
      body.position.y = 0.17; body.castShadow = true; grp.add(body);
      const well = new THREE.Mesh(this.box(0.56, 0.06, 0.46), mat(0x24211c, { roughness: 0.45 }));
      well.position.y = 0.33; grp.add(well);
      const oil = new THREE.Mesh(this.box(0.52, 0.02, 0.42),
        new THREE.MeshStandardMaterial({ color: 0xd9a840, roughness: 0.18, metalness: 0.1,
          transparent: true, opacity: 0.85, emissive: 0x6b4a10, emissiveIntensity: 0.25 }));
      oil.position.y = 0.365; grp.add(oil);
      const wire = mat(0x8f959c, { metalness: 0.8, roughness: 0.35 });
      const bF = new THREE.Mesh(this.box(0.4, 0.14, 0.02), wire); bF.position.set(-0.06, 0.42, 0.16); grp.add(bF);
      const bB = new THREE.Mesh(this.box(0.4, 0.14, 0.02), wire); bB.position.set(-0.06, 0.42, -0.16); grp.add(bB);
      const bL = new THREE.Mesh(this.box(0.02, 0.14, 0.34), wire); bL.position.set(-0.26, 0.42, 0); grp.add(bL);
      const bR = new THREE.Mesh(this.box(0.02, 0.14, 0.34), wire); bR.position.set(0.14, 0.42, 0); grp.add(bR);
      const bBot = new THREE.Mesh(this.box(0.4, 0.02, 0.34), wire); bBot.position.set(-0.06, 0.36, 0); grp.add(bBot);
      // MESH texture: vertical wires along the front + crossbar, so the basket
      // unmistakably reads as a wire fry basket, not a solid tray
      for (let wx = -0.2; wx <= 0.1; wx += 0.06) {
        const v = new THREE.Mesh(this.box(0.012, 0.14, 0.012), wire);
        v.position.set(wx, 0.42, 0.165); grp.add(v);
        const v2 = new THREE.Mesh(this.box(0.012, 0.14, 0.012), wire);
        v2.position.set(wx, 0.42, -0.165); grp.add(v2);
      }
      const cross = new THREE.Mesh(this.box(0.4, 0.014, 0.014), wire);
      cross.position.set(-0.06, 0.46, 0.165); grp.add(cross);
      const handle = new THREE.Mesh(this.box(0.3, 0.03, 0.05), mat(0x22242a, { roughness: 0.6 }));
      handle.position.set(0.32, 0.46, 0); grp.add(handle);
      const panel = new THREE.Mesh(this.box(0.7, 0.1, 0.06), mat(0x3a3f47, { metalness: 0.5, roughness: 0.4 }));
      panel.position.set(0, 0.07, 0.3); grp.add(panel);
      const lamp = new THREE.Mesh(this.sph(0.025), new THREE.MeshStandardMaterial({ color: 0xff9a3c, emissive: 0xff7a1e, emissiveIntensity: 1.4 }));
      lamp.position.set(0.24, 0.07, 0.34); grp.add(lamp);
    } else if (t === "cutting_board") {
      const b = new THREE.Mesh(this.box(0.6, 0.06, 0.42), mat(0xceA45e, { roughness: 0.7 }));
      b.position.y = 0.03; b.castShadow = true; grp.add(b);
      const knife = new THREE.Mesh(this.box(0.04, 0.02, 0.26), mat(0xd6dade, { metalness: 0.8, roughness: 0.3 }));
      knife.position.set(0.16, 0.08, 0.06); knife.rotation.y = 0.3; grp.add(knife);
    } else if (t === "sink") {
      const b = new THREE.Mesh(this.box(0.74, 0.18, 0.6), mat(0xb8bdc2, { metalness: 0.5, roughness: 0.35 }));
      b.position.y = 0.09; b.castShadow = true; grp.add(b);
      const basin = new THREE.Mesh(this.box(0.58, 0.1, 0.44), mat(0x4f555b, { roughness: 0.5 }));
      basin.position.y = 0.14; grp.add(basin);
      const fa = new THREE.Mesh(this.cyl(0.03, 0.03, 0.3, 8), mat(0xcfd4d8, { metalness: 0.8, roughness: 0.3 }));
      fa.position.set(0, 0.28, -0.2); grp.add(fa);
    } else if (t === "serving") {
      // a clear "PASS" hatch: a framed window above a glowing hand-off shelf
      const frameMat = mat(0x3a3f47, { roughness: 0.5, metalness: 0.3 });
      const left = new THREE.Mesh(this.box(0.12, 0.78, 0.5), frameMat); left.position.set(-0.42, 0.39, -0.1); left.castShadow = true; grp.add(left);
      const right = new THREE.Mesh(this.box(0.12, 0.78, 0.5), frameMat); right.position.set(0.42, 0.39, -0.1); right.castShadow = true; grp.add(right);
      const lintel = new THREE.Mesh(this.box(0.96, 0.16, 0.5), frameMat); lintel.position.set(0, 0.86, -0.1); lintel.castShadow = true; grp.add(lintel);
      // dark "kitchen behind" opening
      const opening = new THREE.Mesh(this.box(0.74, 0.6, 0.05), mat(0x14161b, { roughness: 0.9 }));
      opening.position.set(0, 0.5, -0.28); grp.add(opening);
      // glowing hand-off shelf at the front edge — where the plate goes
      const shelf = new THREE.Mesh(this.box(0.84, 0.06, 0.34),
        new THREE.MeshStandardMaterial({ color: 0x8be0b0, emissive: 0x36c07e, emissiveIntensity: 0.9, roughness: 0.4 }));
      shelf.position.set(0, 0.12, 0.16); shelf.castShadow = true; grp.add(shelf);
      // service bell on the shelf
      const bellBase = new THREE.Mesh(this.cyl(0.1, 0.12, 0.03, 16), mat(0xddd6c4)); bellBase.position.set(0.26, 0.17, 0.16); grp.add(bellBase);
      const bell = new THREE.Mesh(this.sph(0.08), mat(0xd9c24a, { metalness: 0.7, roughness: 0.3 })); bell.scale.y = 0.8; bell.position.set(0.26, 0.21, 0.16); grp.add(bell);
      // a downward chevron over the shelf marking "drop here"
      const chev = new THREE.Mesh(this.g("chev", () => new THREE.ConeGeometry(0.12, 0.16, 4)),
        new THREE.MeshStandardMaterial({ color: 0xffe07a, emissive: 0xffc34a, emissiveIntensity: 1.3 }));
      chev.rotation.x = Math.PI; chev.position.set(-0.12, 0.42, 0.12); grp.add(chev);
    } else if (t === "trash") {
      const bin = new THREE.Mesh(this.cyl(0.26, 0.22, 0.5, 16), mat(0x3a3d44, { roughness: 0.6 }));
      bin.position.y = 0.25; bin.castShadow = true; grp.add(bin);
    } else if (t === "plate_source") {
      for (let i = 0; i < 5; i++) {
        const pl = new THREE.Mesh(this.cyl(0.28, 0.28, 0.03, 20), mat(0xf2f4f6, { roughness: 0.3 }));
        pl.position.y = 0.02 + i * 0.035; grp.add(pl);
      }
    } else if (t.endsWith("_source")) {
      const crate = new THREE.Mesh(this.box(0.74, 0.4, 0.74), mat(0x7a5230, { roughness: 0.8 }));
      crate.position.y = 0.2; crate.castShadow = true; grp.add(crate);
      const ing = this._ingredient(t.replace("_source", ""), "raw"); ing.position.y = 0.41; ing.scale.setScalar(0.9); grp.add(ing);  // resting ON the crate, not floating
    }
    grp.traverse(o => { if (o.isMesh) o.castShadow = true; });
    return grp;
  }

  // black glass induction hob with a printed heat-zone ring and control knobs on
  // the aisle side — shared by the pot and pan stations in real mode
  _hob(grp) {
    if (!this._hobMat) {
      this._hobMat = mat(0x1c1e23, { metalness: 0.4, roughness: 0.16 });
      this._hobRingMat = mat(0xb9bec4, { metalness: 0.6, roughness: 0.4 });
      this._knobMat = mat(0x33363b, { metalness: 0.6, roughness: 0.52 });
    }
    const plate = new THREE.Mesh(this.cyl(0.41, 0.42, 0.035, 32), this._hobMat);
    plate.position.y = 0.018; grp.add(plate);
    const ring = new THREE.Mesh(this.g("hobring", () => new THREE.TorusGeometry(0.3, 0.007, 6, 40)), this._hobRingMat);
    ring.rotation.x = Math.PI / 2; ring.position.y = 0.037; grp.add(ring);
    const front = new THREE.Group(); front.rotation.y = this._fixYaw || 0; grp.add(front);
    const panel = new THREE.Mesh(this.box(0.34, 0.07, 0.03), this.steel({ roughness: 0.35 }));
    panel.position.set(0, -0.02, 0.475); front.add(panel);
    for (const sx of [-0.09, 0.09]) {
      const knob = new THREE.Mesh(this.cyl(0.028, 0.032, 0.028, 14), this._knobMat);
      knob.rotation.x = Math.PI / 2; knob.position.set(sx, -0.02, 0.495); front.add(knob);
    }
  }

  // REAL-mode station fixtures: commercial-kitchen equipment in honest materials.
  // Returns `grp` when it handled the station, undefined to fall back to classic.
  _fixtureReal(t, grp) {
    const steel = () => this.steel();
    const darkSteel = mat(0x484d54, { metalness: 0.8, roughness: 0.28 });
    const castIron = mat(0x33363b, { metalness: 0.6, roughness: 0.52 });
    if (t === "pot") {
      // enamel stockpot (photogrammetry model) on an induction hob; the aisle side
      // gets a control knob strip so the "burner" reads as an appliance
      this._hob(grp);
      grp.add(this.model("pot_enamel_01", { targetR: 0.4, y: 0.035, onReady: (h) => {
        h.traverse(o => { if (o.isMesh && /lid/i.test(o.name)) o.visible = false; });  // lid off: contents must read
      } }));
      return grp;
    }
    if (t === "pan") {
      // brass sauté pan (photogrammetry model) on an induction hob — bright metal
      // that reads clearly against the dark hob from the overhead camera
      this._hob(grp);
      const pan = this.model("brass_pan_01", { targetR: 0.52, y: 0.035, centerBowl: true });
      pan.rotation.y = (this._fixYaw || 0) + Math.PI; // bowl on the hob, handle toward the cook
      grp.add(pan);
      return grp;
    }
    if (t === "oven") {
      // stainless convection oven: body, dark glass door with a warm interior glow,
      // full-width pull bar, control knobs — door faces the aisle
      grp.rotation.y = this._fixYaw || 0;
      const body = new THREE.Mesh(this.box(0.86, 0.62, 0.74), steel());
      body.position.y = 0.31; grp.add(body);
      const glass = new THREE.Mesh(this.box(0.62, 0.4, 0.02),
        new THREE.MeshStandardMaterial({ color: 0x0c0d10, metalness: 0.3, roughness: 0.08,
                                         emissive: 0xff7a1e, emissiveIntensity: 0.55 }));
      glass.position.set(0, 0.28, 0.375); grp.add(glass);
      const frame = new THREE.Mesh(this.box(0.7, 0.48, 0.015), darkSteel);
      frame.position.set(0, 0.28, 0.368); grp.add(frame);
      const bar = new THREE.Mesh(this.g("ovbar", () => new THREE.CylinderGeometry(0.022, 0.022, 0.66, 12)), steel());
      bar.rotation.z = Math.PI / 2; bar.position.set(0, 0.56, 0.42); grp.add(bar);
      for (let i = 0; i < 3; i++) {
        const knob = new THREE.Mesh(this.cyl(0.032, 0.036, 0.03, 14), castIron);
        knob.rotation.x = Math.PI / 2; knob.position.set(-0.22 + i * 0.22, 0.66, 0.36); grp.add(knob);
      }
      return grp;
    }
    if (t === "cutting_board") {
      // thick end-grain butcher block + a real kitchen knife (photogrammetry model)
      const b = new THREE.Mesh(this.box(0.64, 0.09, 0.46),
        this.pbr("wood", { repeat: 0.8, roughness: 0.6 }));
      b.position.y = 0.045; grp.add(b);
      // years of knife scars — a spotless board reads as a 3D render
      const scarM = mat(0x74522e, { roughness: 0.85 });
      for (let i = 0; i < 7; i++) {
        const scar = new THREE.Mesh(this.box(0.2 + (i % 3) * 0.09, 0.0015, 0.005), scarM);
        scar.position.set(-0.08 + ((i * 17) % 23) / 90, 0.0915, -0.14 + i * 0.045);
        scar.rotation.y = (i - 3) * 0.31; grp.add(scar);
      }
      // juice groove routed around the edge + a dark work-stain in the middle
      const grooveM = mat(0x5c4023, { roughness: 0.9 });
      for (const [gw, gd, gx, gz] of [[0.5, 0.012, 0, 0.17], [0.5, 0.012, 0, -0.17], [0.012, 0.35, 0.25, 0], [0.012, 0.35, -0.25, 0]]) {
        const gr2 = new THREE.Mesh(this.box(gw, 0.004, gd), grooveM);
        gr2.position.set(gx, 0.0905, gz); grp.add(gr2);
      }
      const stain = new THREE.Mesh(this.g("bstain", () => new THREE.CircleGeometry(0.13, 16)),
        new THREE.MeshStandardMaterial({ color: 0x6b4a26, transparent: true, opacity: 0.4, roughness: 0.75, depthWrite: false }));
      stain.rotation.x = -Math.PI / 2; stain.position.set(-0.06, 0.0912, 0.03);
      stain.scale.set(1.3, 0.8, 1); grp.add(stain);
      const knife = this.model("fish_knife", { targetR: 0.2, y: 0.09, rot: [-Math.PI / 2, 0, 0] });
      knife.position.set(0.12, 0, 0.1); knife.rotation.y = (this._fixYaw || 0) + 0.4;
      grp.add(knife);
      return grp;
    }
    if (t === "sink") {
      // deep drop-in stainless basin: visible interior walls + drain, a solid
      // gooseneck faucet with cross-handle valves (the old one read paper-thin)
      const deck = new THREE.Mesh(this.box(0.8, 0.07, 0.66), steel());
      deck.position.y = 0.035; grp.add(deck);
      const basinIn = new THREE.Mesh(this.box(0.56, 0.02, 0.42), mat(0x62676d, { metalness: 0.9, roughness: 0.35 }));
      basinIn.position.y = -0.04; grp.add(basinIn);
      for (const [wx, wz, dx, dz] of [[0.02, 0.42, -0.28, 0], [0.02, 0.42, 0.28, 0], [0.56, 0.02, 0, -0.2], [0.56, 0.02, 0, 0.2]]) {
        const wallM = new THREE.Mesh(this.box(wx, 0.12, wz), mat(0x585d63, { metalness: 0.88, roughness: 0.35 }));
        wallM.position.set(dx, 0.02, dz); grp.add(wallM);
      }
      const drain = new THREE.Mesh(this.cyl(0.045, 0.045, 0.012, 16), mat(0x3c4046, { metalness: 0.9, roughness: 0.35 }));
      drain.position.set(0, -0.028, 0.05); grp.add(drain);
      const riser = new THREE.Mesh(this.g("faucr2", () => new THREE.CylinderGeometry(0.034, 0.04, 0.36, 14)), steel());
      riser.position.set(0, 0.25, -0.26); grp.add(riser);
      const neck = new THREE.Mesh(this.g("faucn2", () => new THREE.TorusGeometry(0.13, 0.028, 12, 22, Math.PI)), steel());
      neck.position.set(0, 0.43, -0.13); neck.rotation.y = Math.PI / 2; grp.add(neck);
      const spout = new THREE.Mesh(this.g("faucs2", () => new THREE.CylinderGeometry(0.022, 0.022, 0.1, 12)), steel());
      spout.position.set(0, 0.39, 0.0); grp.add(spout);
      for (const s of [-1, 1]) {                        // hot/cold cross valves
        const stem = new THREE.Mesh(this.g("valst", () => new THREE.CylinderGeometry(0.018, 0.02, 0.07, 10)), steel());
        stem.position.set(s * 0.14, 0.1, -0.26); grp.add(stem);
        const bar1 = new THREE.Mesh(this.box(0.09, 0.016, 0.016), steel());
        bar1.position.set(s * 0.14, 0.14, -0.26); grp.add(bar1);
        const bar2 = new THREE.Mesh(this.box(0.016, 0.016, 0.09), steel());
        bar2.position.set(s * 0.14, 0.14, -0.26); grp.add(bar2);
      }
      return grp;
    }
    if (t === "serving") {
      // stainless pass window with red-glowing heat lamps over the shelf — the
      // universal "send it" signal in a restaurant kitchen. The whole hatch turns
      // to face the aisle (it used to show its side to the cook on east/west walls).
      grp.rotation.y = this._fixYaw || 0;
      const frameM = steel();
      const left = new THREE.Mesh(this.box(0.12, 0.78, 0.5), frameM); left.position.set(-0.42, 0.39, -0.1); grp.add(left);
      const right = new THREE.Mesh(this.box(0.12, 0.78, 0.5), frameM); right.position.set(0.42, 0.39, -0.1); grp.add(right);
      const lintel = new THREE.Mesh(this.box(0.96, 0.16, 0.5), frameM); lintel.position.set(0, 0.86, -0.1); grp.add(lintel);
      // the hatch is OPEN — you look straight through it into the dining room
      const shelf = new THREE.Mesh(this.box(0.84, 0.05, 0.36), this.steel({ roughness: 0.36 }));
      shelf.position.set(0, 0.12, 0.14); grp.add(shelf);
      for (const sx of [-0.2, 0.2]) {                    // heat lamps under the lintel
        const cone = new THREE.Mesh(this.g("hlamp", () => new THREE.ConeGeometry(0.085, 0.12, 16, 1, true)), darkSteel);
        cone.position.set(sx, 0.74, 0.1); grp.add(cone);
        const bulb = new THREE.Mesh(this.sph(0.045),
          new THREE.MeshStandardMaterial({ color: 0xff5a3c, emissive: 0xff3a18, emissiveIntensity: 2.4 }));
        bulb.position.set(sx, 0.7, 0.1); grp.add(bulb);
      }
      // order-ticket rail clipped to the lintel, tickets tucked under the bar
      const railBar = new THREE.Mesh(this.g("tkrail", () => new THREE.CylinderGeometry(0.02, 0.02, 0.9, 10)), this.steel({ roughness: 0.3 }));
      railBar.rotation.z = Math.PI / 2; railBar.position.set(0, 0.795, 0.17); grp.add(railBar);
      for (let i = 0; i < 3; i++) {
        const tk = new THREE.Mesh(this.g("tkpp", () => new THREE.PlaneGeometry(0.105, 0.145)),
          new THREE.MeshStandardMaterial({ map: this._ticketTex(i), roughness: 0.9, side: THREE.DoubleSide }));
        tk.position.set(-0.24 + i * 0.24, 0.735, 0.174);
        tk.rotation.y = (i - 1) * 0.07; tk.rotation.x = -0.1 + i * 0.05; grp.add(tk);
      }
      // warm heat-lamp pool on the shelf — the lamps visibly DO something
      const pool = new THREE.Mesh(this.g("hlpool", () => new THREE.CircleGeometry(0.3, 20)),
        new THREE.MeshBasicMaterial({ color: 0xff9a5c, transparent: true, opacity: 0.16, depthWrite: false }));
      pool.rotation.x = -Math.PI / 2; pool.position.set(0, 0.148, 0.12); pool.scale.set(1.4, 0.7, 1);
      grp.add(pool);
      // counter service bell: dome on a base with a plunger, not a floating orb
      const bellBase = new THREE.Mesh(this.cyl(0.055, 0.06, 0.016, 16), mat(0x2c2e32, { metalness: 0.6, roughness: 0.4 }));
      bellBase.position.set(0.3, 0.155, 0.16); grp.add(bellBase);
      const bell = new THREE.Mesh(this.sph(0.048), mat(0xd8d9db, { metalness: 0.95, roughness: 0.12 }));
      bell.scale.y = 0.72; bell.position.set(0.3, 0.175, 0.16); grp.add(bell);
      const plunger = new THREE.Mesh(this.cyl(0.011, 0.011, 0.025, 8), mat(0x2c2e32, { metalness: 0.6, roughness: 0.4 }));
      plunger.position.set(0.3, 0.215, 0.16); grp.add(plunger);
      return grp;
    }
    if (t === "trash") {
      // full-height pedal bin standing on the FLOOR (a table-top wastebasket on
      // a cabinet read as a prop mistake)
      grp.position.y = 0;
      const bin = new THREE.Mesh(this.cyl(0.3, 0.25, 0.82, 22), this.steel({ roughness: 0.4 }));
      bin.position.y = 0.41; grp.add(bin);
      const lid = new THREE.Mesh(this.cyl(0.31, 0.31, 0.045, 22), darkSteel);
      lid.position.y = 0.845; grp.add(lid);
      const knob = new THREE.Mesh(this.sph(0.035, 10), darkSteel);
      knob.position.y = 0.885; grp.add(knob);
      const pedal = new THREE.Mesh(this.box(0.12, 0.035, 0.16), darkSteel);
      pedal.position.set(0, 0.02, 0.36); grp.add(pedal);
      grp.add(this._contactShadow(0, 0.006, 0, 0.42, 0.3));
      return grp;
    }
    if (t === "plate_source") {
      // porcelain stack with real plate profiles: flared body + rolled rim, jitter
      const porc = mat(0xf5f6f7, { roughness: 0.16, metalness: 0.02 });
      const gapM = mat(0x777c84, { roughness: 0.6 });
      for (let i = 0; i < 6; i++) {
        const jx = Math.sin(i * 2.1) * 0.016, jz = Math.cos(i * 1.7) * 0.016, py = 0.02 + i * 0.042;
        const one = new THREE.Group();
        const pl = new THREE.Mesh(this.cyl(0.25, 0.19, 0.03, 26), porc);
        one.add(pl);
        const rim = new THREE.Mesh(this.g("plrim", () => new THREE.TorusGeometry(0.245, 0.015, 8, 26)), porc);
        rim.rotation.x = Math.PI / 2; rim.position.y = 0.018; one.add(rim);
        // dark seam under each plate so the stack reads as separate dishes
        const gap = new THREE.Mesh(this.cyl(0.235, 0.235, 0.008, 24), gapM);
        gap.position.y = -0.016; one.add(gap);
        one.position.set(jx, py, jz);
        one.rotation.z = Math.sin(i * 3.7) * 0.02;   // hand-stacked, not machined
        one.rotation.y = i * 0.9;
        grp.add(one);
      }
      return grp;
    }
    if (t.endsWith("_source")) {
      // stainless gastronorm ingredient well set into a steel chiller body — how
      // ingredients actually live on a prep line. The well is stocked FULL (a
      // colour mass fills the pan) so it reads from the overhead camera, and the
      // label chip faces the aisle (duplicated on the back for good measure).
      const ingName = t.replace("_source", "");
      const ingKey = ING[ingName] || { c: 0xcccccc, c2: 0x999999 };
      const bodyB = new THREE.Mesh(this.box(0.78, 0.3, 0.78), this.steel());
      bodyB.position.y = 0.15; grp.add(bodyB);
      // a real recessed gastronorm pan: raised rim FRAME around a sunken food
      // mass, so from eye level you look INTO a bin instead of at a closed box
      const rimM = mat(0xcdd2d7, { metalness: 0.92, roughness: 0.2 });
      for (const [rw, rd, rx, rz] of [[0.7, 0.06, 0, 0.32], [0.7, 0.06, 0, -0.32], [0.06, 0.7, 0.32, 0], [0.06, 0.7, -0.32, 0]]) {
        const bar = new THREE.Mesh(this.box(rw, 0.09, rd), rimM);
        bar.position.set(rx, 0.345, rz); grp.add(bar);
      }
      const wellIn = new THREE.Mesh(this.box(0.6, 0.02, 0.6), mat(0x707579, { metalness: 0.85, roughness: 0.35 }));
      wellIn.position.y = 0.3; grp.add(wellIn);
      // the bin is stocked with a PILE of the actual product — irregular, over-
      // lapping pieces (a flat colour slab read as painted plastic)
      const spots = [[-0.16, -0.15, 0.68], [0.13, -0.16, 0.62], [-0.14, 0.14, 0.6],
                     [0.15, 0.13, 0.66], [0.0, -0.02, 0.8], [-0.02, 0.19, 0.5], [0.19, -0.01, 0.52]];
      spots.forEach(([sx, sz, sc], i) => {
        const piece = this._ingredient(ingName, "raw");
        piece.position.set(sx, 0.312 + (i % 3) * 0.012, sz);
        piece.scale.setScalar(sc);
        piece.rotation.y = i * 1.7;
        piece.rotation.z = (i % 2 ? 1 : -1) * 0.07;
        // natural produce variation: no two pieces the exact same shade/sheen
        const k2 = 0.9 + ((i * 37) % 5) * 0.045;
        piece.traverse(o => {
          if (!o.isMesh) return;
          // the food library shares cached materials between instances: clone before the
          // per-piece tweak, or every crate darkens every other one (2026-09-09)
          const tweak = (m) => { if (!m || !m.color) return m; const c = m.clone(); c.color.multiplyScalar(k2); c.roughness = Math.max(0.3, (c.roughness ?? 0.55) - 0.12); return c; };
          o.material = Array.isArray(o.material) ? o.material.map(tweak) : tweak(o.material);
        });
        grp.add(piece);
      });
      const chipFor = (yaw) => {
        const cg = new THREE.Group(); cg.rotation.y = yaw;
        const chip = new THREE.Mesh(this.g("srclbl", () => new THREE.PlaneGeometry(0.3, 0.11)),
          new THREE.MeshStandardMaterial({ map: this._sourceLabelTex(ingName, ingKey.c), roughness: 0.55 }));
        chip.position.set(0, 0.15, 0.398); cg.add(chip);
        return cg;
      };
      grp.add(chipFor(this._fixYaw || 0));
      grp.add(chipFor((this._fixYaw || 0) + Math.PI));
      return grp;
    }
    return undefined;
  }

  // ---- REAL-mode restaurant surround ------------------------------------
  // The kitchen block used to float on a bare 50x50 slab ("island in a void").
  // This builds the actual restaurant around it: a dining room beyond the pass
  // (wood floor, tables, seated guests, pendant lights, chalk menu), back-of-
  // house walls with stocked shelving / fridge / swing door behind the line, a
  // ventilation hood over the hobs, rubber mats, ticket rail and an "ORDER UP"
  // neon that lights when a dish is served. Everything lives in staticGroup so
  // it rebuilds with the level and never touches the classic path.
  _buildSurround(st) {
    const w = st.width, h = st.height;
    const A0 = { x: (w - 1) / 2 + 0.5, z: (h - 1) / 2 + 0.5 };   // kitchen half extents
    // dining direction = the board edge that holds the serving pass
    let dir = [1, 0];
    outer: for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
      if (st.terrain[y][x] === "serving") {
        if (x === w - 1) dir = [1, 0]; else if (x === 0) dir = [-1, 0];
        else if (y === 0) dir = [0, -1]; else dir = [0, 1];
        break outer;
      }
    const yawFor = { "1,0": 0, "0,1": -Math.PI / 2, "-1,0": Math.PI, "0,-1": Math.PI / 2 };
    const R = new THREE.Group();                       // local +x = toward dining
    R.rotation.y = yawFor[dir[0] + "," + dir[1]];
    this.staticGroup.add(R);
    const A = dir[0] !== 0 ? A0.x : A0.z;              // half extent along dining axis
    const B = dir[0] !== 0 ? A0.z : A0.x;              // half extent across
    const BK = A + 5.5, SD = B + 5.5, DIN = A + 11;    // back / side / dining wall dists
    const FLOOR = -0.3, WALLH = 4.25;

    // --- materials (all cached/shared) ---
    const plaster = this.pbr("plaster", { repeat: 4, roughness: 0.93, color: 0xddd6c9 });
    const paintDin = this.pbr("plaster", { repeat: 4, roughness: 0.9, color: 0xd8dec9 });  // sage dining paint
    const woodFloor = this.pbr("wood", { repeat: 7, roughness: 0.74, color: 0xc9ab82 });
    const woodDark = this.pbr("wood", { repeat: 1.4, roughness: 0.5, color: 0x7c5a3a });
    const blackIron = mat(0x24262a, { roughness: 0.55, metalness: 0.6 });
    const steelM = this.steel({ roughness: 0.4 });

    // --- floors: terrazzo back-of-house, oiled oak planks in the dining room ---
    const kzW = BK + A + 0.4, roomZ = 2 * SD;
    const terr = new THREE.Mesh(new THREE.PlaneGeometry(kzW, roomZ),
      this.pbr("stone", { repeat: 20, roughness: 0.82, color: 0xa3a29c, noRoughMap: true, normScale: 1.3 }));
    terr.rotation.x = -Math.PI / 2; terr.position.set((-BK + A + 0.2) / 2, FLOOR, 0);
    terr.receiveShadow = true; R.add(terr);
    const wf = new THREE.Mesh(new THREE.PlaneGeometry(DIN - A - 0.2, roomZ), woodFloor);
    wf.rotation.x = -Math.PI / 2; wf.position.set((A + 0.2 + DIN) / 2, FLOOR + 0.004, 0);
    wf.receiveShadow = true; R.add(wf);
    // dark concrete apron outside the building (reads as "beyond the walls")
    const apron = new THREE.Mesh(new THREE.PlaneGeometry(70, 70), mat(0x191a1c, { roughness: 0.97 }));
    apron.rotation.x = -Math.PI / 2; apron.position.y = FLOOR - 0.02; R.add(apron);

    // --- room shell: thick walls with a visible top edge from overhead ---
    const wall = (len, m) => new THREE.Mesh(new THREE.BoxGeometry(len, WALLH, 0.24), m);
    const wb = wall(roomZ + 0.5, plaster); wb.rotation.y = Math.PI / 2;
    wb.position.set(-BK, FLOOR + WALLH / 2, 0); wb.receiveShadow = true; R.add(wb);
    const wd = wall(roomZ + 0.5, paintDin); wd.rotation.y = Math.PI / 2;
    wd.position.set(DIN, FLOOR + WALLH / 2, 0); wd.receiveShadow = true; R.add(wd);
    for (const s of [-1, 1]) {
      const ws = wall(DIN + BK + 0.5, plaster);
      ws.position.set((DIN - BK) / 2, FLOOR + WALLH / 2, s * SD);
      ws.receiveShadow = true; R.add(ws);
    }
    // white hygiene-tile band around the back-of-house walls (the universal
    // commercial-kitchen wall finish) + an electrical conduit run above it
    // built from ~2-unit segments so the square tile texture is never stretched
    // (one long box face maps UV 0..1 across the whole wall = plank artefacts).
    // The band is procedural white subway tile — the PBR "tile" asset is a beige
    // mosaic that read as brown wicker on walls.
    if (!this._subwayM) {
      const C = document.createElement("canvas"); C.width = 256; C.height = 128;
      const g2 = C.getContext("2d");
      g2.fillStyle = "#cfcdc6"; g2.fillRect(0, 0, 256, 128);      // grout
      for (let row = 0; row < 4; row++) {
        const off = (row % 2) * 32;
        for (let col = -1; col < 5; col++) {
          const x = col * 64 + off, y = row * 32;
          const tint = 240 + ((row * 7 + col * 13) % 3) * 4;
          g2.fillStyle = `rgb(${tint},${tint},${tint - 4})`;
          g2.fillRect(x + 2, y + 2, 60, 28);
          g2.fillStyle = "rgba(255,255,255,.35)";                 // glaze highlight
          g2.fillRect(x + 4, y + 4, 56, 5);
        }
      }
      const t = new THREE.CanvasTexture(C);
      t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(2, 4);
      t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
      this._subwayM = new THREE.MeshStandardMaterial({ map: t, roughness: 0.3 });
    }
    const hygM = this._subwayM;
    const condM = mat(0x9b9fa3, { metalness: 0.7, roughness: 0.45 });
    const bandH = 1.9, bandY = FLOOR + 0.45 + bandH / 2, SEG = 2.0;
    const bandRun = (x0, z0, dx, dz, len) => {
      for (let d = 0; d < len; d += SEG) {
        const w = Math.min(SEG, len - d);
        const b = new THREE.Mesh(new THREE.BoxGeometry(Math.abs(dx) ? w : 0.06, bandH, Math.abs(dz) ? w : 0.06), hygM);
        b.position.set(x0 + dx * (d + w / 2), bandY, z0 + dz * (d + w / 2));
        b.receiveShadow = true; R.add(b);
      }
    };
    bandRun(-BK + 0.15, -(roomZ / 2), 0, 1, roomZ);                       // back wall
    for (const s of [-1, 1]) {
      bandRun(-BK + 0.2, s * (SD - 0.15), 1, 0, A + BK + 1.8);            // side walls
      const cd = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, A + BK + 1.4, 8), condM);
      cd.rotation.z = Math.PI / 2; cd.position.set((A + 1.8 - BK) / 2 - 0.8, FLOOR + 2.75, s * (SD - 0.14)); R.add(cd);
    }
    const cdb = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, roomZ - 0.3, 8), condM);
    cdb.rotation.x = Math.PI / 2; cdb.position.set(-BK + 0.14, FLOOR + 2.75, 0); R.add(cdb);
    const panel = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.62, 0.46), mat(0x84898e, { metalness: 0.6, roughness: 0.5 }));
    panel.position.set(-BK + 0.17, FLOOR + 2.2, -2.6); R.add(panel);
    // galvanised HVAC trunk ducts running along the wall tops — the dark space
    // above the tile band reads as serviced ceiling void, not empty render
    const ductM = new THREE.MeshStandardMaterial({ color: 0x878c90, metalness: 0.7, roughness: 0.5 });
    ductM.envMapIntensity = 0.5;
    const d1 = new THREE.Mesh(new THREE.BoxGeometry(roomZ - 0.6, 0.5, 0.6), ductM);
    d1.rotation.y = Math.PI / 2; d1.position.set(-BK + 0.45, FLOOR + 3.55, 0); R.add(d1);
    const d2 = new THREE.Mesh(new THREE.BoxGeometry(A + BK + 1.4, 0.46, 0.55), ductM);
    d2.position.set((A + 1.6 - BK) / 2 - 0.8, FLOOR + 3.5, -(SD - 0.45)); R.add(d2);
    for (let i = 0; i < 3; i++) {   // hanger straps
      const strap = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.75, 0.04), mat(0x5c6064, { metalness: 0.6, roughness: 0.5 }));
      strap.position.set(-BK + 2.2 + i * 3.1, FLOOR + 3.85, -(SD - 0.45)); R.add(strap);
    }

    // --- dining side: wainscot, chalk menu, framed prints, neon "ORDER UP" ---
    const wainZ = new THREE.Mesh(new THREE.BoxGeometry(roomZ - 0.2, 0.95, 0.07), woodDark);
    wainZ.rotation.y = Math.PI / 2; wainZ.position.set(DIN - 0.15, FLOOR + 0.475, 0); R.add(wainZ);
    for (const s of [-1, 1]) {   // side-wall wainscot only inside the dining zone
      const wn = new THREE.Mesh(new THREE.BoxGeometry(DIN - A - 3.3, 0.95, 0.07), woodDark);
      wn.position.set((A + 3.0 + DIN - 0.3) / 2, FLOOR + 0.475, s * (SD - 0.15)); R.add(wn);
    }
    const menu = new THREE.Mesh(this.g("menubd", () => new THREE.PlaneGeometry(2.6, 1.55)),
      new THREE.MeshStandardMaterial({ map: this._menuTex(), roughness: 0.85 }));
    menu.rotation.y = -Math.PI / 2; menu.position.set(DIN - 0.14, FLOOR + 2.05, -2.2); R.add(menu);
    const menuFrM = mat(0x5a4026, { roughness: 0.6 });
    for (const [fw, fh, fx, fy] of [[2.75, 0.08, 0, 0.815], [2.75, 0.08, 0, -0.815], [0.08, 1.71, 1.37, 0], [0.08, 1.71, -1.37, 0]]) {
      const fr = new THREE.Mesh(new THREE.BoxGeometry(0.05, fh, fw), menuFrM);
      fr.position.set(DIN - 0.13, FLOOR + 2.05 + fy, -2.2 - fx); R.add(fr);
    }
    for (const [pz, pc] of [[2.0, 0x9c8468], [3.4, 0x707a6c]]) {   // muted framed prints
      const fr = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.78, 0.58), mat(0x2c2620, { roughness: 0.5 }));
      fr.position.set(DIN - 0.14, FLOOR + 2.0, pz); R.add(fr);
      const cv = new THREE.Mesh(new THREE.BoxGeometry(0.052, 0.66, 0.46), mat(pc, { roughness: 0.9 }));
      cv.position.set(DIN - 0.145, FLOOR + 2.0, pz); R.add(cv);
    }
    // neon sign on the dining wall facing the kitchen — flashes on every delivery
    this._neonMat = new THREE.MeshStandardMaterial({ map: this._neonTex(), transparent: true,
      emissive: 0xffffff, emissiveIntensity: 0.55, emissiveMap: this._neonTex(), color: 0x111111 });
    // soft red bounce halo painted on the wall behind the tubes
    if (!this._neonHaloTex) {
      const C = document.createElement("canvas"); C.width = 128; C.height = 64;
      const g2 = C.getContext("2d");
      const gr = g2.createRadialGradient(64, 32, 4, 64, 32, 62);
      gr.addColorStop(0, "rgba(255,74,48,.55)"); gr.addColorStop(1, "rgba(255,74,48,0)");
      g2.fillStyle = gr; g2.scale(1, 0.5); g2.fillRect(0, 0, 128, 128);
      this._neonHaloTex = new THREE.CanvasTexture(C);
    }
    const halo = new THREE.Mesh(this.g("neonhalo", () => new THREE.PlaneGeometry(2.9, 1.1)),
      new THREE.MeshBasicMaterial({ map: this._neonHaloTex, transparent: true, opacity: 0.6, depthWrite: false }));
    halo.rotation.y = -Math.PI / 2; halo.position.set(DIN - 0.135, FLOOR + 2.72, 0.6); R.add(halo);
    const neon = new THREE.Mesh(this.g("neonpl", () => new THREE.PlaneGeometry(1.9, 0.5)), this._neonMat);
    neon.rotation.y = -Math.PI / 2; neon.position.set(DIN - 0.13, FLOOR + 2.72, 0.6); R.add(neon);
    this._neonFlash = 0;

    // --- dining furniture: pedestal tables, bistro chairs, guests, pendants ---
    const seatCfg = [[A + 4.2, -3.6, 3, 0.5], [A + 4.4, 0.3, 2, 2.2], [A + 4.2, 3.9, 3, 4.1],
                     [A + 7.9, -4.3, 2, 1.2], [A + 8.1, 0.0, 3, 3.3], [A + 7.9, 4.2, 2, 5.4]];
    let pi = 0;
    for (const [tx, tz, nGuests, a0] of seatCfg) {
      R.add(this._dinTable(tx, FLOOR, tz, nGuests));
      R.add(this._contactShadow(tx, FLOOR, tz, 1.15, 0.34));
      for (let k = 0; k < 4; k++) {
        const a = a0 + k * (Math.PI / 2) + Math.sin(tx * 7 + k * 3) * 0.22;
        const cx = tx + Math.cos(a) * 0.86, cz = tz + Math.sin(a) * 0.86;
        R.add(this._dinChair(cx, FLOOR, cz, a));
        R.add(this._contactShadow(cx, FLOOR, cz, 0.32, 0.22));
        if (k < nGuests) R.add(this._patron(tx + Math.cos(a) * 0.78, FLOOR, tz + Math.sin(a) * 0.78, a, pi++));
      }
      R.add(this._pendant(tx, FLOOR, tz));
    }
    // evening windows on the dining side walls: dark glass, thin mullions, a
    // hint of cool exterior glow — kills the big bare plaster planes
    const winFrM = mat(0x2e2a24, { roughness: 0.6 });
    const winGlM = new THREE.MeshStandardMaterial({ color: 0x131c26, roughness: 0.06, metalness: 0.55,
      emissive: 0x1d2a3a, emissiveIntensity: 0.5 });
    for (const s of [-1, 1]) {
      for (const wx of [A + 4.6, A + 8.2]) {
        const wg = new THREE.Group(); wg.position.set(wx, FLOOR + 2.15, s * (SD - 0.13));
        const fr = new THREE.Mesh(this.g("winfr", () => new THREE.BoxGeometry(1.7, 1.35, 0.08)), winFrM);
        wg.add(fr);
        const pane = new THREE.Mesh(this.g("winpn", () => new THREE.BoxGeometry(1.54, 1.19, 0.05)), winGlM);
        pane.position.z = -s * 0.02; wg.add(pane);
        const mulV = new THREE.Mesh(this.g("winmv", () => new THREE.BoxGeometry(0.05, 1.19, 0.09)), winFrM);
        wg.add(mulV);
        const mulH = new THREE.Mesh(this.g("winmh", () => new THREE.BoxGeometry(1.54, 0.05, 0.09)), winFrM);
        mulH.position.y = 0.18; wg.add(mulH);
        R.add(wg);
      }
    }
    // waiter's sideboard: plates, cutlery tray, water glasses at the ready
    const sb = new THREE.Group(); sb.position.set(A + 2.6, FLOOR, -(SD - 0.62));
    const sbBody = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.92, 0.52), this.pbr("wood", { repeat: 1.1, roughness: 0.5, color: 0x67482c }));
    sbBody.position.y = 0.46; sbBody.castShadow = true; sb.add(sbBody);
    const sbPorc = mat(0xf1f2f0, { roughness: 0.3 });
    for (let i = 0; i < 4; i++) {
      const pl = new THREE.Mesh(this.g("sbpl", () => new THREE.CylinderGeometry(0.16, 0.13, 0.024, 18)), sbPorc);
      pl.position.set(-0.45, 0.945 + i * 0.026, 0); sb.add(pl);
    }
    const ctray = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.06, 0.3), mat(0x3a3d42, { roughness: 0.7 }));
    ctray.position.set(0.12, 0.955, 0); sb.add(ctray);
    for (let i = 0; i < 3; i++) {
      const gl = new THREE.Mesh(this.g("sbgl", () => new THREE.CylinderGeometry(0.04, 0.034, 0.13, 10)), mat(0xdfe4e8, { roughness: 0.1, metalness: 0.15 }));
      gl.position.set(0.52, 0.99, -0.12 + i * 0.12); sb.add(gl);
    }
    const napk = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.09, 0.2), mat(0xf3f1e9, { roughness: 0.95 }));
    napk.position.set(-0.12, 0.97, 0.02); napk.rotation.y = 0.2; sb.add(napk);
    R.add(sb);
    R.add(this._contactShadow(A + 2.6, FLOOR, -(SD - 0.62), 0.8, 0.3));
    const dinLight = new THREE.PointLight(0xffd7a8, 16, 17, 2);
    dinLight.position.set(A + 6, FLOOR + 3.2, 0); R.add(dinLight);
    // warm "front of house" wash so the dining wall and the guests' kitchen-facing
    // sides aren't lit only by bounce (they rendered near-black otherwise)
    const wash = new THREE.DirectionalLight(0xffe6c6, 0.75);
    wash.position.set(-2, 8.5, 0); wash.target.position.set(12, 0, 0);
    R.add(wash); R.add(wash.target);
    // dark ceiling over the dining zone only: pendants terminate on it, the wide
    // and pass views stop showing raw sky, and the follow camera (which looks
    // down over the OPEN kitchen zone) never crosses it
    const ceil = new THREE.Mesh(new THREE.PlaneGeometry(DIN - A - 2.6, roomZ), mat(0x2c2a27, { roughness: 0.95 }));
    ceil.rotation.x = Math.PI / 2; ceil.position.set((A + 2.8 + DIN) / 2, FLOOR + 4.18, 0);
    R.add(ceil);
    const soffit = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.62, roomZ), mat(0x35322e, { roughness: 0.9 }));
    soffit.position.set(A + 2.85, FLOOR + 3.99, 0); R.add(soffit);   // top overlaps the slab: no lit sliver at the junction
    for (const [tx, tz] of [[A + 4.2, -3.6], [A + 4.4, 0.3], [A + 4.2, 3.9], [A + 7.9, -4.3], [A + 8.1, 0.0], [A + 7.9, 4.2]]) {
      const can = new THREE.Mesh(this.g("ceilcan", () => new THREE.CylinderGeometry(0.09, 0.09, 0.02, 14)),
        new THREE.MeshStandardMaterial({ color: 0xfff0d6, emissive: 0xffdca8, emissiveIntensity: 1.1 }));
      can.position.set(tx + 0.9, FLOOR + 4.16, tz); R.add(can);
    }

    // --- back-of-house dressing ---
    // stocked wall shelves + reach-in fridge on the wall behind the hob line
    R.add(this._wallShelf(-1.2, FLOOR, -SD + 0.16, 3.6, 0));
    R.add(this._fridge(-BK + 0.62, FLOOR, -SD + 2.0, Math.PI / 2));
    R.add(this._contactShadow(-BK + 0.62, FLOOR, -SD + 2.0, 0.85, 0.35));
    R.add(this._crateStack(-BK + 0.55, FLOOR, SD - 1.4));
    R.add(this._contactShadow(-BK + 0.55, FLOOR, SD - 1.4, 0.55, 0.32));
    R.add(this._crateStack(2.2, FLOOR, SD - 0.55, 0.8));
    R.add(this._contactShadow(2.2, FLOOR, SD - 0.55, 0.55, 0.32));
    R.add(this._wallShelf(0.8, FLOOR, SD - 0.16, 2.8, Math.PI));
    R.add(this._swingDoor(-BK + 0.13, FLOOR, 1.8));
    const clock = new THREE.Mesh(this.g("clockf", () => new THREE.CylinderGeometry(0.26, 0.26, 0.05, 24)),
      [blackIron, new THREE.MeshStandardMaterial({ map: this._clockTex(), roughness: 0.6 }), blackIron]);
    clock.rotation.x = Math.PI / 2; clock.position.set(2.6, FLOOR + 2.5, -SD + 0.15); R.add(clock);

    // potted plants soften the dining corners
    for (const s of [-1, 1]) {
      const pg = new THREE.Group(); pg.position.set(DIN - 1.0, FLOOR, s * (SD - 1.0));
      const pot = new THREE.Mesh(this.g("plpot", () => new THREE.CylinderGeometry(0.22, 0.17, 0.34, 14)), mat(0x9c5b3c, { roughness: 0.85 }));
      pot.position.y = 0.17; pg.add(pot);
      const trunk = new THREE.Mesh(this.g("pltr", () => new THREE.CylinderGeometry(0.03, 0.04, 0.5, 8)), mat(0x5a4530, { roughness: 0.9 }));
      trunk.position.y = 0.55; pg.add(trunk);
      for (let i = 0; i < 4; i++) {
        const fol = new THREE.Mesh(this.sph(0.24, 10), mat(i % 2 ? 0x3e5c33 : 0x4a6c3c, { roughness: 0.9 }));
        fol.position.set(Math.cos(i * 2.2) * 0.14, 0.92 + (i % 2) * 0.18, Math.sin(i * 2.2) * 0.14);
        fol.scale.y = 0.85; pg.add(fol);
      }
      pg.traverse(o => { if (o.isMesh) o.castShadow = false; });
      R.add(pg);
      R.add(this._contactShadow(DIN - 1.0, FLOOR, s * (SD - 1.0), 0.4, 0.3));
    }
    // A waiter used to stand at the dining side of the pass. REMOVED 2026-09-09: a standing,
    // upright human figure a metre from the serving tile reads as a SECOND COOK — the
    // participant asked what it was, and the assistant sees the same pixels and can mistake it
    // for the cook it is supposed to be watching. Set window.__waiter = true to bring it back
    // for a screenshot; the study never should. The seated diners stay: they are clearly seated,
    // far from the counter, and they are what makes the room read as a restaurant.
    if (window.__waiter) {
      for (let y = 0; y < h && dir; y++) {
        const x = st.terrain[y].indexOf("serving");
        if (x < 0) continue;
        const p = this.toWorld(x, y);
        const wtx = p.x + dir[0] * 1.25 + dir[1] * 0.45, wtz = p.z + dir[1] * 1.25 + dir[0] * 0.45;
        this.staticGroup.add(this._waiter(wtx, wtz, Math.atan2(-dir[0], -dir[1])));
        this.staticGroup.add(this._contactShadow(wtx, -0.292, wtz, 0.4, 0.28));
        break;
      }
    }

    // --- world-space extras keyed to actual station tiles ---
    this._hoodAndMats(st);
  }

  // standing front-of-house figure: white shirt, black waist apron, no face —
  // same background-people idiom as the seated guests
  _waiter(x, z, yaw) {
    const g = new THREE.Group(); g.position.set(x, -0.3, z); g.rotation.y = yaw;
    const legs = new THREE.Mesh(this.g("wtleg", () => new THREE.CylinderGeometry(0.13, 0.15, 0.55, 12)), mat(0x1e1f22, { roughness: 0.85 }));
    legs.position.y = 0.28; g.add(legs);
    const apron = new THREE.Mesh(this.g("wtapr", () => new THREE.CylinderGeometry(0.155, 0.17, 0.34, 12)), mat(0x27292d, { roughness: 0.8 }));
    apron.position.y = 0.62; g.add(apron);
    const torso = new THREE.Mesh(this.g("wttor", () => new THREE.CapsuleGeometry(0.2, 0.34, 4, 12)), mat(0xf0eee8, { roughness: 0.85 }));
    torso.position.y = 0.98; torso.scale.set(1, 0.95, 0.8); g.add(torso);
    const head = new THREE.Mesh(this.sph(0.125, 12), mat(0xd9a97e, { roughness: 0.65 }));
    head.position.y = 1.42; g.add(head);
    const hair = new THREE.Mesh(this.sph(0.13, 12), mat(0x2a2320, { roughness: 0.9 }));
    hair.position.y = 1.45; hair.scale.set(1, 0.7, 1); g.add(hair);
    for (const s of [-1, 1]) {     // arms folded low, holding a round tray
      const arm = new THREE.Mesh(this.g("wtarm", () => new THREE.CapsuleGeometry(0.05, 0.26, 3, 8)), mat(0xf0eee8, { roughness: 0.85 }));
      arm.position.set(s * 0.24, 1.0, 0.08); arm.rotation.x = -0.9; arm.rotation.z = s * 0.35; g.add(arm);
    }
    const tray = new THREE.Mesh(this.g("wttray", () => new THREE.CylinderGeometry(0.19, 0.19, 0.02, 18)), mat(0x3a3d41, { metalness: 0.6, roughness: 0.4 }));
    tray.position.set(0, 0.92, 0.26); g.add(tray);
    g.traverse(o => { if (o.isMesh) o.castShadow = false; });
    return g;
  }

  _dinTable(x, y, z, nGuests) {
    const g = new THREE.Group();
    const woodT = this.pbr("wood", { repeat: 1.2, roughness: 0.42, color: 0x6e4f30 });
    const top = new THREE.Mesh(this.g("tbtop", () => new THREE.CylinderGeometry(0.62, 0.62, 0.05, 24)), woodT);
    top.position.set(x, y + 0.74, z); g.add(top);
    const col = new THREE.Mesh(this.g("tbcol", () => new THREE.CylinderGeometry(0.05, 0.05, 0.7, 12)), mat(0x1f2124, { roughness: 0.5, metalness: 0.7 }));
    col.position.set(x, y + 0.37, z); g.add(col);
    const base = new THREE.Mesh(this.g("tbbase", () => new THREE.CylinderGeometry(0.3, 0.34, 0.04, 20)), mat(0x1f2124, { roughness: 0.5, metalness: 0.7 }));
    base.position.set(x, y + 0.03, z); g.add(base);
    // table setting: plates and drinking glasses for the seated party
    const porc = mat(0xf1f2f0, { roughness: 0.3 });
    for (let i = 0; i < nGuests; i++) {
      const a = i * 2.1 + x;
      const pl = new THREE.Mesh(this.g("tbplate", () => new THREE.CylinderGeometry(0.14, 0.11, 0.022, 18)), porc);
      pl.position.set(x + Math.cos(a) * 0.34, y + 0.78, z + Math.sin(a) * 0.34); g.add(pl);
      const gl = new THREE.Mesh(this.g("tbglass", () => new THREE.CylinderGeometry(0.035, 0.03, 0.11, 10)),
        mat(0xc8b06a, { roughness: 0.12, metalness: 0.1 }));
      gl.position.set(x + Math.cos(a + 0.7) * 0.45, y + 0.82, z + Math.sin(a + 0.7) * 0.45); g.add(gl);
    }
    return g;
  }
  _dinChair(x, y, z, aToTable) {
    const g = new THREE.Group();
    const wood = this.pbr("wood", { repeat: 0.9, roughness: 0.55, color: 0x5e4227 });
    const legM = mat(0x232528, { roughness: 0.55, metalness: 0.6 });
    const seat = new THREE.Mesh(this.g("chseat", () => new THREE.BoxGeometry(0.4, 0.045, 0.4)), wood);
    seat.position.set(x, y + 0.46, z); g.add(seat);
    for (const [sx, sz] of [[-0.16, -0.16], [0.16, -0.16], [-0.16, 0.16], [0.16, 0.16]]) {
      const leg = new THREE.Mesh(this.g("chleg", () => new THREE.CylinderGeometry(0.016, 0.014, 0.45, 8)), legM);
      leg.position.set(x + sx, y + 0.225, z + sz); g.add(leg);
    }
    const back = new THREE.Mesh(this.g("chback", () => new THREE.BoxGeometry(0.4, 0.42, 0.035)), wood);
    // backrest on the side AWAY from the table
    back.position.set(x + Math.cos(aToTable + Math.PI) * 0.19, y + 0.72, z + Math.sin(aToTable + Math.PI) * 0.19);
    back.rotation.y = -aToTable + Math.PI / 2; back.rotation.x = 0.06; g.add(back);
    g.traverse(o => { if (o.isMesh) o.castShadow = false; });
    return g;
  }
  // a seated guest: soft muted colours, no face detail — background people
  _patron(x, y, z, aFromTable, i) {
    const g = new THREE.Group();
    const SHIRT = [0x6b5d4f, 0x49566b, 0x7a5f52, 0x546052, 0x6d5a6e, 0x5d4d3b, 0x726a58, 0x4e5a66];
    const SKIN = [0xe8bd9a, 0xc9976c, 0x9a6b47, 0xf0cdb0];
    const shirt = mat(SHIRT[i % SHIRT.length], { roughness: 0.9 });
    const skin = mat(SKIN[i % SKIN.length], { roughness: 0.65 });
    const hairM = mat([0x2a2320, 0x4a3826, 0x6b6862, 0x1c1a18][i % 4], { roughness: 0.9 });
    const dx = -Math.cos(aFromTable), dz = -Math.sin(aFromTable);   // toward table
    const torso = new THREE.Mesh(this.g("patbody", () => new THREE.CapsuleGeometry(0.21, 0.3, 4, 12)), shirt);
    torso.position.set(x, y + 0.82, z); torso.scale.set(1, 0.92, 0.82); g.add(torso);
    // lean gently toward the table (head follows the tipped torso top)
    torso.quaternion.setFromAxisAngle(new THREE.Vector3(dz, 0, -dx), 0.14);
    const head = new THREE.Mesh(this.sph(0.13, 12), skin);
    head.position.set(x + dx * 0.055, y + 1.21, z + dz * 0.055); g.add(head);
    const hair = new THREE.Mesh(this.sph(0.135, 12), hairM);
    hair.position.set(x + dx * 0.05, y + 1.235, z + dz * 0.05); hair.scale.set(1, 0.72, 1);
    hair.rotation.y = aFromTable; g.add(hair);
    g.traverse(o => { if (o.isMesh) o.castShadow = false; });
    return g;
  }
  _pendant(x, y, z) {
    const g = new THREE.Group();
    const rod = new THREE.Mesh(this.g("pdrod", () => new THREE.CylinderGeometry(0.012, 0.012, 2.1, 6)), mat(0x1b1c1e, { roughness: 0.5 }));
    rod.position.set(x, y + 3.13, z); g.add(rod);
    const shade = new THREE.Mesh(this.g("pdshade", () => new THREE.ConeGeometry(0.26, 0.24, 18, 1, true)),
      new THREE.MeshStandardMaterial({ color: 0x2e3a33, roughness: 0.35, metalness: 0.4, side: THREE.DoubleSide }));
    shade.position.set(x, y + 2.06, z); g.add(shade);
    const bulb = new THREE.Mesh(this.sph(0.05, 10),
      new THREE.MeshStandardMaterial({ color: 0xffe6bb, emissive: 0xffc37a, emissiveIntensity: 3.2 }));
    bulb.position.set(x, y + 1.99, z); g.add(bulb);
    return g;
  }
  _wallShelf(x, y, z, len, ry) {
    // two cantilevered steel shelves stocked with stockpots, tubs and sacks
    const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = ry;
    const steelM = this.steel({ roughness: 0.42 });
    const potM = mat(0xb9bec3, { metalness: 0.9, roughness: 0.3 });
    const tubM = mat(0xeceae2, { roughness: 0.55 });
    const lidM = mat(0x8f959b, { metalness: 0.85, roughness: 0.35 });
    for (const sy of [1.35, 2.0]) {
      const plank = new THREE.Mesh(new THREE.BoxGeometry(len, 0.04, 0.42), steelM);
      plank.position.set(0, sy, 0.24); g.add(plank);
      for (const bx of [-len / 2 + 0.2, 0, len / 2 - 0.2]) {
        const br = new THREE.Mesh(this.g("shbr", () => new THREE.BoxGeometry(0.035, 0.3, 0.36)), steelM);
        br.position.set(bx, sy - 0.17, 0.22); g.add(br);
      }
    }
    // stock: pots on the lower shelf, tubs + sacks on the upper
    for (let i = 0; i < 4; i++) {
      const px = -len / 2 + 0.45 + i * (len - 0.9) / 3;
      const pot = new THREE.Mesh(this.g("shpot", () => new THREE.CylinderGeometry(0.16, 0.15, 0.26, 16)), potM);
      pot.position.set(px, 1.5, 0.24); g.add(pot);
      const lid = new THREE.Mesh(this.g("shlid", () => new THREE.CylinderGeometry(0.165, 0.165, 0.02, 16)), lidM);
      lid.position.set(px, 1.64, 0.24); g.add(lid);
      const knob = new THREE.Mesh(this.sph(0.03, 8), lidM);
      knob.position.set(px, 1.67, 0.24); g.add(knob);
    }
    for (let i = 0; i < 3; i++) {
      const px = -len / 2 + 0.5 + i * (len - 1.0) / 2;
      const tub = new THREE.Mesh(this.g("shtub", () => new THREE.BoxGeometry(0.3, 0.24, 0.3)), tubM);
      tub.position.set(px, 2.14, 0.22); tub.rotation.y = i * 0.2; g.add(tub);
    }
    const sack = new THREE.Mesh(this.sph(0.17, 12), mat(0xcfc0a0, { roughness: 0.95 }));
    sack.position.set(len / 2 - 0.35, 2.13, 0.24); sack.scale.set(1, 0.85, 0.8); g.add(sack);
    g.traverse(o => { if (o.isMesh) o.castShadow = false; });
    return g;
  }
  _fridge(x, y, z, ry) {
    const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = ry;
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.15, 2.15, 0.85), this.steel({ roughness: 0.34 }));
    body.position.y = 1.075; body.castShadow = true; g.add(body);
    const grille = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.2, 0.86), mat(0x2a2c2f, { roughness: 0.7, metalness: 0.4 }));
    grille.position.set(0, 2.02, 0); g.add(grille);
    const seam = new THREE.Mesh(new THREE.BoxGeometry(0.014, 1.75, 0.015), mat(0x33363a, { roughness: 0.6 }));
    seam.position.set(0, 1.0, 0.428); g.add(seam);
    for (const sx of [-0.14, 0.14]) {
      const hd = new THREE.Mesh(this.g("frhdl", () => new THREE.CylinderGeometry(0.02, 0.02, 0.7, 8)), this.steel({ roughness: 0.3 }));
      hd.position.set(sx, 1.15, 0.46); g.add(hd);
    }
    return g;
  }
  _crateStack(x, y, z, ry = 0) {
    const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = ry;
    const wood = this.pbr("wood", { repeat: 0.8, roughness: 0.85, color: 0xa78255 });
    let cy = 0.19;
    for (const s of [0.62, 0.58, 0.55]) {
      const c = new THREE.Mesh(new THREE.BoxGeometry(s, 0.36, s), wood);
      c.position.set(Math.sin(cy * 31) * 0.05, cy, Math.cos(cy * 17) * 0.04);
      c.rotation.y = cy * 2.2; g.add(c);
      cy += 0.37;
    }
    // produce peeking out of the top crate
    for (let i = 0; i < 5; i++) {
      const p = new THREE.Mesh(this.sph(0.075, 10), mat(i % 2 ? 0xc23b28 : 0x7ba23c, { roughness: 0.5 }));
      p.position.set(Math.cos(i * 2.4) * 0.14, cy - 0.2, Math.sin(i * 2.4) * 0.14); g.add(p);
    }
    g.traverse(o => { if (o.isMesh) o.castShadow = false; });
    return g;
  }
  _swingDoor(x, y, z) {
    // double in/out doors to the storeroom, porthole windows + steel kick plates
    const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = Math.PI / 2;
    const frameM = mat(0x3f4246, { roughness: 0.6, metalness: 0.3 });
    for (const s of [-0.98, 0.98]) {
      const j = new THREE.Mesh(new THREE.BoxGeometry(0.1, 2.5, 0.14), frameM);
      j.position.set(s, 1.25, 0); g.add(j);
    }
    const head = new THREE.Mesh(new THREE.BoxGeometry(2.06, 0.1, 0.14), frameM);
    head.position.set(0, 2.54, 0); g.add(head);
    const doorM = mat(0x8b8f94, { roughness: 0.55, metalness: 0.35 });
    const winM = mat(0x14161a, { roughness: 0.15, metalness: 0.3 });
    for (const s of [-0.47, 0.47]) {
      const door = new THREE.Mesh(new THREE.BoxGeometry(0.9, 2.46, 0.05), doorM);
      door.position.set(s, 1.23, 0.02); g.add(door);
      const win = new THREE.Mesh(this.g("dwin", () => new THREE.CylinderGeometry(0.15, 0.15, 0.06, 18)), winM);
      win.rotation.x = Math.PI / 2; win.position.set(s, 1.72, 0.03); g.add(win);
      const kick = new THREE.Mesh(new THREE.BoxGeometry(0.88, 0.42, 0.052), this.steel({ roughness: 0.35 }));
      kick.position.set(s, 0.23, 0.026); g.add(kick);
    }
    return g;
  }
  // soft contact-shadow decal — grounds furniture without extra shadow lights
  _contactShadow(x, y, z, r, opacity = 0.28) {
    if (!this._ctShTex) {
      const C = document.createElement("canvas"); C.width = C.height = 64;
      const ctx = C.getContext("2d");
      const gr = ctx.createRadialGradient(32, 32, 4, 32, 32, 32);
      gr.addColorStop(0, "rgba(0,0,0,.95)"); gr.addColorStop(0.65, "rgba(0,0,0,.5)"); gr.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = gr; ctx.fillRect(0, 0, 64, 64);
      this._ctShTex = new THREE.CanvasTexture(C);
    }
    const m = new THREE.Mesh(this.g("ctsh", () => new THREE.PlaneGeometry(1, 1)),
      new THREE.MeshBasicMaterial({ map: this._ctShTex, transparent: true, opacity, depthWrite: false }));
    m.rotation.x = -Math.PI / 2; m.position.set(x, y + 0.006, z);
    m.scale.set(r * 2, r * 2, 1); m.renderOrder = 2;
    return m;
  }
  // ventilation hood suspended over the hob line (open-top: the follow camera
  // shoots down through it) + rubber anti-fatigue mats on the working aisle
  _hoodAndMats(st) {
    const hobs = [];
    for (let y = 0; y < st.height; y++) for (let x = 0; x < st.width; x++)
      if (st.terrain[y][x] === "pot" || st.terrain[y][x] === "pan") hobs.push([x, y]);
    const matM = mat(0x393d41, { roughness: 0.94 });
    for (let y = 0; y < st.height; y++) for (let x = 0; x < st.width; x++) {
      const t = st.terrain[y][x];
      if (!["pot", "pan", "sink", "cutting_board"].includes(t)) continue;
      const yaw = this._floorYaw(st, x, y);
      const fv = [Math.sin(yaw), Math.cos(yaw)];
      const fx = x + Math.round(fv[0]), fy = y + Math.round(fv[1]);
      const row = st.terrain[fy];
      if (!row || row[fx] !== "floor") continue;
      const mp = this.toWorld(fx, fy);
      const rmat = new THREE.Mesh(this.g("rmat", () => new THREE.BoxGeometry(0.9, 0.022, 0.9)), matM);
      rmat.position.set(mp.x, 0.012, mp.z); rmat.receiveShadow = true;
      this.staticGroup.add(rmat);
      if (t === "sink") {         // wet sheen where dish water gets splashed
        const wet = new THREE.Mesh(this.g("wetp", () => new THREE.CircleGeometry(0.5, 22)),
          new THREE.MeshStandardMaterial({ color: 0x30343a, transparent: true, opacity: 0.15,
                                           roughness: 0.06, metalness: 0.35, depthWrite: false }));
        wet.rotation.x = -Math.PI / 2;
        wet.position.set(mp.x + fv[1] * 0.75, 0.005, mp.z - fv[0] * 0.75);
        wet.renderOrder = 1;
        this.staticGroup.add(wet);
      }
    }
    // VENT HOOD REMOVED (user 2026-07-26): it read as realistic but occluded the
    // hob line from too many angles. _buildHood is kept below, unused, in case a
    // low-profile variant is ever wanted.
  }
  _buildHood(st, hobs) {
    let minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9, ax = 0, az = 0;
    for (const [x, y] of hobs) {
      const p = this.toWorld(x, y);
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
      const yaw = this._floorYaw(st, x, y);
      ax += Math.sin(yaw); az += Math.cos(yaw);         // aisle direction (avg)
    }
    const n = hobs.length; ax /= n; az /= n;
    // canopy footprint: hob bbox + margin, pushed toward the wall, clipped on
    // the aisle side so the steep follow camera still sees pans and progress
    const cx = (minX + maxX) / 2 - ax * 0.18, cz = (minZ + maxZ) / 2 - az * 0.18;
    const sx = maxX - minX + (Math.abs(ax) > 0.5 ? 1.15 : 1.5);
    const sz = maxZ - minZ + (Math.abs(az) > 0.5 ? 1.15 : 1.5);
    const hood = new THREE.Group(); hood.position.set(cx, 0, cz);
    // dull satin steel — brushed PBR + flat white was reading as folded paper
    const bandM = new THREE.MeshStandardMaterial({ color: 0x9fa5aa, metalness: 0.72, roughness: 0.5 });
    bandM.envMapIntensity = 0.55;
    const Y0 = 2.28, BH = 0.55;      // rim bottom / band height — open top AND bottom
    for (const [bx, bz, bw, bd] of [[0, -sz / 2, sx, 0.05], [0, sz / 2, sx, 0.05],
                                    [-sx / 2, 0, 0.05, sz], [sx / 2, 0, 0.05, sz]]) {
      const b = new THREE.Mesh(new THREE.BoxGeometry(bw + 0.05, BH, bd + 0.05), bandM);
      b.position.set(bx, Y0 + BH / 2, bz); hood.add(b);
    }
    // grease-filter strip along the inside lower edge + warm task light strip
    const filt = new THREE.Mesh(new THREE.BoxGeometry(sx - 0.2, 0.1, sz - 0.2), mat(0x3a3e43, { metalness: 0.7, roughness: 0.45 }));
    filt.position.set(0, Y0 + 0.03, 0); hood.add(filt);
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(sx * 0.6, 0.03, 0.08),
      new THREE.MeshStandardMaterial({ color: 0xfff2d8, emissive: 0xffd9a0, emissiveIntensity: 1.6 }));
    lamp.position.set(0, Y0 + 0.02, az * (sz / 2 - 0.18)); hood.add(lamp);
    // sloped canopy skirt + exhaust duct: reads as a real suspended extraction
    // hood, not a floating planter. All rays from the follow camera to the hobs
    // pass well aisle-side of these faces (verified against the camera geometry),
    // so pans and progress bars stay fully visible.
    // boxy canopy top + exhaust duct rising above it — the honest silhouette of
    // a commercial extraction hood (an earlier sloped skirt warped under the
    // scale-then-rotate transform and read as folded paper)
    const capRim = new THREE.Mesh(new THREE.BoxGeometry(sx + 0.12, 0.07, sz + 0.12), bandM);
    capRim.position.y = Y0 + BH + 0.035; hood.add(capRim);
    const duct = new THREE.Mesh(new THREE.BoxGeometry(Math.min(0.95, sx * 0.42), 1.1, Math.min(0.75, sz * 0.62)), bandM);
    duct.position.y = Y0 + BH + 0.62; hood.add(duct);
    // utensil rail hanging under the hood's wall-side edge
    const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, sx * 0.8, 8), bandM);
    rail.rotation.z = Math.PI / 2; rail.position.set(0, Y0 - 0.35, -az * (sz / 2 - 0.1));
    if (Math.abs(ax) > 0.5) { rail.rotation.z = 0; rail.rotation.x = Math.PI / 2; rail.position.set(-ax * (sx / 2 - 0.1), Y0 - 0.35, 0); }
    hood.add(rail);
    const utM = mat(0x2e3134, { metalness: 0.75, roughness: 0.4 });
    const along = Math.abs(ax) > 0.5 ? [0, 0, 1] : [1, 0, 0];
    for (let i = -1; i <= 1; i++) {
      const hx = rail.position.x + along[0] * i * sx * 0.26, hz = rail.position.z + along[2] * i * sx * 0.26;
      const hook = new THREE.Mesh(this.g("uthk", () => new THREE.CylinderGeometry(0.008, 0.008, 0.12, 6)), utM);
      hook.position.set(hx, Y0 - 0.42, hz); hood.add(hook);
      if (i === 0) {          // hanging ladle
        const cup = new THREE.Mesh(this.sph(0.055, 10), utM);
        cup.position.set(hx, Y0 - 0.52, hz); cup.scale.set(1, 0.7, 1); hood.add(cup);
      } else {                // hanging pan silhouette
        const pb = new THREE.Mesh(this.g("utpan", () => new THREE.CylinderGeometry(0.11, 0.1, 0.035, 14)), utM);
        pb.rotation.x = Math.PI / 2; pb.position.set(hx, Y0 - 0.58, hz); hood.add(pb);
      }
    }
    hood.traverse(o => { if (o.isMesh) o.castShadow = false; });
    this.staticGroup.add(hood);
  }

  // --- surround canvas textures ---
  _menuTex() {
    if (this._menuTexC) return this._menuTexC;
    const C = document.createElement("canvas"); C.width = 512; C.height = 320;
    const g = C.getContext("2d");
    g.fillStyle = "#242a25"; g.fillRect(0, 0, 512, 320);
    for (let i = 0; i < 500; i++) {  // chalk dust
      g.fillStyle = "rgba(255,255,255," + (Math.random() * 0.05) + ")";
      g.fillRect(Math.random() * 512, Math.random() * 320, 2, 2);
    }
    g.fillStyle = "#e9e4d2"; g.textAlign = "center";
    g.font = "italic 44px Georgia"; g.fillText("~ Today ~", 256, 62);
    g.strokeStyle = "rgba(233,228,210,.7)"; g.lineWidth = 2;
    g.beginPath(); g.moveTo(110, 82); g.lineTo(402, 82); g.stroke();
    const names = [];
    const seen = new Set();
    for (const o of (this.state && this.state.orders) || [])
      if (o.name && !seen.has(o.name)) { seen.add(o.name); names.push([o.name, o.reward]); }
    if (!names.length) names.push(["Cheeseburger", 34], ["Soup of the day", 21]);
    g.font = "32px Georgia";
    names.slice(0, 4).forEach(([nm, rw], i) => {
      g.textAlign = "left"; g.fillText(nm, 70, 138 + i * 52);
      g.textAlign = "right"; g.fillText("$" + Math.max(6, Math.round((rw || 24) / 3)), 442, 138 + i * 52);
    });
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace;
    this._menuTexC = t; return t;
  }
  _neonTex() {
    if (this._neonTexC) return this._neonTexC;
    const C = document.createElement("canvas"); C.width = 512; C.height = 136;
    const g = C.getContext("2d");
    g.clearRect(0, 0, 512, 136);
    g.font = "bold 78px Arial"; g.textAlign = "center"; g.textBaseline = "middle";
    g.shadowColor = "#ff4a30"; g.shadowBlur = 26;
    g.strokeStyle = "#ff6a50"; g.lineWidth = 7;
    g.strokeText("ORDER UP", 256, 68);
    g.shadowBlur = 0; g.strokeStyle = "#ffd9c8"; g.lineWidth = 2.5;
    g.strokeText("ORDER UP", 256, 68);
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace;
    this._neonTexC = t; return t;
  }
  _clockTex() {
    const C = document.createElement("canvas"); C.width = C.height = 128;
    const g = C.getContext("2d");
    g.fillStyle = "#f2efe8"; g.beginPath(); g.arc(64, 64, 62, 0, 6.29); g.fill();
    g.strokeStyle = "#2b2b2e"; g.lineWidth = 3;
    for (let i = 0; i < 12; i++) {
      const a = i * Math.PI / 6;
      g.beginPath(); g.moveTo(64 + Math.cos(a) * 52, 64 + Math.sin(a) * 52);
      g.lineTo(64 + Math.cos(a) * 58, 64 + Math.sin(a) * 58); g.stroke();
    }
    g.lineWidth = 5; g.beginPath(); g.moveTo(64, 64); g.lineTo(64 + 28, 64 - 20); g.stroke();
    g.lineWidth = 3.5; g.beginPath(); g.moveTo(64, 64); g.lineTo(64 - 12, 64 - 40); g.stroke();
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  // blotchy grayscale surface for pot liquids — update() slowly swirls it so a
  // cooking pot visibly simmers instead of holding a frozen flat disc
  _simmerT() {
    if (this._simmerTexC) return this._simmerTexC;
    const C = document.createElement("canvas"); C.width = C.height = 64;
    const g = C.getContext("2d");
    g.fillStyle = "#c6c6c6"; g.fillRect(0, 0, 64, 64);
    for (let i = 0; i < 46; i++) {
      const x = (i * 13) % 64, y = (i * 29 + i * i) % 64, r = 2 + (i % 5);
      g.fillStyle = i % 2 ? "rgba(255,255,255,.55)" : "rgba(110,110,110,.4)";
      g.beginPath(); g.arc(x, y, r, 0, 6.28); g.fill();
    }
    const t = new THREE.CanvasTexture(C);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.center.set(0.5, 0.5);
    this._simmerTexC = t; return t;
  }
  _sourceLabelTex(name, color) {
    const key = "lbl_" + name;
    if (this._foodTexCache[key]) return this._foodTexCache[key];
    const C = document.createElement("canvas"); C.width = 160; C.height = 60;
    const g = C.getContext("2d");
    g.fillStyle = "#f4f2ec"; g.fillRect(0, 0, 160, 60);
    g.fillStyle = "#" + color.toString(16).padStart(6, "0");
    g.fillRect(0, 46, 160, 14);
    g.fillStyle = "#33343a"; g.font = "bold 26px Arial";
    g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(name.toUpperCase(), 80, 26);
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace;
    this._foodTexCache[key] = t; return t;
  }
  _ticketTex(i) {
    const C = document.createElement("canvas"); C.width = 72; C.height = 96;
    const g = C.getContext("2d");
    g.fillStyle = "#f6f3ea"; g.fillRect(0, 0, 72, 96);
    g.fillStyle = "#3a3a40"; g.font = "bold 13px monospace"; g.textAlign = "center";
    g.fillText("#" + (12 + i * 3), 36, 16);
    g.strokeStyle = "rgba(60,60,70,.75)"; g.lineWidth = 2;
    for (let k = 0; k < 4; k++) {
      g.beginPath(); g.moveTo(8, 30 + k * 15); g.lineTo(10 + ((i * 13 + k * 29) % 50), 30 + k * 15); g.stroke();
    }
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }

  // ---- ingredient / plate meshes --------------------------------------
  _ingredient(name, state) {
    const k = ING[name] || { c: 0xcccccc, c2: 0x999999 };
    // 2026-09-09: per-food COOKED / CHOPPED looks. The generic pass below tints every cooked
    // item 80% toward dark brown and caps it with a crust blob, so cooked rice, fries and
    // seared steak were indistinguishable brown lumps on the plate. Foods with a real-world
    // prepared look get their own mesh here (white fluffy rice, golden fries, seared strips,
    // fried egg with a yolk, battered fish, pink cooked shrimp ...); everything else keeps
    // the generic treatment.
    if (this.real) {
      // photoreal library (web/js/food3d.js): displaced geometry, generated PBR textures,
      // physical materials, the photoscanned bun. Classic mode keeps the legacy look.
      const real = foodMesh(name, state);
      if (real) return real;
    }
    const special = this.real ? this._foodSpecial(name, state, k) : null;
    if (special) return special;
    const col = tintFor(state, k.c);
    const grp = new THREE.Group();
    const add = (geo, m, x = 0, y = 0, z = 0, sx, sy, sz) => {
      const mesh = new THREE.Mesh(geo, m); mesh.position.set(x, y, z);
      if (sx !== undefined) mesh.scale.set(sx, sy, sz); mesh.castShadow = true; grp.add(mesh); return mesh;
    };
    const M = mat(col, { roughness: 0.55 });
    switch (name) {
      case "onion": add(this.sph(0.16), M, 0, 0.16); add(this.cyl(0.0, 0.03, 0.1, 6), mat(0x6cae3a), 0, 0.32); break;
      case "tomato": add(this.sph(0.15), M, 0, 0.15); add(this.sph(0.05), mat(k.c2), 0, 0.29); break;
      case "lettuce":
        if (this.real) {                       // ruffled head: flattened overlapping leaves
          for (let i = 0; i < 8; i++) {
            const a = i / 8 * 6.28, rr = i % 2 ? 0.1 : 0.055;
            const leaf = add(this.sph(0.11), mat(i % 2 ? 0x7ccc55 : 0x55a53c, { roughness: 0.5 }),
                             Math.cos(a) * rr, 0.06 + (i % 3) * 0.022, Math.sin(a) * rr);
            leaf.scale.set(1.15, 0.38, 0.8); leaf.rotation.y = a + 0.4;
          }
        } else for (let i = 0; i < 5; i++) { const a = i / 5 * 6.28; add(this.sph(0.1), mat(i % 2 ? k.c : k.c2), Math.cos(a) * 0.08, 0.13, Math.sin(a) * 0.08); }
        break;
      case "mushroom": add(this.cyl(0.06, 0.07, 0.16, 10), mat(0xefe6d8), 0, 0.08); add(this.sph(0.15), M, 0, 0.18, 0, 1, 0.6, 1); break;
      case "meat":
        if (this.real) {                       // ground patty with a photo-ish surface:
          const mm = new THREE.MeshStandardMaterial({ map: this._meatTex(state), // raw grind / seared crust
            roughness: state === "cooked" ? 0.4 : 0.6 });
          const patty = add(this.cyl(0.18, 0.175, 0.08, 24), mm, 0, 0.075);
          patty.scale.y = 1;                   // (kept explicit for clarity)
        } else add(this.cyl(0.18, 0.18, 0.09, 16), M, 0, 0.09);
        break;
      case "fish": { const b = add(this.sph(0.15), M, 0, 0.12, 0, 1.5, 0.7, 0.8); add(this.cyl(0, 0.12, 0.02, 3), M, -0.26, 0.12, 0).rotation.z = Math.PI / 2; break; }
      case "bun":
        if (this.real) {                       // proper burger bun: heel + golden domed
          const heel = add(this.cyl(0.15, 0.14, 0.05, 22), mat(0xeacb92, { roughness: 0.65 }), 0, 0.025);  // crown + sesame seeds
          heel.castShadow = true;
          add(this.sph(0.163), mat(tintFor(state, 0xc9853f), { roughness: 0.52 }), 0, 0.062, 0, 1, 0.8, 1);
          const seedM = mat(0xf6ecd2, { roughness: 0.5 });
          for (let i = 0; i < 11; i++) {
            const a = i * 2.4, rr = 0.045 + (i % 3) * 0.038;
            const sx = Math.cos(a) * rr, sz = Math.sin(a) * rr;
            const sy = 0.062 + Math.sqrt(Math.max(0.001, 0.163 * 0.163 - sx * sx - sz * sz)) * 0.8;
            const seed = add(this.sph(0.011, 6), seedM, sx, sy, sz);
            seed.scale.set(1, 0.5, 1.5); seed.rotation.y = a;
          }
        } else add(this.sph(0.17), M, 0, 0.12, 0, 1, 0.62, 1);
        break;
      case "cheese":
        if (this.real) {                       // two square deli slices with holes,
          const chM = new THREE.MeshStandardMaterial({ map: this._cheeseTex(), roughness: 0.42 });  // slightly fanned
          const s1 = add(this.box(0.3, 0.02, 0.3), chM, 0, 0.012); s1.rotation.y = 0.25;
          const s2 = add(this.box(0.3, 0.02, 0.3), chM, 0.012, 0.034, -0.01); s2.rotation.y = 0.6;
        } else { const m = add(this.cyl(0.0, 0.2, 0.16, 3), M, 0, 0.1); m.rotation.y = 0.5; }
        break;
      case "rice": add(this.cyl(0.16, 0.12, 0.12, 16), mat(tintFor(state, 0xf6f1e6), { roughness: 0.8 }), 0, 0.08); break;
      case "dough": add(this.sph(0.17), M, 0, 0.1, 0, 1, 0.55, 1); break;
      case "egg": add(this.sph(0.16), mat(tintFor(state, 0xfaf3e2)), 0, 0.07, 0, 1, 0.5, 1); add(this.sph(0.06), mat(state === "cooked" ? 0xf2a51e : 0xf6b73c), 0, 0.12); break;
      case "potato": add(this.sph(0.16), M, 0, 0.13, 0, 1.2, 0.85, 0.95); break;
      case "steak": {                          // flat red slab with white fat marbling —
        const slab = add(this.box(0.34, 0.07, 0.24), M, 0, 0.045);   // clearly NOT a patty
        slab.rotation.y = 0.35;
        const fat = mat(tintFor(state, k.c2), { roughness: 0.5 });
        const rim = add(this.box(0.34, 0.06, 0.035), fat, -0.04, 0.045, 0.105); rim.rotation.y = 0.35;
        const m1 = add(this.box(0.28, 0.012, 0.022), fat, 0.01, 0.085, -0.03); m1.rotation.y = 0.62;
        const m2 = add(this.box(0.2, 0.012, 0.018), fat, -0.03, 0.085, 0.045); m2.rotation.y = 0.18;
        break;
      }
      case "tortilla": add(this.cyl(0.2, 0.2, 0.025, 20), M, 0, 0.02); add(this.cyl(0.17, 0.17, 0.02, 20), mat(k.c2, { roughness: 0.6 }), 0, 0.04); break;
      case "noodles": {                       // a nest of pale strands
        add(this.cyl(0.17, 0.13, 0.1, 16), M, 0, 0.06);
        for (let i = 0; i < 4; i++) { const a = i / 4 * 6.28; const strand = add(this.g("noodle", () => new THREE.TorusGeometry(0.09, 0.02, 6, 16)), mat(k.c2, { roughness: 0.6 }), Math.cos(a) * 0.06, 0.12 + (i % 2) * 0.02, Math.sin(a) * 0.06); strand.rotation.x = -Math.PI / 2 + 0.25 * (i % 2); strand.rotation.z = a; }
        break;
      }
      case "shrimp": {                        // curved body + fanned tail
        const body = add(this.g("shrimpbody", () => new THREE.TorusGeometry(0.1, 0.05, 8, 12, Math.PI * 1.35)), M, 0, 0.12);
        body.rotation.z = 0.5;
        add(this.cyl(0.0, 0.06, 0.08, 6), mat(k.c2, { roughness: 0.5 }), 0.12, 0.09, 0).rotation.z = -0.9;
        break;
      }
      case "lemon": {                         // bright ellipsoid with two nubs — reads citrus
        add(this.sph(0.15), M, 0, 0.13, 0, 1.35, 0.85, 0.85);
        add(this.sph(0.045), mat(k.c2), 0.2, 0.13); add(this.sph(0.045), mat(k.c2), -0.2, 0.13);
        break;
      }
      case "peas": {                          // a mound of little green spheres
        for (let i = 0; i < 7; i++) { const a = i / 7 * 6.28; add(this.sph(0.055), M, Math.cos(a) * 0.09, 0.06 + (i % 2) * 0.02, Math.sin(a) * 0.09); }
        add(this.sph(0.055), mat(k.c2), 0, 0.12);
        break;
      }
      case "butter": {                        // pale rectangular block on a foil edge
        add(this.box(0.28, 0.12, 0.18), M, 0, 0.07);
        add(this.box(0.3, 0.02, 0.2), mat(k.c2, { roughness: 0.4 }), 0, 0.012);
        break;
      }
      case "pepper": {                        // tall dark grinder — unmistakably not food
        add(this.cyl(0.09, 0.11, 0.26, 12), M, 0, 0.14);
        add(this.cyl(0.07, 0.09, 0.08, 12), mat(k.c2, { roughness: 0.5 }), 0, 0.31);
        break;
      }
      case "avocado": {                       // dark ovoid, lighter cut face tilt
        add(this.sph(0.16), M, 0, 0.13, 0, 0.9, 1.15, 0.9);
        add(this.sph(0.1), mat(k.c2, { roughness: 0.55 }), 0.05, 0.19, 0, 0.9, 0.5, 0.9);
        break;
      }
      case "corn": {                          // yellow cob with green husk stub
        const cob = add(this.cyl(0.08, 0.08, 0.32, 10), M, 0, 0.09); cob.rotation.z = Math.PI / 2;
        add(this.cyl(0.05, 0.02, 0.12, 8), mat(k.c2, { roughness: 0.6 }), 0.2, 0.09, 0).rotation.z = Math.PI / 2 + 0.3;
        break;
      }
      default: add(this.sph(0.15), M, 0, 0.15);
    }
    if (this.real && name === "bun" && state === "raw") return bunModel("whole", 0.165, grp);
    if (state === "chopped") {
      // chopped = a little pile of diced cubes, clearly different from raw
      grp.clear();
      const dice = mat(col, { roughness: 0.5 });
      const cube = this.g("dice", () => new THREE.BoxGeometry(0.09, 0.09, 0.09));
      const spots = [[-0.1, 0.05, -0.05], [0.08, 0.05, 0.04], [0, 0.05, 0.11], [-0.04, 0.05, 0.05], [0.11, 0.05, -0.06], [-0.12, 0.14, 0.03]];
      for (const [x, y, z] of spots) {
        const d = new THREE.Mesh(cube, dice); d.position.set(x, y, z);
        d.rotation.set(Math.sin(x * 9), x * 4 + z, Math.cos(z * 7)); d.castShadow = true; grp.add(d);
      }
    }
    if (state === "cooked") {
      // warm sear glow + glossier surface so even pale foods (rice/egg/potato)
      // read as unmistakably COOKED, not raw
      grp.traverse(o => {
        if (o.isMesh && o.material.emissive) {
          o.material = o.material.clone();
          o.material.emissive = new THREE.Color(0x5a2400); o.material.emissiveIntensity = 0.45;
          o.material.roughness = Math.max(0.2, (o.material.roughness ?? 0.5) - 0.25);
        }
      });
      // a darker browned "crust" cap sitting on top makes the state change obvious
      // (real mode: meat/bun/cheese already carry their own sear/toast look —
      // a floating crust blob on top would break the photo read)
      if (!(this.real && (name === "meat" || name === "bun" || name === "cheese"))) {
        const crust = new THREE.Mesh(this.sph(0.13), new THREE.MeshStandardMaterial({ color: 0x3d1c08, roughness: 0.4, emissive: 0x2a1000, emissiveIntensity: 0.3 }));
        crust.scale.set(1, 0.32, 1); crust.position.y = 0.24; grp.add(crust);
      }
    }
    return grp;
  }

  // Bespoke prepared-food meshes. Returns a Group (origin at the food's base) or null.
  _foodSpecial(name, state, k) {
    if (state !== "cooked" && state !== "chopped") {
      if (name === "shrimp" && state === "raw") {          // raw shrimp are grey-blue, not orange
        const grp = new THREE.Group();
        const M = mat(0xb9c3c8, { roughness: 0.45 });
        const body = new THREE.Mesh(this.g("shrimpbody", () => new THREE.TorusGeometry(0.1, 0.05, 8, 12, Math.PI * 1.35)), M);
        body.position.y = 0.12; body.rotation.z = 0.5; body.castShadow = true; grp.add(body);
        const tail = new THREE.Mesh(this.cyl(0.0, 0.06, 0.08, 6), mat(0x8fa0aa, { roughness: 0.5 }));
        tail.position.set(0.12, 0.09, 0); tail.rotation.z = -0.9; grp.add(tail);
        return grp;
      }
      return null;
    }
    const grp = new THREE.Group();
    const add = (geo, m, x = 0, y = 0, z = 0, sx, sy, sz) => {
      const mesh = new THREE.Mesh(geo, m); mesh.position.set(x, y, z);
      if (sx !== undefined) mesh.scale.set(sx, sy, sz); mesh.castShadow = true; grp.add(mesh); return mesh;
    };
    const cooked = state === "cooked";
    // deterministic pseudo-random scatter (same look every rebuild, no flicker)
    const rnd = (i, j = 0) => { const v = Math.sin(i * 12.9898 + j * 78.233) * 43758.5453; return v - Math.floor(v); };
    switch (name) {
      case "rice": {
        if (!cooked) return null;
        // a fluffy WHITE mound with visible grains — stays white, never browns
        const base = mat(0xf7f3ea, { roughness: 0.9 });
        add(this.sph(0.2), base, 0, 0.07, 0, 1, 0.55, 1);
        add(this.sph(0.15), base, 0.05, 0.12, -0.03, 1, 0.6, 1);
        const grain = mat(0xffffff, { roughness: 0.75 });
        for (let i = 0; i < 26; i++) {
          const a = rnd(i) * 6.28, r = Math.sqrt(rnd(i, 1)) * 0.19;
          const x = Math.cos(a) * r, z = Math.sin(a) * r;
          const y = 0.07 + Math.sqrt(Math.max(0, 0.2 * 0.2 - x * x - z * z)) * 0.55 + 0.005;
          const g = add(this.sph(0.018, 6), grain, x, y, z, 1, 0.55, 1.9);
          g.rotation.y = rnd(i, 2) * 6.28; g.rotation.x = (rnd(i, 3) - 0.5) * 0.8;
        }
        return grp;
      }
      case "potato": {
        // chopped = raw pale fries; cooked = golden deep-fried fries, criss-crossed
        const body = cooked ? mat(0xe6b04c, { roughness: 0.5 }) : mat(0xf0e2a8, { roughness: 0.6 });
        const tip = cooked ? mat(0xb8792a, { roughness: 0.5 }) : mat(0xe4d494, { roughness: 0.6 });
        const stick = this.g("fry", () => new THREE.BoxGeometry(0.052, 0.052, 0.27));
        const cap = this.g("frycap", () => new THREE.BoxGeometry(0.054, 0.054, 0.04));
        for (let i = 0; i < 9; i++) {
          const layer = Math.floor(i / 3);
          const f = add(stick, body, (rnd(i) - 0.5) * 0.2, 0.03 + layer * 0.052, (rnd(i, 1) - 0.5) * 0.12);
          f.rotation.y = (layer % 2 ? 1.2 : 0) + (rnd(i, 2) - 0.5) * 0.9; f.rotation.x = (rnd(i, 3) - 0.5) * 0.25;
          const c1 = new THREE.Mesh(cap, tip); c1.position.z = 0.135; f.add(c1);
          const c2 = new THREE.Mesh(cap, tip); c2.position.z = -0.135; f.add(c2);
        }
        return grp;
      }
      case "steak": {
        // chopped = raw red strips with a fat streak; cooked = dark seared strips, pink edge
        const outer = cooked ? new THREE.MeshStandardMaterial({ map: this._steakTex(), roughness: 0.42 })
                             : mat(0xc23b2e, { roughness: 0.5 });
        // NB: the first-person "in hand" copy renders with depth testing OFF, so any child
        // mesh paints over its parent in insertion order — decorations must sit ON TOP of the
        // strip (a fat streak on raw strips), never inside it
        const fat = mat(0xf2e8dc, { roughness: 0.5 });
        for (let i = 0; i < 5; i++) {
          const st = add(this.box(0.26, 0.05, 0.065), outer, (rnd(i) - 0.5) * 0.1, 0.028 + (i >= 3 ? 0.05 : 0), (i - 2) * 0.075 * (i >= 3 ? 0.4 : 1));
          st.rotation.y = (rnd(i, 1) - 0.5) * 0.7 + (i >= 3 ? 0.5 : 0);
          if (!cooked) { const s2 = new THREE.Mesh(this.box(0.2, 0.008, 0.016), fat); s2.position.set(0.01, 0.027, 0.012); st.add(s2); }
        }
        return grp;
      }
      case "fish": {
        if (cooked) {                                  // golden battered fillet with a bumpy crust
          const crust = mat(0xd4913c, { roughness: 0.55 });
          add(this.sph(0.16), crust, 0, 0.09, 0, 1.55, 0.55, 0.85);
          const bump = mat(0xc27c2b, { roughness: 0.6 });
          for (let i = 0; i < 10; i++) add(this.sph(0.035, 7), bump, (rnd(i) - 0.5) * 0.4, 0.14 + rnd(i, 1) * 0.02, (rnd(i, 2) - 0.5) * 0.2, 1, 0.6, 1);
          return grp;
        }
        // chopped = pale pink sashimi slices with white lines
        const flesh = mat(0xf2a79c, { roughness: 0.35 }), line = mat(0xfff2ee, { roughness: 0.4 });
        for (let i = 0; i < 4; i++) {
          const sl = add(this.box(0.18, 0.03, 0.1), flesh, (i - 1.5) * 0.06, 0.016 + i * 0.012, (i % 2) * 0.05 - 0.025);
          sl.rotation.y = 0.3;
          const ln = new THREE.Mesh(this.box(0.18, 0.032, 0.012), line); ln.position.z = 0.02; sl.add(ln);
        }
        return grp;
      }
      case "egg": {
        if (!cooked) return null;                      // fried egg: white with a glossy yolk
        add(this.sph(0.15), mat(0xfbf8f1, { roughness: 0.5 }), 0, 0.02, 0, 1.15, 0.22, 1);
        add(this.sph(0.12), mat(0xfbf8f1, { roughness: 0.5 }), 0.06, 0.025, 0.04, 1, 0.24, 1);
        add(this.sph(0.07), mat(0xf5b117, { roughness: 0.25 }), 0.0, 0.05, -0.01, 1, 0.8, 1);
        return grp;
      }
      case "noodles": {
        if (!cooked) return null;                      // glossy yellow cooked noodles, not brown
        const M = mat(0xf3d98a, { roughness: 0.28 });
        add(this.cyl(0.17, 0.13, 0.1, 16), M, 0, 0.06);
        for (let i = 0; i < 5; i++) { const a = i / 5 * 6.28; const strand = add(this.g("noodle", () => new THREE.TorusGeometry(0.09, 0.02, 6, 16)), M, Math.cos(a) * 0.06, 0.12 + (i % 2) * 0.02, Math.sin(a) * 0.06); strand.rotation.x = -Math.PI / 2 + 0.25 * (i % 2); strand.rotation.z = a; }
        return grp;
      }
      case "shrimp": {
        if (!cooked) return null;                      // cooked shrimp turn pink-orange
        const M = mat(0xf07a4f, { roughness: 0.4 });
        const body = add(this.g("shrimpbody", () => new THREE.TorusGeometry(0.1, 0.05, 8, 12, Math.PI * 1.35)), M, 0, 0.12);
        body.rotation.z = 0.5;
        const band = add(this.g("shrimpband", () => new THREE.TorusGeometry(0.1, 0.052, 8, 12, Math.PI * 0.25)), mat(0xfbe4d8, { roughness: 0.4 }), 0, 0.12);
        band.rotation.z = 1.1;
        add(this.cyl(0.0, 0.06, 0.08, 6), mat(0xd8553a, { roughness: 0.5 }), 0.12, 0.09, 0).rotation.z = -0.9;
        return grp;
      }
      case "dough": {
        if (!cooked) return null;                      // baked: golden-brown roll with a score line
        add(this.sph(0.17), mat(0xc98a3f, { roughness: 0.6 }), 0, 0.09, 0, 1.2, 0.7, 0.9);
        add(this.box(0.24, 0.02, 0.03), mat(0xf0d9a8, { roughness: 0.7 }), 0, 0.2, 0).rotation.y = 0.3;
        return grp;
      }
      case "onion": {
        if (!cooked) return null;                      // soft golden translucent pieces (soup / stew)
        const M = new THREE.MeshStandardMaterial({ color: 0xe9c98a, roughness: 0.3, transparent: true, opacity: 0.9 });
        for (let i = 0; i < 7; i++) add(this.box(0.09, 0.03, 0.09), M, (rnd(i) - 0.5) * 0.22, 0.02 + (i % 2) * 0.03, (rnd(i, 1) - 0.5) * 0.22).rotation.y = rnd(i, 2) * 3;
        return grp;
      }
      case "tomato": {
        if (!cooked) return null;                      // cooked tomato collapses into a red sauce mound
        add(this.sph(0.19), mat(0xc7402a, { roughness: 0.28 }), 0, 0.03, 0, 1, 0.35, 1);
        for (let i = 0; i < 5; i++) add(this.sph(0.045, 8), mat(0xd9583d, { roughness: 0.35 }), (rnd(i) - 0.5) * 0.24, 0.07, (rnd(i, 1) - 0.5) * 0.24, 1, 0.6, 1);
        return grp;
      }
      case "mushroom": {
        if (!cooked) return null;                      // sautéed: glossy brown pieces
        const M = mat(0x7a5a3e, { roughness: 0.3 });
        for (let i = 0; i < 6; i++) add(this.sph(0.06, 10), M, (rnd(i) - 0.5) * 0.24, 0.05, (rnd(i, 1) - 0.5) * 0.22, 1, 0.6, 1.3).rotation.y = rnd(i, 2) * 3;
        return grp;
      }
      case "lettuce": {
        if (cooked) return null;                       // chopped = shredded ribbons
        const a = mat(0x8ad45c, { roughness: 0.5 }), b = mat(0x5fae3f, { roughness: 0.5 });
        for (let i = 0; i < 12; i++) {
          const rb = add(this.box(0.15, 0.012, 0.045), i % 2 ? a : b, (rnd(i) - 0.5) * 0.26, 0.012 + (i % 3) * 0.02, (rnd(i, 1) - 0.5) * 0.26);
          rb.rotation.set((rnd(i, 3) - 0.5) * 0.6, rnd(i, 2) * 6.28, (rnd(i, 4) - 0.5) * 0.5);
        }
        return grp;
      }
      case "tomato_chopped_unused": return null;
      default: return null;
    }
  }
  _steakTex() {
    return this._foodTex("steak_cooked", (ctx) => {
      ctx.fillStyle = "#6e3a1f"; ctx.fillRect(0, 0, 128, 128);
      for (let i = 0; i < 160; i++) {
        const x = (i * 37) % 128, y = (i * 61 + (i * i) % 17) % 128, r = 1 + (i % 3);
        ctx.fillStyle = i % 3 ? "rgba(150,90,50,.45)" : "rgba(60,28,10,.55)";
        ctx.beginPath(); ctx.arc(x, y, r, 0, 6.28); ctx.fill();
      }
      ctx.strokeStyle = "rgba(28,12,4,.85)"; ctx.lineWidth = 6;
      for (let k = -1; k < 5; k++) { ctx.beginPath(); ctx.moveTo(-10, k * 34); ctx.lineTo(138, k * 34 + 30); ctx.stroke(); }
    });
  }

  _isHotSoup(plate) {
    const c = plate.contents || [];
    return c.length >= 2 && c.every(i => i.state === "cooked") && c.every(i => i.name === c[0].name);
  }
  // what a finished soup/stew of this ingredient looks like (used for the plated soup and
  // the pot liquid): onion soup is golden, tomato soup red, not a uniform dark brown
  _soupColor(name) {
    return ({ onion: 0xd9a24a, tomato: 0xcf3f2a, mushroom: 0x8f6b4a, potato: 0xe6cd86,
              rice: 0xf3eee0, noodles: 0xefd48a }[name]) || tintFor("cooked", (ING[name] || { c: 0xcc8844 }).c);
  }

  _plate(plate) {
    const grp = new THREE.Group();
    const dirty = plate.dirty;
    // a proper bowl: rim + concave interior, so plated food clearly sits in it
    const bowl = new THREE.Mesh(this.cyl(0.34, 0.22, 0.1, 28), mat(dirty ? 0x9c8e74 : 0xf3f5f7, { roughness: 0.28 }));
    bowl.position.y = 0.05; bowl.castShadow = true; grp.add(bowl);
    const rim = new THREE.Mesh(this.g("bowlrim", () => new THREE.TorusGeometry(0.32, 0.03, 10, 28)), mat(dirty ? 0x8a7c64 : 0xe6e9ec, { roughness: 0.3 }));
    rim.rotation.x = Math.PI / 2; rim.position.y = 0.1; grp.add(rim);
    if (!dirty && plate.contents && plate.contents.length) {
      // a shaded inner well so light foods (rice, egg, bun) clearly stand out
      const well = new THREE.Mesh(this.cyl(0.27, 0.27, 0.02, 24), mat(0xc4ccd6, { roughness: 0.5 }));
      well.position.y = 0.11; grp.add(well);
    }
    if (!dirty && plate.contents && plate.contents.length) {
      const cont = plate.contents;
      if (this._isHotSoup({ contents: cont })) {
        const col = this._soupColor(cont[0].name);
        // a generous domed serving of soup that fills the bowl
        const soup = new THREE.Mesh(this.cyl(0.3, 0.28, 0.1, 28),
          new THREE.MeshStandardMaterial({ color: col, roughness: 0.32, emissive: new THREE.Color(col).multiplyScalar(0.16) }));
        soup.position.y = 0.12; grp.add(soup);
        const dome = new THREE.Mesh(this.sph(0.27), new THREE.MeshStandardMaterial({ color: col, roughness: 0.32 }));
        dome.scale.y = 0.35; dome.position.y = 0.15; grp.add(dome);
      } else if (this.real) {
        // REAL: the dish is assembled VERTICALLY in list order like a real
        // plate (bun heel → seared patty → draped cheese → sesame crown …).
        // Every added component changes the silhouette, so an agent can read
        // the exact contents purely from pixels.
        let y = 0.115, buns = 0, run = 0;
        cont.forEach((c, i) => {
          const isWholeBun = c.name === "bun" && c.state !== "chopped";
          const { g: layer, h } = this._platedLayer(c, isWholeBun ? buns++ : 0);
          // a SECOND portion of the same item (two fries, two chips) mounds beside the first
          // instead of towering on top of it — a real plate, and the order stays readable
          const prev = i > 0 ? cont[i - 1] : null;
          const same = prev && prev.name === c.name && prev.state === c.state && !isWholeBun;
          run = same ? run + 1 : 0;
          const side = same ? (run % 2 ? -1 : 1) * 0.11 : 0;
          // hand-stacked look: tiny deterministic per-layer offset + twist
          layer.position.set(Math.sin(i * 2.1) * 0.014 + side, same ? y - h * 0.55 : y, Math.cos(i * 1.7) * 0.014 + (same ? (run % 2 ? 0.05 : -0.05) : 0));
          layer.rotation.y = i * 0.6;
          grp.add(layer); y += same ? h * 0.45 : h;
        });
      } else {
        // a clear, generous mound of the components sitting up in the bowl so a
        // finished dish is never mistaken for an empty plate. Big builds (the
        // 5-layer Loaded Burrito) widen the ring, shrink pieces a touch and
        // alternate heights so every component stays individually legible
        const n = cont.length;
        cont.forEach((c, i) => {
          const m = this._ingredient(c.name, c.state);
          m.scale.setScalar(n >= 5 ? 0.68 : 0.85);
          const ang = n > 1 ? (i / n) * 6.28 : 0, rr = n > 1 ? (n >= 5 ? 0.18 : 0.13) : 0;
          m.position.set(Math.cos(ang) * rr, 0.2 + (n >= 5 ? (i % 2) * 0.05 : 0), Math.sin(ang) * rr);
          grp.add(m);
        });
      }
    }
    return grp;
  }
  // REAL mode only: one visually distinct, stackable layer per plated
  // component. Returns { g, h } — a group whose origin is the layer's BASE and
  // the height the next layer should be raised by. Raw/cooked/burnt looks are
  // preserved (meat keeps its sear/grill-mark canvas texture, bun/cheese tint
  // via tintFor), so doneness stays readable after plating.
  _platedLayer(c, bunIdx) {
    const g = new THREE.Group();
    const put = (mesh, y = 0) => { mesh.position.y += y; mesh.castShadow = true; g.add(mesh); return mesh; };
    const chopped = c.state === "chopped";
    if (c.name === "bun" && !chopped && c.state !== "burnt") {
      // photoscanned bun (food3d.js): the scan is cut into heel / crown; the procedural
      // slice below stays visible until the model has loaded
      const fb = new THREE.Group();
      const part = bunIdx === 0 ? "heel" : "crown";
      if (bunIdx === 0) {
        fb.add(new THREE.Mesh(this.cyl(0.17, 0.162, 0.05, 24), mat(0xe0b273, { roughness: 0.62 }))).position.y = 0.025;
      } else {
        const dome = new THREE.Mesh(this.g("buncrown", () => new THREE.SphereGeometry(0.16, 22, 12, 0, Math.PI * 2, 0, Math.PI / 2)), mat(0xc9853f, { roughness: 0.52 }));
        dome.scale.set(1, 0.85, 1); fb.add(dome);
      }
      g.add(bunModel(part, 0.165, fb));
      return { g, h: BUN_HEIGHT[part] };
    }
    if (c.name === "meat" && !chopped && c.state !== "burnt") {
      const p = foodMesh("meat", c.state); g.add(p);
      return { g, h: LAYER_H["meat/" + c.state] || 0.08 };
    }
    if (false) {
      if (bunIdx === 0) {                       // FIRST bun = heel: flat golden slice, pale cut face up
        // wider than the patty so its golden rim stays readable under the stack
        put(new THREE.Mesh(this.cyl(0.17, 0.162, 0.05, 24), mat(tintFor(c.state, 0xe0b273), { roughness: 0.62 })), 0.025);
        put(new THREE.Mesh(this.cyl(0.167, 0.167, 0.012, 24), mat(tintFor(c.state, 0xf8f2e2), { roughness: 0.78 })), 0.054);
        return { g, h: 0.06 };
      }
      // SECOND bun = crown: golden hemisphere dome + sesame seeds
      const crM = mat(tintFor(c.state, 0xc9853f), { roughness: 0.52 });
      put(new THREE.Mesh(this.cyl(0.158, 0.15, 0.025, 24), crM), 0.0125);   // cut edge
      const dome = put(new THREE.Mesh(
        this.g("buncrown", () => new THREE.SphereGeometry(0.16, 22, 12, 0, Math.PI * 2, 0, Math.PI / 2)), crM), 0.025);
      dome.scale.set(1, 0.85, 1);
      const seedM = mat(0xf6ecd2, { roughness: 0.5 });
      for (let i = 0; i < 11; i++) {
        const a = i * 2.4, rr = 0.045 + (i % 3) * 0.038;
        const sx = Math.cos(a) * rr, sz = Math.sin(a) * rr;
        const sy = 0.025 + Math.sqrt(Math.max(0.001, 0.16 * 0.16 - sx * sx - sz * sz)) * 0.85;
        const seed = put(new THREE.Mesh(this.sph(0.011, 6), seedM), 0);
        seed.position.set(sx, sy, sz); seed.scale.set(1, 0.5, 1.5); seed.rotation.y = a;
      }
      return { g, h: 0.15 };
    }
    if (c.name === "meat" && !chopped) {        // patty pressed flat; sear/grill-mark texture on the face
      // cooked: same warm sear lift as _ingredient, so at overview distance the
      // browned patty reads "seared", not "burnt black"
      const emi = new THREE.Color(c.state === "cooked" ? 0x5a2400 : 0x000000);
      const topM = new THREE.MeshStandardMaterial({ map: this._meatTex(c.state),
        roughness: c.state === "cooked" ? 0.38 : 0.6, emissive: emi,
        emissiveIntensity: c.state === "cooked" ? 0.4 : 0 });
      // plain crust side (no texture): banded grill lines wrapping the rim read
      // as a SECOND patty from eye level
      const sideM = new THREE.MeshStandardMaterial({
        color: c.state === "cooked" ? 0x6b3a1c : c.state === "burnt" ? 0x1a1410 : 0xa84a3c,
        roughness: 0.55, emissive: emi, emissiveIntensity: c.state === "cooked" ? 0.35 : 0 });
      put(new THREE.Mesh(this.cyl(0.148, 0.142, 0.07, 24), [sideM, topM, topM]), 0.035);
      return { g, h: 0.07 };
    }
    if (c.name === "cheese" && !chopped) {      // slice draped over the layer below, corners overhanging
      const chM = c.state === "burnt"
        ? mat(tintFor("burnt", 0xf2c94c), { roughness: 0.5 })
        : new THREE.MeshStandardMaterial({ map: this._cheeseTex(), roughness: c.state === "cooked" ? 0.32 : 0.42 });
      const slice = put(new THREE.Mesh(this.box(0.3, 0.016, 0.3), chM), 0.008);
      slice.rotation.y = Math.PI / 4;           // corners point out past the patty edge
      for (let k = 0; k < 4; k++) {             // melty drooping corner flaps
        const a = k * Math.PI / 2;
        const pivot = new THREE.Group();
        pivot.position.set(Math.cos(a) * 0.145, 0.012, Math.sin(a) * 0.145);
        pivot.rotation.y = -a;                  // local +x now points radially outward
        const flap = new THREE.Mesh(this.box(0.11, 0.014, 0.11), chM);
        flap.position.set(0.05, -0.012, 0); flap.rotation.z = -0.65; flap.castShadow = true;
        pivot.add(flap); g.add(pivot);
      }
      return { g, h: 0.028 };
    }
    if (c.name === "lettuce" && !chopped) {     // frilly ruffle ring poking past the stack edge
      for (let i = 0; i < 9; i++) {
        const a = i / 9 * 6.28;
        const leaf = put(new THREE.Mesh(this.sph(0.1), mat(i % 2 ? 0x7ccc55 : 0x55a53c, { roughness: 0.5 })), 0);
        leaf.position.set(Math.cos(a) * 0.14, 0.02 + (i % 2) * 0.012, Math.sin(a) * 0.14);
        leaf.scale.set(1.3, 0.22, 0.85); leaf.rotation.y = -a;
      }
      return { g, h: 0.05 };
    }
    if (c.name === "tomato" && !chopped) {      // fanned slices, not a whole fruit
      const tm = mat(tintFor(c.state, 0xd93a2a), { roughness: 0.35 });
      for (let i = 0; i < 3; i++) {
        const s = put(new THREE.Mesh(this.cyl(0.105, 0.105, 0.02, 18), tm), 0.012 + i * 0.007);
        s.position.x = (i - 1) * 0.075; s.position.z = (i % 2) * 0.05 - 0.025;
      }
      return { g, h: 0.04 };
    }
    if (c.name === "onion" && !chopped) {       // separated rings
      const om = mat(tintFor(c.state, 0xf3e6c4), { roughness: 0.45 });
      const rg = this.g("onionring", () => new THREE.TorusGeometry(0.075, 0.016, 8, 20));
      for (let i = 0; i < 3; i++) {
        const r = put(new THREE.Mesh(rg, om), 0);
        r.rotation.x = Math.PI / 2; r.position.set((i - 1) * 0.06, 0.018, (i % 2) * 0.06 - 0.03);
      }
      return { g, h: 0.038 };
    }
    // everything else: the normal ingredient mesh (keeps raw/cooked/burnt and
    // chopped-dice looks), stacked by an approximate per-item height
    const m = this._ingredient(c.name, c.state);
    m.scale.setScalar(0.8); g.add(m);
    const H = { onion: 0.3, tomato: 0.28, mushroom: 0.26, fish: 0.2, rice: 0.14, egg: 0.12,
                potato: 0.22, dough: 0.14, lettuce: 0.14, meat: 0.11, bun: 0.16, cheese: 0.06,
                tortilla: 0.06, noodles: 0.16, shrimp: 0.2, steak: 0.12 };
    // the bespoke prepared looks (2026-09-09) are flatter than the raw meshes
    const HP = { "rice/cooked": 0.23, "potato/cooked": 0.15, "potato/chopped": 0.15, "steak/cooked": 0.12,
                 "steak/chopped": 0.12, "fish/cooked": 0.17, "fish/chopped": 0.08, "egg/cooked": 0.08,
                 "onion/cooked": 0.08, "tomato/cooked": 0.1, "mushroom/cooked": 0.1, "lettuce/chopped": 0.08,
                 "noodles/cooked": 0.16, "shrimp/cooked": 0.2, "dough/cooked": 0.22 };
    const lh = LAYER_H[c.name + "/" + c.state];
    const hp = lh !== undefined ? lh : HP[c.name + "/" + c.state];
    return { g, h: (hp !== undefined ? hp : (chopped ? 0.11 : (H[c.name] ?? 0.18))) * 0.8 };
  }
  _itemMesh(item) { return item.kind === "plate" ? this._plate(item) : this._ingredient(item.name, item.state); }
  _itemSig(item) { return item ? JSON.stringify(item) : "0"; }

  // ---- chef -----------------------------------------------------------
  _buildCook(p) {
    const grp = new THREE.Group();
    const col = COOK_COLORS[p.id % COOK_COLORS.length];
    // real mode: proper chef whites — the identity colour moves to the apron and
    // neckerchief (like a real brigade), instead of a cartoon-coloured jumpsuit
    const bodyCol = this.real ? 0xf2f0ea : col;
    const body = new THREE.Mesh(this.g("cookbody", () => new THREE.CapsuleGeometry(0.26, 0.34, 6, 16)), mat(bodyCol, { roughness: this.real ? 0.85 : 0.6 }));
    body.position.y = 0.5; body.castShadow = true; grp.add(body);
    const apron = new THREE.Mesh(this.box(0.34, 0.4, 0.06), mat(this.real ? 0x3a3d42 : 0xf3f1ea, { roughness: 0.75 }));
    apron.position.set(0, 0.46, 0.22); grp.add(apron);
    if (this.real) {
      const scarf = new THREE.Mesh(this.g("scarf", () => new THREE.TorusGeometry(0.155, 0.035, 8, 18)), mat(col, { roughness: 0.7 }));
      scarf.position.y = 0.78; scarf.rotation.x = Math.PI / 2; grp.add(scarf);
      const strap = new THREE.Mesh(this.box(0.3, 0.05, 0.05), mat(col, { roughness: 0.7 }));
      strap.position.set(0, 0.64, 0.23); grp.add(strap);
      for (let i = 0; i < 3; i++) {                       // double-breasted buttons
        const btn = new THREE.Mesh(this.sph(0.016, 8), mat(0x2c2e33, { roughness: 0.4 }));
        btn.position.set(0.05, 0.7 - i * 0.11, 0.255); grp.add(btn);
      }
      // side towel tucked into the apron waist — every working cook carries one
      const stw = new THREE.Mesh(this.box(0.11, 0.17, 0.03), mat(0xf1efe6, { roughness: 0.95 }));
      stw.position.set(0.24, 0.4, 0.18); stw.rotation.z = 0.12; stw.rotation.y = 0.25; grp.add(stw);
    }
    const head = new THREE.Mesh(this.sph(0.18), mat(0xf2c9a0, { roughness: 0.6 }));
    head.position.y = 0.92; head.castShadow = true; grp.add(head);
    const hatB = new THREE.Mesh(this.cyl(0.19, 0.19, 0.12, 16), mat(0xffffff, { roughness: this.real ? 0.85 : 0.5 }));
    hatB.position.y = 1.04; grp.add(hatB);
    const puff = new THREE.Mesh(this.sph(0.16), mat(0xffffff, { roughness: this.real ? 0.85 : 0.5 }));
    puff.position.y = 1.16; grp.add(puff);
    if (this.real) { puff.scale.set(1, 0.72, 1); puff.position.y = 1.12; }   // lower, flatter toque
    // arms (real mode: attached at the shoulder, with skin-tone hands)
    const arm = (sx) => {
      const a = new THREE.Mesh(this.g("arm", () => new THREE.CapsuleGeometry(0.07, 0.22, 4, 8)), mat(this.real ? 0xf2f0ea : col, { roughness: this.real ? 0.85 : 0.6 }));
      a.position.set(sx * (this.real ? 0.27 : 0.3), this.real ? 0.56 : 0.5, this.real ? 0.12 : 0.16);
      a.rotation.x = -0.7; grp.add(a);
      if (this.real) {
        const hand = new THREE.Mesh(this.sph(0.055, 10), mat(0xf2c9a0, { roughness: 0.6 }));
        hand.position.y = -0.18; a.add(hand);            // rides the arm during work pulses
      }
      return a;
    };
    const aL = arm(-1), aR = arm(1);
    // legs (real mode: dark work shoes attached as feet)
    const leg = (sx) => {
      const l = new THREE.Mesh(this.g("leg", () => new THREE.CapsuleGeometry(0.09, 0.18, 4, 8)), mat(0x33343c, { roughness: 0.7 }));
      l.position.set(sx * 0.12, 0.18, 0); grp.add(l);
      if (this.real) {
        const shoe = new THREE.Mesh(this.box(0.11, 0.06, 0.17), mat(0x17181c, { roughness: 0.45 }));
        shoe.position.set(0, -0.15, 0.03); l.add(shoe);
      }
      return l;
    };
    const lL = leg(-1), lR = leg(1);
    // eyes
    const eye = (sx) => { const e = new THREE.Mesh(this.sph(0.025, 8), mat(0x222228)); e.position.set(sx * 0.07, 0.93, 0.16); grp.add(e); };
    eye(-1); eye(1);
    // number label sprite
    const spr = this._textSprite(String(p.id + 1), col);
    spr.position.y = 1.45; spr.scale.set(0.4, 0.4, 1); grp.add(spr);
    // real mode: soft blob under the cook — grounds the character everywhere,
    // even where the single shadow light barely reaches
    let blob = null;
    if (this.real) { blob = this._contactShadow(0, 0.008, 0, 0.4, 0.3); grp.add(blob); }
    // an upper-body pivot so the cook can visibly lean in to interact
    grp.userData = { legs: [lL, lR], arms: [aL, aR], torso: body, head, badge: spr, heldSig: "0", held: null, reach: 0, pop: 0, blob };
    this.dynGroup.add(grp);
    return grp;
  }

  // ---- dynamic update --------------------------------------------------
  update(dt) {
    this._t += dt;
    if (!this.state) { return; }
    const st = this.state;
    const seenItems = new Set();
    this._updateCamera(dt);
    this._updateEgoBars();

    // cooks
    const seenCooks = new Set();
    const interp = Math.min(1, dt * 12);
    for (const p of st.players) {
      seenCooks.add(p.id);
      let c = this.cooks[p.id];
      if (!c) { c = this.cooks[p.id] = { grp: this._buildCook(p), x: p.x, y: p.y, phase: Math.random() * 6 }; }
      if (this.filmLerp != null) {
        // Re-baseline at the START of every tick, not only when the target tile changes.
        // Keying on the target alone replays the previous move on every tick the cook
        // stands still (turn / interact / idle): the baseline stays one cell back, so
        // alpha sweeps the cook across that cell again and again. The caller advances
        // alpha upward within a tick, so a DECREASE in alpha is the new-tick edge.
        // Strictly less, never <=: one captured frame calls update() more than once (the
        // top-down pass and then the ego pass), and those repeats carry the SAME alpha.
        // Treating an equal alpha as a new tick re-baselines mid-frame, which compounds
        // the lerp and turns the intended constant walk into an ease-out that arrives
        // early and then stands still for the rest of the tick.
        const newTick = (c.tx !== p.x || c.ty !== p.y ||
                         c._la === undefined || this.filmLerp < c._la);
        if (newTick) { c.x0 = c.x; c.y0 = c.y; c.tx = p.x; c.ty = p.y; }
        c._la = this.filmLerp;
        c.x = lerp(c.x0, p.x, this.filmLerp); c.y = lerp(c.y0, p.y, this.filmLerp);
      } else {
        c._la = undefined;                            // leaving film mode resets the edge
        c.tx = p.x; c.ty = p.y;                       // keep film mode's baseline fresh
        c.x = lerp(c.x, p.x, interp); c.y = lerp(c.y, p.y, interp);
      }
      c.dir = p.dir_name;
      const moving = Math.abs(c.x - p.x) > 0.02 || Math.abs(c.y - p.y) > 0.02;
      c.phase += dt * (moving ? 11 : 2.5);
      const bob = moving ? Math.abs(Math.sin(c.phase)) * 0.06 : Math.sin(c.phase) * 0.015;
      const wp = this.toWorld(c.x, c.y);
      c.grp.position.set(wp.x, bob, wp.z);
      c.grp.rotation.y = THREE.MathUtils.lerp(c.grp.rotation.y, DIR_YAW[p.dir_name] || 0, 0.3);
      const sw = moving ? Math.sin(c.phase) * 0.5 : 0;
      const ud = c.grp.userData;
      ud.legs[0].rotation.x = sw; ud.legs[1].rotation.x = -sw;
      if (ud.blob) ud.blob.position.y = 0.014 - bob;   // blob stays ON the floor while the cook bobs

      // --- interaction animation: a strong, timed "work" pulse -------
      // Triggered once per state where the cook interacts (so even a single
      // pick-up/place/serve plays a full, visible motion), and re-triggered
      // every tick while chopping/cooking so it reads as repeated action.
      const facing = DIR_VEC[p.dir_name] || [0, 0];
      if (this._pulse[p.id]) {
        ud.actT = 0.5;                       // (re)start a half-second work pulse
        this._useBurst(wp.x + facing[0] * 0.7, wp.z + facing[1] * 0.7);
        delete this._pulse[p.id];
      }
      ud.actT = Math.max(0, (ud.actT || 0) - dt);
      // ramp up fast, ease down — two pumps over the pulse for a chop/grab feel
      const a = ud.actT / 0.5;                                   // 1 -> 0
      const env = Math.sin(Math.min(1, a) * Math.PI);            // 0..1..0 envelope
      const pump = Math.abs(Math.sin(a * Math.PI * 2));          // two strokes
      const act = env * (0.55 + 0.45 * pump);                    // strong overall

      ud.torso.rotation.x = act * 0.6;                           // big lean toward station
      ud.head.position.z = 0.02 + act * 0.14;
      // arms swing out and down toward the station (clear reaching/chopping)
      const armAng = -0.7 - act * 1.5;
      ud.arms[0].rotation.x = armAng; ud.arms[1].rotation.x = armAng;
      // the whole cook lunges toward the tile it faces, then settles back
      c.grp.position.set(wp.x + facing[0] * act * 0.22, bob, wp.z + facing[1] * act * 0.22);

      // held item
      const sig = this._itemSig(p.holding);
      if (sig !== ud.heldSig) {
        if (ud.held) c.grp.remove(ud.held);
        ud.held = null;
        if (p.holding) {
          const m = this._itemMesh(p.holding);
          c.grp.add(m); ud.held = m;
          ud.pop = 1;                                  // pop-in when just acquired
          // a quick puff at the faced tile so the grab reads as a deliberate
          // TAKE/transfer, not a teleport (round-14: pickups felt "magical")
          this._useBurst(wp.x + facing[0] * 0.5, wp.z + facing[1] * 0.5);
        }
        ud.heldSig = sig;
      }
      if (ud.held) {
        ud.pop = lerp(ud.pop, 0, Math.min(1, dt * 5));   // ~0.2s grab arc
        // hand position (with the work-thrust toward the station)
        const hy = 0.5 + act * 0.12, hz = 0.4 + act * 0.28;
        // when just acquired (pop~1) the item starts out at the faced station
        // and travels into the hands, so a viewer sees it being TAKEN
        ud.held.position.set(0, lerp(hy, 0.62, ud.pop), lerp(hz, 0.85, ud.pop));
        ud.held.scale.setScalar(0.9);
        // a hot, finished dish in hand visibly steams
        if (p.holding.kind === "plate" && (p.holding.contents || []).some(it => it.state === "cooked") && Math.random() < 0.2) {
          this._spawn(wp.x + facing[0] * 0.42, 0.95, wp.z + facing[1] * 0.42, "steam");
        }
      }
    }
    for (const id in this.cooks) if (!seenCooks.has(+id)) { this.dynGroup.remove(this.cooks[id].grp); delete this.cooks[id]; }

    // station dynamic content
    for (const s of st.stations) {
      const key = s.x + "," + s.y;
      const sig = JSON.stringify(s);
      let d = this.stationDyn[key];
      if (!d) d = this.stationDyn[key] = { grp: new THREE.Group(), sig: "" };
      if (d.sig !== sig) {
        d.grp.clear();
        this._stationContent(d.grp, s);
        const wp = this.toWorld(s.x, s.y); d.grp.position.set(wp.x, CH + 0.07, wp.z);
        if (!d.added) { this.dynGroup.add(d.grp); d.added = true; }
        d.sig = sig;
      }
      this._stationFx(s);
    }

    // loose counter items
    const present = new Set();
    for (const o of st.objects) {
      const key = o.x + "," + o.y; present.add(key);
      const sig = this._itemSig(o.item);
      let it = this.items[key];
      if (!it || it.sig !== sig) {
        if (it) this.dynGroup.remove(it.grp);
        const m = this._itemMesh(o.item); const wp = this.toWorld(o.x, o.y);
        // real mode: items SIT on the steel top (the legacy 0.12 hover reads as
        // floating from eye level); classic keeps its exact legacy placement
        m.position.set(wp.x, this.real ? CH + 0.055 : CH + 0.12, wp.z); this.dynGroup.add(m);
        this.items[key] = { grp: m, sig };
      }
    }
    for (const key in this.items) if (!present.has(key)) { this.dynGroup.remove(this.items[key].grp); delete this.items[key]; }

    // simmering pots: swirl the liquid surface texture
    if (this.real && this._simmerTexC) {
      this._simmerTexC.rotation = this._t * 0.5;
      this._simmerTexC.offset.x = Math.sin(this._t * 1.3) * 0.03;
    }
    // "ORDER UP" neon: dim idle glow, bright flash (with a flicker) on delivery
    if (this.real && this._neonMat) {
      this._neonFlash = Math.max(0, (this._neonFlash || 0) - dt);
      const f = this._neonFlash;
      this._neonMat.emissiveIntensity = f > 0
        ? 1.4 + 1.6 * Math.min(1, f / 0.6) * (0.82 + 0.18 * Math.sin(this._t * 34))
        : 0.55 + 0.05 * Math.sin(this._t * 2.1);
    }
    this._updateParticles(dt);
    this._updateFlyers(dt);
    this.composer.render();
  }

  _stationContent(grp, s) {
    if ((s.type === "pot" || s.type === "oven") && s.contents && s.contents.length) {
      // show the actual ingredient pieces (what & how many is legible). The
      // oven is enclosed, so its contents sit on a visible tray on top of it.
      // both stockpots (real enamel + classic steel) are tall, so contents ride
      // high to stay visible above the rim; the classic oven's tray sits above
      // its control band
      const baseY = s.type === "oven" ? (this.real ? 0.66 : 0.74)
        : s.type === "pot" ? (this.real ? 0.5 : 0.56) : 0.27;
      const n = s.contents.length;
      s.contents.forEach((ing, i) => {
        const m = this._ingredient(ing.name, ing.state); m.scale.setScalar(0.62);
        const ang = n > 1 ? i / n * 6.28 : 0, rr = n > 1 ? 0.11 : 0;
        m.position.set(Math.cos(ang) * rr, baseY + 0.04, Math.sin(ang) * rr); grp.add(m);
      });
      if (s.status === "cooking" || s.status === "cooked" || s.status === "burnt") {
        const col = this._mix(s.contents, s.status);
        const liq = new THREE.Mesh(this.cyl(0.27, 0.25, 0.04, 20),
          new THREE.MeshStandardMaterial({ color: col, roughness: 0.4, transparent: true, opacity: this.real ? 0.75 : 0.5,
            map: this.real ? this._simmerT() : null,
            emissive: new THREE.Color(col).multiplyScalar(s.status === "burnt" ? 0 : 0.12) }));
        liq.position.y = baseY - 0.04; grp.add(liq);
      }
    } else if (s.type === "pan" && s.contents && s.contents.length) {
      s.contents.forEach((ing, i) => { const m = this._ingredient(ing.name, ing.state); m.position.set((i - (s.contents.length - 1) / 2) * 0.18, 0.11, 0); m.scale.setScalar(0.92); grp.add(m); });
    } else if (s.type === "fryer" && s.contents && s.contents.length) {
      // food rides in the wire basket above the oil well
      s.contents.forEach((ing, i) => {
        const m = this._ingredient(ing.name, ing.state);
        m.position.set(-0.06 + (i - (s.contents.length - 1) / 2) * 0.16, 0.38, 0);
        m.scale.setScalar(0.7); grp.add(m);
      });
      if (s.status === "cooking" || s.status === "cooked" || s.status === "burnt") {
        // rolling-boil oil: brighter, slightly raised surface while frying
        const boil = new THREE.Mesh(this.box(0.52, 0.02, 0.42),
          new THREE.MeshStandardMaterial({ color: s.status === "burnt" ? 0x5a3208 : 0xf2c258,
            roughness: 0.15, transparent: true, opacity: 0.9,
            emissive: s.status === "burnt" ? 0x200e00 : 0x9a6a14, emissiveIntensity: 0.5 }));
        boil.position.y = 0.375; grp.add(boil);
        if (s.status === "cooking") {
          // BUBBLES popping around the basket (content rebuilds as progress
          // ticks, so they jitter — a live rolling boil)
          const bub = new THREE.MeshStandardMaterial({ color: 0xffe9b0, emissive: 0xd9a840,
            emissiveIntensity: 0.8, roughness: 0.2 });
          for (let i = 0; i < 6; i++) {
            const a = i * 2.4 + (s.progress || 0) * 21;
            const b = new THREE.Mesh(this.sph(0.018 + 0.012 * ((i * 7 + Math.floor((s.progress || 0) * 40)) % 3)), bub);
            b.position.set(-0.06 + Math.cos(a) * 0.2, 0.395, Math.sin(a) * 0.15);
            grp.add(b);
          }
        }
      }
    } else if (s.type === "cutting_board" && s.item) {
      const m = this._ingredient(s.item.name, s.item.state); m.position.set(0, 0.1, 0); m.scale.setScalar(1.0); grp.add(m);
    } else if (s.type === "sink" && (s.dirty > 0 || s.clean_ready > 0)) {
      const water = new THREE.Mesh(this.box(0.56, 0.04, 0.42), new THREE.MeshStandardMaterial({ color: 0x6fc3e8, transparent: true, opacity: 0.85, roughness: 0.15, metalness: 0.3, emissive: 0x1d6f9c, emissiveIntensity: 0.3 }));
      water.position.y = 0.16; grp.add(water);
    }
    // progress ring — a dim full-circle track + a bright filled arc, on EVERY
    // station that is actively processing, so "how far along" is always clear
    let frac = -1, color = 0x66d98a;
    if (s.status === "cooking") frac = s.progress;
    else if (s.status === "cooked") frac = 1;
    else if (s.type === "cutting_board" && s.item && !s.done) { frac = s.progress; color = 0xe8c14a; }
    else if (s.type === "sink" && s.progress > 0) { frac = s.progress; color = 0x6fc3e8; }
    if (frac >= 0) {
      const yR = 1.05;   // well above the station & its contents
      const track = new THREE.Mesh(this.g("ringtrack", () => new THREE.TorusGeometry(0.36, 0.07, 10, 36)),
        new THREE.MeshStandardMaterial({ color: 0x1a1a1a, emissive: 0x0a0a0a, transparent: true, opacity: 0.7 }));
      track.rotation.x = -Math.PI / 2; track.position.y = yR; grp.add(track);
      // a bold UPRIGHT progress bar standing above the station — unmistakable
      // "how far along" from the angled follow-camera, on EVERY active station
      const BW = 0.62, BH = 0.18, byBase = 1.32;
      const btrack = new THREE.Mesh(this.box(BW + 0.06, BH + 0.06, 0.05), new THREE.MeshStandardMaterial({ color: 0x101216, emissive: 0x050608, transparent: true, opacity: 0.92 }));
      btrack.position.y = byBase; grp.add(btrack);
      const fillW = Math.max(0.001, BW * Math.min(1, frac));
      const fill = new THREE.Mesh(this.box(fillW, BH, 0.07), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 2.0 }));
      fill.position.set(-BW / 2 + fillW / 2, byBase, 0.01); grp.add(fill);
      if (frac > 0.001) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.1, 10, 36, Math.PI * 2 * frac),
          new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.8 }));
        ring.rotation.x = -Math.PI / 2; ring.position.y = yR; grp.add(ring);
        if (frac >= 1) {  // a clear "done" tick floats up when ready
          const done = new THREE.Mesh(this.sph(0.07), new THREE.MeshStandardMaterial({ color: 0x9af0bf, emissive: 0x4cd07a, emissiveIntensity: 1.5 }));
          done.position.y = yR; grp.add(done);
          const big = new THREE.Mesh(this.box(BW + 0.06, BH + 0.06, 0.08), new THREE.MeshStandardMaterial({ color: 0x9af0bf, emissive: 0x4cd07a, emissiveIntensity: 2.2 }));
          big.position.y = byBase; grp.add(big);  // bar flashes solid green when READY
        }
      }
    }
  }

  // ---- EGO progress billboards -----------------------------------------
  // The third-person progress ring/bar sit ~0.4-0.7 ABOVE the ego eye height and
  // fall outside the first-person frustum right when you stand at the station.
  // In ego mode every actively-processing station gets a camera-facing sprite bar
  // at chest height instead - never occluded, readable from any approach angle.
  _egoBarTex(frac, color, warn) {
    const cv = document.createElement("canvas"); cv.width = 128; cv.height = 28;
    const g = cv.getContext("2d");
    g.fillStyle = "rgba(8,9,12,.85)"; g.fillRect(0, 0, 128, 28);
    g.strokeStyle = warn ? "#ff5a3c" : "rgba(255,255,255,.35)";
    g.lineWidth = warn ? 4 : 2; g.strokeRect(1, 1, 126, 26);
    const c = "#" + color.toString(16).padStart(6, "0");
    g.fillStyle = c; g.fillRect(4, 4, Math.max(2, 120 * Math.min(1, frac)), 20);
    const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }
  _updateEgoBars() {
    if (!this._egoBars) this._egoBars = {};
    const active = new Set();
    if (this.ego !== null && this.state) {
      for (const st of this.state.stations) {
        let frac = -1, color = 0x66d98a;
        if (st.status === "cooking") frac = st.progress;
        else if (st.status === "cooked") frac = 1;
        else if (st.type === "cutting_board" && st.item && !st.done) { frac = st.progress; color = 0xe8c14a; }
        else if (st.type === "sink" && st.progress > 0) { frac = st.progress; color = 0x6fc3e8; }
        if (frac < 0) continue;
        if (frac >= 1) color = 0x9af0bf;
        const warn = !!st.smoking;
        if (warn) color = 0xff8a3c;
        const key = st.x + "," + st.y;
        active.add(key);
        let b = this._egoBars[key];
        if (!b) {
          const spr = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, transparent: true }));
          spr.scale.set(0.52, 0.115, 1); spr.renderOrder = 996;
          this.scene.add(spr);
          b = this._egoBars[key] = { spr, sig: "" };
        }
        const sig = Math.round(frac * 50) + "_" + color + "_" + (warn ? 1 : 0);
        if (b.sig !== sig) {
          if (b.spr.material.map) b.spr.material.map.dispose();
          b.spr.material.map = this._egoBarTex(frac, color, warn);
          b.spr.material.needsUpdate = true;
          b.sig = sig;
        }
        const wp = this.toWorld(st.x, st.y);
        b.spr.position.set(wp.x, CH + 0.55, wp.z);   // chest height: inside the ego frustum
        b.spr.visible = true;
      }
    }
    for (const key in this._egoBars) {
      if (!active.has(key)) {
        this.scene.remove(this._egoBars[key].spr);
        if (this._egoBars[key].spr.material.map) this._egoBars[key].spr.material.map.dispose();
        delete this._egoBars[key];
      }
    }
  }

  _stationFx(s) {
    const wp = this.toWorld(s.x, s.y);
    const top = CH + 0.5;
    if (s.type === "oven" && this._ovenWins) {
      // the oven window's inner glow tracks its status: dark when idle, warm
      // while baking, bright ember when done/burnt
      const m = this._ovenWins[wp.x.toFixed(2) + "," + wp.z.toFixed(2)];
      if (m) m.emissiveIntensity = s.status === "cooking" ? 1.5
        : s.status === "cooked" ? 0.9 : s.status === "burnt" ? 2.2 : 0.18;
    }
    if ((s.type === "pot" || s.type === "oven" || s.type === "pan" || s.type === "fryer") && s.status === "cooking" && Math.random() < 0.3)
      this._spawn(wp.x, top, wp.z, "steam");
    if (s.status === "cooked" && !s.smoking && Math.random() < 0.4) this._spawn(wp.x, top, wp.z, "steam");  // cooked = clearly steaming
    if (s.smoking) {
      // WARNING phase: grey smoke column, density/size ramp with how close the fire is
      const p = s.burn_progress || 0;
      this._spawn(wp.x, top, wp.z, "smoke", 1.4 + 2.2 * p);
      if (Math.random() < 0.5 + 0.5 * p) this._spawn(wp.x, top + 0.35, wp.z, "smoke", 1.8 + 2.2 * p);
      if (p > 0.5) this._spawn(wp.x, top + 0.7, wp.z, "smoke", 2.6 + 1.6 * p);
    }
    if (s.status === "burnt") {
      // FIRE: unmistakable — flames + thick smoke column
      if (Math.random() < 0.9) this._spawn(wp.x, top, wp.z, "fire", 1.6);
      if (Math.random() < 0.7) this._spawn(wp.x, top + 0.4, wp.z, "smoke", 2.4);
    }
  }

  _mix(contents, status) {
    // average the per-ingredient colours AT THEIR CURRENT STATE, so a cooked
    // soup looks browned (not raw-coloured) and a burnt one looks charred.
    let r = 0, g = 0, b = 0;
    for (const c of contents) {
      const base = (ING[c.name] || { c: 0xcccccc }).c;
      const col = status === "burnt" ? tintFor("burnt", base)
        : c.state === "cooked" ? this._soupColor(c.name) : tintFor(c.state, base);
      r += (col >> 16) & 255; g += (col >> 8) & 255; b += col & 255;
    }
    const n = contents.length || 1;
    return (Math.round(r / n) << 16) | (Math.round(g / n) << 8) | Math.round(b / n);
  }

  // ---- particles (sprites) --------------------------------------------
  _spawn(x, y, z, kind, scale = 1.0) {
    // real mode: keep smoke wispy — the huge dark blob read as a floating sphere
    if (this.real && kind === "smoke") scale = Math.min(scale, 1.4);
    const tex = kind === "fire" ? this._fireTex
      : kind === "smoke" ? (this.real ? this._smokeTexReal : this._smokeTex) : this._steamTex;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, blending: kind === "fire" ? THREE.AdditiveBlending : THREE.NormalBlending }));
    spr.position.set(x + (Math.random() - .5) * 0.3, y, z + (Math.random() - .5) * 0.3);
    spr.material.opacity = kind === "fire" ? 0.85 : kind === "smoke" ? (this.real ? 0.42 : 0.8) : 0.4;
    spr.scale.setScalar(scale);
    this.scene.add(spr);
    this.particles.push({ spr, life: 1, vy: kind === "fire" ? 1.2 : kind === "smoke" ? 1.0 : 0.8, kind, sc: scale });
  }
  _delivery(delta) {
    if (this.real) this._neonFlash = 3.0;   // "ORDER UP" neon lights up when a dish is served
    // sparkle burst at the serving pass...
    for (const s of this.state.stations) if (s.type === "serving") {
      const wp = this.toWorld(s.x, s.y);
      for (let i = 0; i < 18; i++) {
        const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: this._sparkTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
        spr.position.set(wp.x, CH + 0.7, wp.z); spr.scale.setScalar(0.2); this.scene.add(spr);
        const a = Math.random() * 6.28, sp = 1 + Math.random() * 2.2;
        this.particles.push({ spr, life: 1, vy: Math.sin(a) * sp * 0.4 + 1.6, vx: Math.cos(a) * sp * 0.6, vz: Math.sin(a) * sp * 0.6, kind: "spark", grav: -4 });
      }
    }
    // ...and a brief "ORDER DONE +N" banner anchored at the camera focus, so it
    // is in-frame regardless of where the pass is. Short-lived so it doesn't
    // linger over the cook (round-13: a stale banner obscured the character).
    const f = this._camFocus || new THREE.Vector3();
    // [removed] text FX disabled: a "+N" banner is engine feedback in TEXT form — the
    // supervisor must read outcomes from the world, not from a HUD popup
    // this._popText(f.x, 1.8, f.z, "✓ +" + (delta || ""), "#5fe39a", 1.5, 2.4);
  }
  _failFeedback() {
    const f = this._camFocus || new THREE.Vector3();
    // [removed] same rationale as the "+N" banner
    // this._popText(f.x, 1.7, f.z, "✗ WRONG", "#ff6a5a", 2.0, 2.2);
  }
  _popText(x, y, z, text, color, life, sc) {
    sc = sc || 1.9;
    const C = document.createElement("canvas"); C.width = 512; C.height = 160;
    const ctx = C.getContext("2d");
    ctx.font = "bold 110px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.lineWidth = 14; ctx.strokeStyle = "rgba(0,0,0,.78)"; ctx.strokeText(text, 256, 80);
    ctx.fillStyle = color; ctx.fillText(text, 256, 80);
    const tex = new THREE.CanvasTexture(C); tex.colorSpace = THREE.SRGBColorSpace;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    spr.position.set(x, y, z); spr.scale.set(sc, sc * 0.31, 1); this.scene.add(spr);
    this.particles.push({ spr, life: life || 2.0, vy: 1.3, kind: "pop" });  // rises up & out quickly
  }
  // A small, brief puff at the worked tile — a subtle contact cue. The real
  // signal is the cook's lunge + the visibly changing item, NOT a glowing flash
  // (round-6: the old bright ring read as a "magic light" that hid the action).
  _useBurst(x, z) {
    const top = CH + 0.32;
    // in first-person the burst happens right at the camera — keep it subtle
    // there so it doesn't white out the view (Gemini review: glare on pickup)
    const ego = this.ego !== null;
    for (let i = 0; i < 3; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this._steamTex, transparent: true, depthWrite: false, blending: THREE.NormalBlending, color: 0xfff0d0 }));
      s.position.set(x + (Math.random() - .5) * 0.2, top, z + (Math.random() - .5) * 0.2);
      s.scale.setScalar(ego ? 0.08 : 0.14); s.material.opacity = ego ? 0.25 : 0.45; this.scene.add(s);
      const a = Math.random() * 6.28;
      this.particles.push({ spr: s, life: 0.5, vx: Math.cos(a) * 0.4, vz: Math.sin(a) * 0.4, vy: 0.6, kind: "puff", sc: ego ? 0.35 : 1 });
    }
  }
  _flyItem(item, from, to) {
    if (!item) return;
    const m = item.kind === "plate" ? this._plate(item) : this._ingredient(item.name, item.state);
    m.scale.setScalar(0.8); m.position.set(from.x, from.y, from.z);
    this.scene.add(m);
    this._flyers.push({ m, from, to, t: 0, dur: 0.32 });
  }
  _updateFlyers(dt) {
    const alive = [];
    for (const f of this._flyers) {
      f.t += dt / f.dur;
      if (f.t >= 1) { this.scene.remove(f.m); continue; }
      const e = f.t;
      f.m.position.set(
        f.from.x + (f.to.x - f.from.x) * e,
        f.from.y + (f.to.y - f.from.y) * e + Math.sin(e * Math.PI) * 0.3,  // little arc
        f.from.z + (f.to.z - f.from.z) * e);
      f.m.scale.setScalar(0.8);
      alive.push(f);
    }
    this._flyers = alive;
  }
  _updateParticles(dt) {
    const alive = [];
    for (const p of this.particles) {
      p.life -= dt * (p.kind === "spark" ? 1.5 : p.kind === "ring" ? 2.6 : p.kind === "pop" ? 0.5 : 1.0);
      p.spr.position.y += (p.vy || 0) * dt;
      if (p.vx) p.spr.position.x += p.vx * dt;
      if (p.vz) p.spr.position.z += p.vz * dt;
      if (p.grav) p.vy += p.grav * dt;
      if (p.kind === "pop") {
        p.spr.material.opacity = Math.min(1, p.life * 1.6);    // float up & fade
      } else if (p.kind === "ring") {
        p.spr.scale.setScalar(0.3 + (1 - p.life) * 1.1);
        p.spr.material.opacity = Math.max(0, p.life) * 0.9;
      } else {
        const baseScale = (p.kind === "spark" ? 0.16 : 0.3 + (1 - p.life) * 0.5) * (p.sc || 1);
        p.spr.scale.setScalar(baseScale);
        p.spr.material.opacity = Math.max(0, p.life) * (p.kind === "fire" ? 0.8 : p.kind === "spark" ? 1 : p.kind === "smoke" ? 0.85 : 0.4);
      }
      if (p.life > 0) alive.push(p); else this.scene.remove(p.spr);
    }
    this.particles = alive;
  }
  _ringTexture() {
    const C = document.createElement("canvas"); C.width = C.height = 64; const ctx = C.getContext("2d");
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 7; ctx.beginPath(); ctx.arc(32, 32, 24, 0, 6.28); ctx.stroke();
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }

  // ---- textures --------------------------------------------------------
  _floorTex(w, h) {
    const C = document.createElement("canvas"); const px = 64; C.width = w * px; C.height = h * px;
    const ctx = C.getContext("2d");
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      ctx.fillStyle = ((x + y) % 2) ? "#b8ac90" : "#c4b89c";
      ctx.fillRect(x * px, y * px, px, px);
      ctx.strokeStyle = "rgba(120,105,82,.55)"; ctx.lineWidth = 2; ctx.strokeRect(x * px, y * px, px, px);
      // subtle per-tile speckle for texture
      ctx.fillStyle = "rgba(90,78,60,.06)";
      for (let k = 0; k < 18; k++) ctx.fillRect(x * px + Math.random() * px, y * px + Math.random() * px, 2, 2);
    }
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
  }
  _woodTex(base, dark) {
    const C = document.createElement("canvas"); C.width = C.height = 128; const ctx = C.getContext("2d");
    const hex = (h) => "#" + h.toString(16).padStart(6, "0");
    ctx.fillStyle = hex(base); ctx.fillRect(0, 0, 128, 128);
    ctx.strokeStyle = hex(dark); ctx.globalAlpha = 0.35;
    for (let i = 0; i < 22; i++) {
      ctx.lineWidth = 1 + (i % 3); ctx.beginPath();
      const yy = (i / 22) * 128 + Math.sin(i) * 3;
      ctx.moveTo(0, yy);
      for (let x = 0; x <= 128; x += 16) ctx.lineTo(x, yy + Math.sin((x + i) * 0.12) * 4);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  // cached procedural food-surface textures (real mode only)
  _foodTex(key, draw) {
    if (this._foodTexCache[key]) return this._foodTexCache[key];
    const C = document.createElement("canvas"); C.width = C.height = 128;
    draw(C.getContext("2d"));
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace;
    this._foodTexCache[key] = t; return t;
  }
  _meatTex(state) {
    const st = state === "cooked" ? "cooked" : state === "burnt" ? "burnt" : "raw";
    return this._foodTex("meat_" + st, (ctx) => {
      const base = st === "raw" ? "#b04a3c" : st === "cooked" ? "#6d3a1d" : "#191310";
      ctx.fillStyle = base; ctx.fillRect(0, 0, 128, 128);
      // ground-meat grain: many small blobs a shade lighter/darker
      for (let i = 0; i < 260; i++) {
        const x = (i * 37) % 128, y = (i * 61 + (i * i) % 17) % 128, r = 1 + (i % 3);
        ctx.fillStyle = st === "raw"
          ? (i % 5 ? "rgba(210,120,105,.5)" : "rgba(235,220,210,.55)")   // meat + fat flecks
          : st === "cooked" ? (i % 4 ? "rgba(140,84,44,.5)" : "rgba(52,24,8,.6)")
          : "rgba(60,52,46,.5)";
        ctx.beginPath(); ctx.arc(x, y, r, 0, 6.28); ctx.fill();
      }
      if (st === "cooked") {                      // char grill lines
        ctx.strokeStyle = "rgba(30,12,4,.8)"; ctx.lineWidth = 7;
        for (let k = -1; k < 5; k++) {
          ctx.beginPath(); ctx.moveTo(-10, k * 34); ctx.lineTo(138, k * 34 + 30); ctx.stroke();
        }
      }
    });
  }
  _cheeseTex() {
    return this._foodTex("cheese", (ctx) => {
      ctx.fillStyle = "#f4c94e"; ctx.fillRect(0, 0, 128, 128);
      const holes = [[24, 30, 9], [70, 22, 6], [102, 58, 10], [44, 78, 7], [88, 100, 8], [18, 108, 5], [62, 52, 4]];
      for (const [x, y, r] of holes) {
        ctx.fillStyle = "#d9a92c"; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.28); ctx.fill();
        ctx.fillStyle = "#c2941f"; ctx.beginPath(); ctx.arc(x + r * 0.2, y + r * 0.25, r * 0.72, 0, 6.28); ctx.fill();
      }
    });
  }
  _softTex(hex) {
    const C = document.createElement("canvas"); C.width = C.height = 64; const ctx = C.getContext("2d");
    const col = "#" + hex.toString(16).padStart(6, "0");
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, col); g.addColorStop(0.4, col); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(32, 32, 32, 0, 6.28); ctx.fill();
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  _textSprite(str, color) {
    const C = document.createElement("canvas"); C.width = C.height = 64; const ctx = C.getContext("2d");
    ctx.fillStyle = "#" + color.toString(16).padStart(6, "0"); ctx.beginPath(); ctx.arc(32, 32, 26, 0, 6.28); ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 4; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.font = "bold 38px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(str, 32, 34);
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace;
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
  }

  resize(W, H) {
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(W, H, false);
    this.composer.setSize(W, H);
    this.bloom.setSize(W, H);
    this.fit(W, H);
  }
}

window.KitchenRenderer = KitchenRenderer3D;
window.dispatchEvent(new Event("cooksim-renderer-ready"));
