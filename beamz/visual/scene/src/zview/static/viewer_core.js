import * as THREE from "https://esm.sh/three@0.160.1";
import { OrbitControls } from "https://esm.sh/three@0.160.1/examples/jsm/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "https://esm.sh/three@0.160.1/examples/jsm/geometries/RoundedBoxGeometry.js";

const THEMES = {
  dark: {
    sceneBackground: "#1a1a1a",
    objectOutline: "#ffffff",
    gizmo: {
      cubeColor: "#2a2a2a",
      cubeRoughness: 0.62,
      cubeMetalness: 0.08,
      cubeOpacity: 0.9,
      faceText: "#f5f5f5",
      faceOpacity: 0.96,
      axisX: "#d4d4d4",
      axisY: "#a3a3a3",
      axisZ: "#737373",
    },
    measurement: {
      edge: 0x737373,
      axis: 0xd4d4d4,
      tick: 0xa3a3a3,
      tickText: "#d4d4d4",
      axisText: "#f5f5f5",
    },
  },
  light: {
    sceneBackground: "#f8fafc",
    objectOutline: "#000000",
    gizmo: {
      cubeColor: "#f4f4f5",
      cubeRoughness: 0.55,
      cubeMetalness: 0.0,
      cubeOpacity: 0.84,
      faceText: "#334155",
      faceOpacity: 0.92,
      axisX: "#404040",
      axisY: "#737373",
      axisZ: "#a3a3a3",
    },
    measurement: {
      edge: 0x6b7280,
      axis: 0x374151,
      tick: 0x4b5563,
      tickText: "#374151",
      axisText: "#1f2937",
    },
  },
};

function colorValue(input, fallback = "#4f46e5") {
  try {
    return new THREE.Color(input || fallback);
  } catch {
    return new THREE.Color(fallback);
  }
}

function normalizeVec3(values, fallback) {
  const source = Array.isArray(values) ? values : fallback;
  const [x, y, z] = source;
  return new THREE.Vector3(Number(x), Number(y), Number(z));
}

function materialSignature(spec, kind) {
  return JSON.stringify({
    kind,
    color: spec?.color || null,
    opacity: Number(spec?.opacity ?? 1),
    wireframe: Boolean(spec?.wireframe),
    metalness: Number(spec?.metalness ?? 0),
    roughness: Number(spec?.roughness ?? 0.85),
    emissive: spec?.emissive || "#000000",
  });
}

function displayOrder(spec, fallbackOrder = 0) {
  return Number(spec?.metadata?.display_order ?? fallbackOrder);
}

function makeMaterial(spec, clippingPlanes, kind, materialCache) {
  const opacity = Math.max(0, Math.min(1, Number(spec?.opacity ?? 1)));
  const signature = materialSignature(spec, kind);
  if (materialCache?.has(signature)) {
    return materialCache.get(signature);
  }
  const material = new THREE.MeshPhysicalMaterial({
    color: colorValue(spec?.color),
    transparent: opacity < 1,
    opacity,
    wireframe: Boolean(spec?.wireframe),
    visible: spec?.visible !== false,
    metalness: Number(spec?.metalness ?? 0),
    roughness: Number(spec?.roughness ?? 0.85),
    emissive: colorValue(spec?.emissive, "#000000"),
    side: THREE.DoubleSide,
    clippingPlanes,
    depthWrite: opacity >= 0.999,
  });
  material.polygonOffset = kind === "plane";
  material.polygonOffsetFactor = kind === "plane" ? -1 : 0;
  material.polygonOffsetUnits = kind === "plane" ? -1 : 0;
  materialCache?.set(signature, material);
  return material;
}

function orientToNormal(object, normal) {
  const target = normalizeVec3(normal, [0, 0, 1]).normalize();
  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), target);
  object.quaternion.copy(quat);
}

function orientPlaneToNormalAndUp(object, normal, up) {
  const zAxis = normalizeVec3(normal, [0, 0, 1]).normalize();
  let yAxis = normalizeVec3(up, [0, 1, 0]);
  yAxis.sub(zAxis.clone().multiplyScalar(yAxis.dot(zAxis)));
  if (yAxis.lengthSq() < 1e-12) {
    yAxis = Math.abs(zAxis.z) < 0.999 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    yAxis.sub(zAxis.clone().multiplyScalar(yAxis.dot(zAxis)));
  }
  yAxis.normalize();
  const xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();
  const basis = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
  object.quaternion.setFromRotationMatrix(basis);
}

function extrusionShape(vertices, holes) {
  const shape = new THREE.Shape();
  vertices.forEach(([x, y], index) => {
    if (index === 0) {
      shape.moveTo(x, y);
      return;
    }
    shape.lineTo(x, y);
  });
  shape.closePath();
  for (const holeVertices of holes || []) {
    const path = new THREE.Path();
    holeVertices.forEach(([x, y], index) => {
      if (index === 0) {
        path.moveTo(x, y);
        return;
      }
      path.lineTo(x, y);
    });
    path.closePath();
    shape.holes.push(path);
  }
  return shape;
}

function boxGeometryFromItem(item) {
  const size = item.size || [1, 1, 1];
  const center = item.center || [0, 0, 0];
  const geometry = new THREE.BoxGeometry(Number(size[0]), Number(size[1]), Number(size[2]));
  geometry.translate(Number(center[0]), Number(center[1]), Number(center[2]));
  return geometry;
}

function planeGeometryFromItem(item) {
  const size = item.size || [1, 1];
  const center = normalizeVec3(item.center, [0, 0, 0]);
  const geometry = new THREE.PlaneGeometry(Number(size[0]), Number(size[1]));
  const temp = new THREE.Object3D();
  temp.position.copy(center);
  orientToNormal(temp, item.normal || [0, 0, 1]);
  temp.updateMatrixWorld(true);
  geometry.applyMatrix4(temp.matrixWorld);
  return geometry;
}

function polyExtrusionGeometryFromItem(item) {
  const shape = extrusionShape(item.vertices || [], item.holes || []);
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: Number(item.depth || 0), bevelEnabled: false });
  geometry.translate(0, 0, Number(item.z0 || 0));
  return geometry;
}

function makeOutlineShell(object, outlineColor, clippingPlanes, scale = 1.01) {
  const outlineMaterial = new THREE.MeshBasicMaterial({
    color: colorValue(outlineColor, "#ffffff"),
    side: THREE.BackSide,
    transparent: false,
    opacity: 1,
    clippingPlanes,
    depthTest: true,
    depthWrite: false,
    toneMapped: false,
  });
  const outlineMesh = new THREE.Mesh(object.geometry.clone(), outlineMaterial);
  outlineMesh.scale.setScalar(scale);
  outlineMesh.renderOrder = 1000;
  outlineMesh.frustumCulled = false;
  outlineMesh.userData.zviewOutlineMaterial = outlineMaterial;
  return outlineMesh;
}

function makeObjectOutline(spec, object, outlineColor, clippingPlanes) {
  if (!object?.isMesh) {
    return null;
  }
  if (
    spec.metadata?.kind !== "structure"
    || spec.material?.wireframe
    || Number(spec.material?.opacity ?? 1) <= 0
  ) {
    return null;
  }
  if (spec.kind === "sphere") {
    return makeOutlineShell(object, outlineColor, clippingPlanes, 1.035);
  }
  if (spec.kind === "box" || spec.kind === "poly_extrusion" || spec.kind === "plane") {
    const outlineMaterial = new THREE.LineBasicMaterial({
      color: colorValue(outlineColor, "#ffffff"),
      transparent: true,
      opacity: 0.95,
      clippingPlanes,
      depthTest: true,
      depthWrite: false,
    });
    outlineMaterial.toneMapped = false;
    const outline = new THREE.LineSegments(new THREE.EdgesGeometry(object.geometry), outlineMaterial);
    outline.frustumCulled = false;
    outline.userData.zviewOutlineMaterial = outlineMaterial;
    return outline;
  }
  return null;
}

function buildObject(spec, clippingPlanes, outlineColor, materialCache, orderIndex = 0) {
  let object = null;
  const material = makeMaterial(spec.material, clippingPlanes, spec.kind, materialCache);
  const geometry = spec.geometry || {};
  const order = displayOrder(spec, orderIndex);

  switch (spec.kind) {
    case "box": {
      const boxGeometry = boxGeometryFromItem(geometry);
      if (spec.material?.wireframe) {
        const lineMaterial = new THREE.LineBasicMaterial({
          color: colorValue(spec.material?.color, "#0f172a"),
          transparent: Number(spec.material?.opacity ?? 1) < 1,
          opacity: Number(spec.material?.opacity ?? 1),
          clippingPlanes,
        });
        object = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeometry), lineMaterial);
      } else {
        object = new THREE.Mesh(boxGeometry, material);
      }
      break;
    }
    case "sphere": {
      const radius = Number(geometry.radius || 1);
      const center = geometry.center || [0, 0, 0];
      object = new THREE.Mesh(new THREE.SphereGeometry(radius, 32, 18), material);
      object.position.copy(normalizeVec3(center, [0, 0, 0]));
      break;
    }
    case "plane": {
      object = new THREE.Mesh(planeGeometryFromItem(geometry), material);
      break;
    }
    case "line": {
      const points = (geometry.points || []).map((point) => normalizeVec3(point, [0, 0, 0]));
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      const lineMaterial = new THREE.LineBasicMaterial({
        color: colorValue(spec.material?.color, "#dc2626"),
        transparent: Number(spec.material?.opacity ?? 1) < 1,
        opacity: Number(spec.material?.opacity ?? 1),
        clippingPlanes,
      });
      object = new THREE.Line(lineGeometry, lineMaterial);
      break;
    }
    case "arrow": {
      const origin = normalizeVec3(geometry.origin, [0, 0, 0]);
      const direction = normalizeVec3(geometry.direction, [1, 0, 0]).normalize();
      const length = Number(geometry.length || 1);
      object = new THREE.ArrowHelper(direction, origin, length, colorValue(spec.material?.color).getHex(), length * 0.2, length * 0.12);
      break;
    }
    case "poly_extrusion": {
      object = new THREE.Mesh(polyExtrusionGeometryFromItem(geometry), material);
      break;
    }
    default:
      return null;
  }

  const outline = makeObjectOutline(spec, object, outlineColor, clippingPlanes);
  if (outline) {
    object.add(outline);
  }
  object.renderOrder = order;
  object.visible = spec.visible !== false;
  object.userData.zview = spec;
  object.traverse?.((node) => {
    node.userData.zview = spec;
    if (node.userData?.zviewOutlineMaterial) {
      node.renderOrder = order + 0.25;
    } else {
      node.renderOrder = order;
    }
  });
  return object;
}

function makeTextTexture(text, { background = null, color = "#6b7280", size = 256, font = "700 46px ui-sans-serif, system-ui, sans-serif" } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, size, size);
  if (background) {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, size, size);
  }
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, size / 2, size / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.anisotropy = 4;
  return texture;
}

function makeSpriteLabel(text, options = {}) {
  const config = typeof options === "string" ? { color: options } : options;
  const canvas = document.createElement("canvas");
  canvas.width = config.width || 192;
  canvas.height = config.height || 80;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = config.color || "#111827";
  ctx.font = config.font || "700 40px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    alphaTest: 0.01,
  });
  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 2000;
  const [sx, sy] = config.scale || [0.48, 0.24];
  sprite.scale.set(sx, sy, 1);
  if (config.screenSizePx) {
    sprite.userData.screenSizePx = config.screenSizePx;
  }
  return sprite;
}

function makeAxisArrow(direction, color, length, shaftRadius = 0.075, headRadius = 0.18, headLength = 0.42) {
  const dir = direction.clone().normalize();
  const shaftLength = Math.max(length - headLength, 0.01);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.35,
    metalness: 0.05,
  });
  const group = new THREE.Group();
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 20),
    material,
  );
  shaft.position.y = shaftLength / 2;
  const head = new THREE.Mesh(
    new THREE.ConeGeometry(headRadius, headLength, 24),
    material,
  );
  head.position.y = shaftLength + headLength / 2;
  group.add(shaft);
  group.add(head);
  group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  return group;
}

function updateOrthoFrustum(camera, aspect) {
  camera.left = -aspect;
  camera.right = aspect;
  camera.top = 1;
  camera.bottom = -1;
  camera.updateProjectionMatrix();
}

function niceTickStep(min, max, targetCount = 5) {
  const span = Math.max(Math.abs(max - min), 1e-9);
  const raw = span / Math.max(targetCount, 2);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  if (normalized <= 1) {
    return magnitude;
  }
  if (normalized <= 2) {
    return 2 * magnitude;
  }
  if (normalized <= 5) {
    return 5 * magnitude;
  }
  return 10 * magnitude;
}

function integerMicronTickValues(min, max, targetCount = 5) {
  const minMicron = Number(min) * 1e6;
  const maxMicron = Number(max) * 1e6;
  if (Math.abs(maxMicron - minMicron) < 1e-6) {
    return [min];
  }
  const span = Math.max(Math.abs(maxMicron - minMicron), 1);
  const raw = Math.max(span / Math.max(targetCount, 2), 1);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const stepMicron = normalized <= 1
    ? magnitude
    : normalized <= 2
      ? 2 * magnitude
      : normalized <= 5
        ? 5 * magnitude
        : 10 * magnitude;
  const start = Math.ceil((minMicron - 1e-9) / stepMicron) * stepMicron;
  const values = [];
  for (let value = start; value <= maxMicron + 1e-9; value += stepMicron) {
    values.push(Number((value * 1e-6).toFixed(12)));
  }
  if (values.length === 0) {
    const midpointMicron = Math.round((minMicron + maxMicron) / 2);
    values.push(Number((midpointMicron * 1e-6).toFixed(12)));
  }
  return values;
}

function formatMicron(value) {
  const microns = Math.round(Number(value) * 1e6);
  return String(microns);
}

function makeSegments(points, material) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.LineSegments(geometry, material);
}

function measurementScale(size) {
  const spans = [size.x, size.y, size.z]
    .map((value) => Math.abs(Number(value) || 0))
    .filter((value) => value > 0);
  if (spans.length === 0) {
    return 1e-9;
  }
  return Math.max(...spans);
}

function setSegmentsPoints(lineSegments, points) {
  lineSegments.geometry.dispose();
  lineSegments.geometry = new THREE.BufferGeometry().setFromPoints(points);
}

function measurementBoxCorners(box) {
  const min = box.min;
  const max = box.max;
  return {
    lbf: new THREE.Vector3(min.x, min.y, min.z),
    rbf: new THREE.Vector3(max.x, min.y, min.z),
    ltf: new THREE.Vector3(min.x, min.y, max.z),
    rtf: new THREE.Vector3(max.x, min.y, max.z),
    lbb: new THREE.Vector3(min.x, max.y, min.z),
    rbb: new THREE.Vector3(max.x, max.y, min.z),
    ltb: new THREE.Vector3(min.x, max.y, max.z),
    rtb: new THREE.Vector3(max.x, max.y, max.z),
  };
}

function measurementProjectedBounds(corners, camera, rect) {
  const points = Object.values(corners).map((corner) => screenPointForWorld(corner, camera, rect));
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  return {
    minX,
    maxX,
    minY,
    maxY,
    width: Math.max(maxX - minX, 1),
    height: Math.max(maxY - minY, 1),
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
  };
}

function measurementAxisCandidates(corners, axisName) {
  if (axisName === "x") {
    return [
      { start: corners.lbf, end: corners.rbf, offsetDir: new THREE.Vector3(0, -1, -1).normalize() },
      { start: corners.ltf, end: corners.rtf, offsetDir: new THREE.Vector3(0, -1, 1).normalize() },
      { start: corners.lbb, end: corners.rbb, offsetDir: new THREE.Vector3(0, 1, -1).normalize() },
      { start: corners.ltb, end: corners.rtb, offsetDir: new THREE.Vector3(0, 1, 1).normalize() },
    ];
  }
  if (axisName === "y") {
    return [
      { start: corners.lbf, end: corners.lbb, offsetDir: new THREE.Vector3(-1, 0, -1).normalize() },
      { start: corners.ltf, end: corners.ltb, offsetDir: new THREE.Vector3(-1, 0, 1).normalize() },
      { start: corners.rbf, end: corners.rbb, offsetDir: new THREE.Vector3(1, 0, -1).normalize() },
      { start: corners.rtf, end: corners.rtb, offsetDir: new THREE.Vector3(1, 0, 1).normalize() },
    ];
  }
  return [
    { start: corners.lbf, end: corners.ltf, offsetDir: new THREE.Vector3(-1, -1, 0).normalize() },
    { start: corners.lbb, end: corners.ltb, offsetDir: new THREE.Vector3(-1, 1, 0).normalize() },
    { start: corners.rbf, end: corners.rtf, offsetDir: new THREE.Vector3(1, -1, 0).normalize() },
    { start: corners.rbb, end: corners.rtb, offsetDir: new THREE.Vector3(1, 1, 0).normalize() },
  ];
}

function screenVectorForWorldOffset(point, offset, camera, rect) {
  const base = screenPointForWorld(point, camera, rect);
  const shifted = screenPointForWorld(point.clone().add(offset), camera, rect);
  return new THREE.Vector2(shifted.x - base.x, shifted.y - base.y);
}

function chooseMeasurementAxisPlacement(axisName, corners, camera, rect) {
  const projectedBounds = measurementProjectedBounds(corners, camera, rect);
  const center = new THREE.Vector2(projectedBounds.centerX, projectedBounds.centerY);
  const viewMatrix = camera.matrixWorldInverse;
  const candidates = measurementAxisCandidates(corners, axisName);
  const probeScale = Math.max(
    corners.rtb.distanceTo(corners.lbf) * 0.03,
    1e-9,
  );
  let best = candidates[0];
  let bestScore = -Infinity;

  for (const candidate of candidates) {
    const midpoint = candidate.start.clone().lerp(candidate.end, 0.5);
    const screen = screenPointForWorld(midpoint, camera, rect);
    const centerToMid = new THREE.Vector2(screen.x - center.x, screen.y - center.y);
    const outwardScreen = screenVectorForWorldOffset(
      midpoint,
      candidate.offsetDir.clone().multiplyScalar(probeScale),
      camera,
      rect,
    );
    const outwardScore = centerToMid.lengthSq() > 1e-9 && outwardScreen.lengthSq() > 1e-9
      ? centerToMid.normalize().dot(outwardScreen.normalize())
      : -1;
    const topScore = 1 - ((screen.y - projectedBounds.minY) / projectedBounds.height);
    const sideScore = Math.abs(screen.x - projectedBounds.centerX) / Math.max(projectedBounds.width * 0.5, 1);
    const frontScore = midpoint.clone().applyMatrix4(viewMatrix).z;

    const score = axisName === "z"
      ? outwardScore * 3.0 + sideScore * 1.75 + frontScore * 0.08
      : outwardScore * 2.4 + topScore * 1.8 + frontScore * 0.08;

    if (score > bestScore) {
      best = candidate;
      bestScore = score;
    }
  }

  return best;
}

function makeMeasurementFrame(box, theme = THEMES.dark) {
  if (!box || box.isEmpty()) {
    return new THREE.Group();
  }

  const group = new THREE.Group();
  const min = box.min.clone();
  const max = box.max.clone();
  const size = box.getSize(new THREE.Vector3());
  const diag = size.length();
  const scale = measurementScale(size);
  const corners = measurementBoxCorners(box);
  const edgeMaterial = new THREE.LineBasicMaterial({ color: theme.measurement.edge, transparent: true, opacity: 0.8 });
  const axisMaterial = new THREE.LineBasicMaterial({ color: theme.measurement.axis, transparent: true, opacity: 0.95 });
  const tickMaterial = new THREE.LineBasicMaterial({ color: theme.measurement.tick, transparent: true, opacity: 0.9 });

  group.add(
    makeSegments(
      [
        corners.lbf, corners.rbf, corners.lbf, corners.lbb, corners.lbf, corners.ltf,
        corners.rbf, corners.rbb, corners.rbf, corners.rtf,
        corners.lbb, corners.rbb, corners.lbb, corners.ltb,
        corners.rbb, corners.rtb,
        corners.ltf, corners.rtf, corners.ltf, corners.ltb,
        corners.rtf, corners.rtb,
        corners.ltb, corners.rtb,
      ],
      edgeMaterial,
    ),
  );

  const tickLength = Math.max(scale * 0.035, diag * 0.015, 1e-9);
  group.userData.boundsBox = new THREE.Box3(min.clone(), max.clone());
  group.userData.tickLength = tickLength;

  const axisConfigs = [
    {
      axisName: "x",
      label: "x (µm)",
      min: min.x,
      max: max.x,
      axisVector: new THREE.Vector3(1, 0, 0),
    },
    {
      axisName: "y",
      label: "y (µm)",
      min: min.y,
      max: max.y,
      axisVector: new THREE.Vector3(0, 1, 0),
    },
    {
      axisName: "z",
      label: "z (µm)",
      min: min.z,
      max: max.z,
      axisVector: new THREE.Vector3(0, 0, 1),
    },
  ];

  for (const axis of axisConfigs) {
    const axisGroup = new THREE.Group();
    axisGroup.userData.axisVector = axis.axisVector.clone();
    axisGroup.userData.axisName = axis.axisName;
    axisGroup.userData.min = axis.min;
    axisGroup.userData.max = axis.max;
    axisGroup.userData.tickValues = integerMicronTickValues(axis.min, axis.max, 5);

    const axisLine = makeSegments([new THREE.Vector3(), new THREE.Vector3()], axisMaterial);
    const tickSegments = makeSegments([new THREE.Vector3(), new THREE.Vector3()], tickMaterial);
    axisGroup.userData.axisLine = axisLine;
    axisGroup.userData.tickSegments = tickSegments;
    axisGroup.add(axisLine);

    const tickLabels = [];
    for (const value of axisGroup.userData.tickValues) {
      const tickLabel = makeSpriteLabel(formatMicron(value), {
        color: theme.measurement.tickText,
        font: "600 60px ui-sans-serif, system-ui, sans-serif",
        width: 384,
        height: 160,
        screenSizePx: [92, 36],
      });
      axisGroup.add(tickLabel);
      tickLabels.push(tickLabel);
    }
    axisGroup.userData.tickLabels = tickLabels;
    axisGroup.add(tickSegments);

    const axisLabel = makeSpriteLabel(axis.label, {
      color: theme.measurement.axisText,
      font: "700 56px ui-sans-serif, system-ui, sans-serif",
      width: 512,
      height: 160,
      screenSizePx: [168, 48],
    });
    axisGroup.add(axisLabel);
    axisGroup.userData.axisLabel = axisLabel;
    group.add(axisGroup);
  }

  return group;
}

function updateMeasurementFrameLayout(frame, camera, renderer) {
  if (!frame) {
    return;
  }
  const box = frame.userData?.boundsBox;
  const tickLength = Number(frame.userData?.tickLength || 0);
  if (!box || !renderer) {
    return;
  }
  const corners = measurementBoxCorners(box);
  const rect = {
    width: renderer.domElement.clientWidth || renderer.domElement.width || 1,
    height: renderer.domElement.clientHeight || renderer.domElement.height || 1,
  };
  const viewDir = new THREE.Vector3();
  camera.getWorldDirection(viewDir);
  const hideDepthAxis = camera.isOrthographicCamera;
  for (const child of frame.children) {
    const axisVector = child.userData?.axisVector;
    if (!axisVector) {
      continue;
    }
    child.visible = !hideDepthAxis || Math.abs(axisVector.dot(viewDir)) < 0.96;
    if (!child.visible) {
      continue;
    }
    const axisName = child.userData.axisName;
    const placement = chooseMeasurementAxisPlacement(axisName, corners, camera, rect);
    const axisStart = placement.start.clone();
    const axisEnd = placement.end.clone();
    const offsetDir = placement.offsetDir.clone();
    setSegmentsPoints(child.userData.axisLine, [axisStart, axisEnd]);

    const min = Number(child.userData.min);
    const max = Number(child.userData.max);
    const values = child.userData.tickValues || [];
    const tickPoints = [];
    for (const [index, value] of values.entries()) {
      const t = Math.abs(max - min) < 1e-9 ? 0 : (value - min) / (max - min);
      const point = axisStart.clone().lerp(axisEnd, THREE.MathUtils.clamp(t, 0, 1));
      const tickEnd = point.clone().add(offsetDir.clone().multiplyScalar(tickLength));
      tickPoints.push(point, tickEnd);
      child.userData.tickLabels[index]?.position.copy(
        point.clone().add(offsetDir.clone().multiplyScalar(tickLength * 1.7)),
      );
    }
    setSegmentsPoints(child.userData.tickSegments, tickPoints);
    child.userData.axisLabel?.position.copy(
      axisStart.clone().lerp(axisEnd, 0.5).add(offsetDir.clone().multiplyScalar(tickLength * 4.8)),
    );
  }
}

function updateScreenSizedSprites(root, camera, renderer) {
  if (!root || !camera?.isOrthographicCamera) {
    return;
  }
  const widthPx = renderer.domElement.clientWidth || renderer.domElement.width || 1;
  const heightPx = renderer.domElement.clientHeight || renderer.domElement.height || 1;
  const visibleWorldWidth = (camera.right - camera.left) / Math.max(camera.zoom, 1e-9);
  const visibleWorldHeight = (camera.top - camera.bottom) / Math.max(camera.zoom, 1e-9);
  root.traverse((node) => {
    const screenSizePx = node.userData?.screenSizePx;
    if (!node.isSprite || !screenSizePx) {
      return;
    }
    const [spriteWidthPx, spriteHeightPx] = screenSizePx;
    node.scale.set(
      (spriteWidthPx / widthPx) * visibleWorldWidth,
      (spriteHeightPx / heightPx) * visibleWorldHeight,
      1,
    );
  });
}

function screenPointForWorld(point, camera, rect) {
  const projected = point.clone().project(camera);
  return {
    x: (projected.x * 0.5 + 0.5) * rect.width,
    y: (-projected.y * 0.5 + 0.5) * rect.height,
  };
}

function screenRadiusForSphere(center, radiusWorld, camera, rect) {
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const edge = center.clone().add(right.multiplyScalar(Math.max(radiusWorld, 1e-9)));
  const centerScreen = screenPointForWorld(center, camera, rect);
  const edgeScreen = screenPointForWorld(edge, camera, rect);
  return Math.max(Math.hypot(edgeScreen.x - centerScreen.x, edgeScreen.y - centerScreen.y), 140);
}

function projectArcballVector(clientX, clientY, center, radiusPx, camera, rect) {
  const screenCenter = screenPointForWorld(center, camera, rect);
  const radius = Math.max(radiusPx, 120);
  let x = (clientX - rect.left - screenCenter.x) / radius;
  let y = (screenCenter.y - (clientY - rect.top)) / radius;
  const lengthSq = x * x + y * y;
  let z = 0;
  if (lengthSq > 1) {
    const scale = 1 / Math.sqrt(lengthSq);
    x *= scale;
    y *= scale;
  } else {
    z = Math.sqrt(1 - lengthSq);
  }
  return new THREE.Vector3(x, y, z).normalize();
}

function worldVectorToCameraTrackball(worldVector, camera) {
  const inverseCamera = camera.quaternion.clone().invert();
  return worldVector.clone().applyQuaternion(inverseCamera).normalize();
}

function quaternionBetweenVectors(fromVec, toVec) {
  const from = fromVec.clone().normalize();
  const to = toVec.clone().normalize();
  const dot = THREE.MathUtils.clamp(from.dot(to), -1, 1);
  if (dot > 0.999999) {
    return new THREE.Quaternion();
  }
  if (dot < -0.999999) {
    const axis = new THREE.Vector3(1, 0, 0).cross(from);
    if (axis.lengthSq() < 1e-12) {
      axis.set(0, 1, 0).cross(from);
    }
    axis.normalize();
    return new THREE.Quaternion().setFromAxisAngle(axis, Math.PI);
  }
  const axis = new THREE.Vector3().crossVectors(from, to).normalize();
  return new THREE.Quaternion().setFromAxisAngle(axis, Math.acos(dot));
}

function applyArcballRotation(camera, controls, center, fromVec, toVec) {
  const dot = THREE.MathUtils.clamp(fromVec.dot(toVec), -1, 1);
  if (dot > 0.99999) {
    return;
  }
  const axisCamera = new THREE.Vector3().crossVectors(toVec, fromVec);
  if (axisCamera.lengthSq() < 1e-12) {
    return;
  }
  axisCamera.normalize();
  const axisWorld = axisCamera.applyQuaternion(camera.quaternion).normalize();
  const angle = Math.acos(dot) * 1.4;
  const rotation = new THREE.Quaternion().setFromAxisAngle(axisWorld, angle);
  const offset = camera.position.clone().sub(center).applyQuaternion(rotation);
  camera.position.copy(center.clone().add(offset));
  camera.up.applyQuaternion(rotation).normalize();
  controls.target.copy(center);
  camera.lookAt(center);
  controls.update();
}

function rotateCameraAroundCenter(camera, controls, center, axis, angle) {
  const rotationAxis = axis.clone().normalize();
  if (rotationAxis.lengthSq() < 1e-12 || Math.abs(angle) < 1e-12) {
    return;
  }
  const rotation = new THREE.Quaternion().setFromAxisAngle(rotationAxis, angle);
  const offset = camera.position.clone().sub(center).applyQuaternion(rotation);
  camera.position.copy(center.clone().add(offset));
  camera.up.applyQuaternion(rotation).normalize();
  controls.target.copy(center);
  camera.lookAt(center);
  controls.update();
}

function clippingPlanes(scene) {
  return (scene.clip_planes || [])
    .filter((plane) => plane.enabled)
    .map((plane) => new THREE.Plane(normalizeVec3(plane.normal, [0, 0, 1]), Number(plane.constant)));
}

function makeRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(globalThis.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth || 900, container.clientHeight || 520, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.sortObjects = false;
  renderer.domElement.className = "zview-canvas";
  container.appendChild(renderer.domElement);
  return renderer;
}

function resize(renderer, camera, element) {
  const width = element.clientWidth || 900;
  const height = element.clientHeight || 520;
  renderer.setSize(width, height, false);
  const aspect = width / Math.max(height, 1);
  camera.aspect = aspect;
  if (camera.isOrthographicCamera) {
    updateOrthoFrustum(camera, aspect);
  }
  camera.updateProjectionMatrix();
}

function createLayout(host) {
  host.classList.add("zview-root");
  host.dataset.theme = "dark";
  host.replaceChildren();

  const toolbar = document.createElement("div");
  toolbar.className = "zview-toolbar";
  const resetButton = document.createElement("button");
  resetButton.className = "zview-button";
  resetButton.textContent = "Reset View";
  toolbar.appendChild(resetButton);
  const rotateButton = document.createElement("button");
  rotateButton.className = "zview-button";
  rotateButton.textContent = "Start Rotate";
  toolbar.appendChild(rotateButton);
  const axesButton = document.createElement("button");
  axesButton.className = "zview-button";
  axesButton.textContent = "Hide Axes";
  toolbar.appendChild(axesButton);
  const themeButton = document.createElement("button");
  themeButton.className = "zview-button";
  themeButton.textContent = "Light Mode";
  toolbar.appendChild(themeButton);

  const canvasHost = document.createElement("div");
  canvasHost.style.width = "100%";
  canvasHost.style.height = "100%";
  canvasHost.style.minHeight = "480px";

  const gizmoHost = document.createElement("div");
  gizmoHost.className = "zview-gizmo";

  const sidebar = document.createElement("div");
  sidebar.className = "zview-sidebar";
  sidebar.dataset.collapsed = "false";

  const sidebarToggleButton = document.createElement("button");
  sidebarToggleButton.className = "zview-button zview-sidebar-toggle";
  sidebarToggleButton.dataset.visible = "false";
  sidebarToggleButton.type = "button";
  sidebarToggleButton.textContent = "Show Sidebar";

  host.appendChild(toolbar);
  host.appendChild(canvasHost);
  host.appendChild(gizmoHost);
  host.appendChild(sidebar);
  host.appendChild(sidebarToggleButton);
  return { resetButton, rotateButton, axesButton, themeButton, canvasHost, gizmoHost, sidebar, sidebarToggleButton };
}

function fitCamera(camera, controls, objectRoot) {
  const box = new THREE.Box3().setFromObject(objectRoot);
  if (box.isEmpty()) {
    return null;
  }
  const center = box.getCenter(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 1e-9);
  const distance = radius * 4.5;
  const isoDir = new THREE.Vector3(1, -1, 1).normalize();
  camera.position.copy(center.clone().add(isoDir.multiplyScalar(distance)));
  camera.near = Math.max(distance / 1000, 1e-9);
  camera.far = distance * 20;
  camera.up.set(0, 0, 1);
  camera.lookAt(center);
  camera.updateMatrixWorld(true);
  const corners = [
    new THREE.Vector3(box.min.x, box.min.y, box.min.z),
    new THREE.Vector3(box.min.x, box.min.y, box.max.z),
    new THREE.Vector3(box.min.x, box.max.y, box.min.z),
    new THREE.Vector3(box.min.x, box.max.y, box.max.z),
    new THREE.Vector3(box.max.x, box.min.y, box.min.z),
    new THREE.Vector3(box.max.x, box.min.y, box.max.z),
    new THREE.Vector3(box.max.x, box.max.y, box.min.z),
    new THREE.Vector3(box.max.x, box.max.y, box.max.z),
  ];
  let halfWidth = 1e-6;
  let halfHeight = 1e-6;
  for (const corner of corners) {
    const cameraSpace = corner.clone().applyMatrix4(camera.matrixWorldInverse);
    halfWidth = Math.max(halfWidth, Math.abs(cameraSpace.x));
    halfHeight = Math.max(halfHeight, Math.abs(cameraSpace.y));
  }
  const margin = 1.12;
  const aspect = Math.max(Math.abs(camera.right), 1e-6);
  camera.zoom = Math.min(aspect / (halfWidth * margin), 1 / (halfHeight * margin));
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
  return { box, sphere, center, radius };
}

function applyCameraSpec(camera, controls, sceneSpec, center) {
  const cameraSpec = sceneSpec.camera || {};
  const centerVec = center || normalizeVec3(cameraSpec.target, [0, 0, 0]);
  if (Array.isArray(cameraSpec.up)) {
    camera.up.copy(normalizeVec3(cameraSpec.up, [0, 0, 1]));
  }
  if (Array.isArray(cameraSpec.position)) {
    camera.position.copy(normalizeVec3(cameraSpec.position, [1.5, 1.5, 1.2]));
  }
  if (camera.isPerspectiveCamera && cameraSpec.fov !== undefined) {
    camera.fov = Number(cameraSpec.fov || 45);
  }
  if (camera.isOrthographicCamera && cameraSpec.zoom !== undefined) {
    camera.zoom = Number(cameraSpec.zoom || camera.zoom);
  }
  controls.target.copy(centerVec);
  camera.lookAt(centerVec);
  camera.updateProjectionMatrix();
  controls.update();
}

function renderObjects(root, sceneSpec, themeName) {
  while (root.children.length) {
    const child = root.children[0];
    root.remove(child);
    child.traverse((node) => {
      node.geometry?.dispose?.();
      if (Array.isArray(node.material)) {
        node.material.forEach((material) => material.dispose?.());
      } else {
        node.material?.dispose?.();
      }
    });
  }

  const planes = clippingPlanes(sceneSpec);
  const content = new THREE.Group();
  const objectMap = new Map();
  const materialCache = new Map();
  for (const [index, spec] of (sceneSpec.objects || []).entries()) {
    const object = buildObject(spec, planes, THEMES[themeName].objectOutline, materialCache, index);
    if (object) {
      content.add(object);
      objectMap.set(spec.id, object);
    }
  }
  root.add(content);
  return { content, planes, objectMap };
}

function makeGizmo(rendererHost, themeName = "dark") {
  const size = 96;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(globalThis.devicePixelRatio || 1);
  renderer.setSize(size, size, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  rendererHost.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1.35, 1.35, 1.35, -1.35, 0.1, 50);
  camera.position.set(0, 0, 8);
  camera.zoom = 0.95;
  camera.updateProjectionMatrix();

  scene.add(new THREE.AmbientLight(0xffffff, 1.55));
  const light = new THREE.DirectionalLight(0xffffff, 1.15);
  light.position.set(3, -4, 6);
  scene.add(light);

  const root = new THREE.Group();
  root.scale.setScalar(0.58);
  scene.add(root);

  const faceDefs = [
    { label: "RIGHT", position: [1, 0, 0], rotation: [0, Math.PI / 2, 0], direction: [1, 0, 0], up: [0, 0, 1] },
    { label: "LEFT", position: [-1, 0, 0], rotation: [0, -Math.PI / 2, 0], direction: [-1, 0, 0], up: [0, 1, 0] },
    { label: "TOP", position: [0, 0, 1], rotation: [0, 0, 0], direction: [0, 0, 1], up: [0, 1, 0] },
    { label: "BOTTOM", position: [0, 0, -1], rotation: [0, Math.PI, 0], direction: [0, 0, -1], up: [0, 1, 0] },
    { label: "FRONT", position: [0, -1, 0], rotation: [Math.PI / 2, 0, 0], direction: [0, -1, 0], up: [0, 0, 1] },
    { label: "BACK", position: [0, 1, 0], rotation: [-Math.PI / 2, 0, 0], direction: [0, 1, 0], up: [1, 0, 0] },
  ];

  const cubeRoot = new THREE.Group();
  const baseGeometry = new RoundedBoxGeometry(2, 2, 2, 4, 0.18);
  const baseMaterial = new THREE.MeshPhysicalMaterial({
    color: THEMES[themeName].gizmo.cubeColor,
    roughness: THEMES[themeName].gizmo.cubeRoughness,
    metalness: THEMES[themeName].gizmo.cubeMetalness,
    transparent: true,
    opacity: THEMES[themeName].gizmo.cubeOpacity,
  });
  const baseCube = new THREE.Mesh(baseGeometry, baseMaterial);
  cubeRoot.add(baseCube);

  const faceMeshes = [];
  const faceViews = new Map();
  for (const face of faceDefs) {
    const faceMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1.84, 1.84),
      new THREE.MeshBasicMaterial({
        map: makeTextTexture(face.label, { background: null, color: THEMES[themeName].gizmo.faceText }),
        transparent: true,
        opacity: THEMES[themeName].gizmo.faceOpacity,
      }),
    );
    faceMesh.position.set(...face.position.map((value) => value * 1.01));
    orientPlaneToNormalAndUp(faceMesh, face.position, face.up);
    faceMesh.userData.view = {
      direction: new THREE.Vector3(...face.direction),
      up: new THREE.Vector3(...face.up),
    };
    faceMesh.userData.label = face.label;
    cubeRoot.add(faceMesh);
    faceMeshes.push(faceMesh);
    faceViews.set(face.label, {
      direction: new THREE.Vector3(...face.direction),
      up: new THREE.Vector3(...face.up),
      label: face.label,
    });
  }

  root.add(cubeRoot);

  const axesRoot = new THREE.Group();
  const axisScale = 2.2;
  const axisOrigin = new THREE.Vector3(-1.15, -1.15, -1.15);
  const axisColors = THEMES[themeName].gizmo;
  const xArrow = makeAxisArrow(
    new THREE.Vector3(1, 0, 0),
    colorValue(axisColors.axisX).getHex(),
    axisScale,
  );
  xArrow.position.copy(axisOrigin);
  const yArrow = makeAxisArrow(
    new THREE.Vector3(0, 0, 1),
    colorValue(axisColors.axisY).getHex(),
    axisScale,
  );
  yArrow.position.copy(axisOrigin);
  const zArrow = makeAxisArrow(
    new THREE.Vector3(0, 1, 0),
    colorValue(axisColors.axisZ).getHex(),
    axisScale,
  );
  zArrow.position.copy(axisOrigin);
  axesRoot.add(xArrow);
  axesRoot.add(yArrow);
  axesRoot.add(zArrow);
  const xLabel = makeSpriteLabel("X", axisColors.axisX);
  xLabel.position.set(1.7, -1.16, -1.15);
  const yLabel = makeSpriteLabel("Y", axisColors.axisY);
  yLabel.position.set(-1.15, -1.15, 1.7);
  const zLabel = makeSpriteLabel("Z", axisColors.axisZ);
  zLabel.position.set(-1.15, 1.7, -1.15);
  axesRoot.add(xLabel);
  axesRoot.add(yLabel);
  axesRoot.add(zLabel);
  root.add(axesRoot);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pickables = [baseCube];

  function faceLabelFromLocalPoint(localPoint) {
    const absX = Math.abs(localPoint.x);
    const absY = Math.abs(localPoint.y);
    const absZ = Math.abs(localPoint.z);
    if (absX >= absY && absX >= absZ) {
      return localPoint.x >= 0 ? "RIGHT" : "LEFT";
    }
    if (absZ >= absX && absZ >= absY) {
      return localPoint.z >= 0 ? "TOP" : "BOTTOM";
    }
    return localPoint.y >= 0 ? "BACK" : "FRONT";
  }

  function faceViewFromHit(hit) {
    if (!hit || hit.object !== baseCube) {
      return null;
    }
    const localPoint = baseCube.worldToLocal(hit.point.clone());
    const label = faceLabelFromLocalPoint(localPoint);
    const view = faceViews.get(label);
    if (!view) {
      return null;
    }
    return {
      direction: view.direction.clone(),
      up: view.up.clone(),
      label,
    };
  }

  function syncFromCamera(mainCamera) {
    root.quaternion.copy(mainCamera.quaternion).invert();
  }

  function pickObject(event, objects = pickables) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(objects, false);
    return hits[0] || null;
  }

  function render() {
    renderer.render(scene, camera);
  }

  function setTheme(nextThemeName) {
    const theme = THEMES[nextThemeName]?.gizmo ?? THEMES.dark.gizmo;
    baseMaterial.color.set(theme.cubeColor);
    baseMaterial.roughness = theme.cubeRoughness;
    baseMaterial.metalness = theme.cubeMetalness;
    baseMaterial.opacity = theme.cubeOpacity;
    baseMaterial.needsUpdate = true;
    for (const faceMesh of faceMeshes) {
      faceMesh.material.map?.dispose?.();
      faceMesh.material.map = makeTextTexture(faceMesh.userData.label, { background: null, color: theme.faceText });
      faceMesh.material.opacity = theme.faceOpacity;
      faceMesh.material.needsUpdate = true;
    }
  }

  return {
    renderer,
    domElement: renderer.domElement,
    camera,
    syncFromCamera,
    pickObject,
    pickFace(event) {
      return faceViewFromHit(pickObject(event, [baseCube]));
    },
    pickDragSurface(event) {
      return pickObject(event, [baseCube]);
    },
    setTheme,
    render,
    destroy() {
      renderer.dispose();
      rendererHost.replaceChildren();
    },
  };
}

function pickHit(raycaster, pointer, event, camera, interactables, canvas) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(interactables, true);
  return hits.find((entry) => entry.object?.userData?.zview) || null;
}

function syncSelection(lastHoveredIdRef, hit, onHover, onSelect, select) {
  if (!hit) {
    if (lastHoveredIdRef.value !== null) {
      lastHoveredIdRef.value = null;
      onHover(null);
    }
    if (select) {
      onSelect(null);
    }
    return;
  }
  const info = hit.object.userData.zview;
  if (lastHoveredIdRef.value !== info.id) {
    lastHoveredIdRef.value = info.id;
    onHover(info.id);
  }
  if (select) {
    onSelect(info.id);
  }
}

function mountZView({ el, sceneSpec, onHover = () => {}, onSelect = () => {} }) {
  const { resetButton, rotateButton, axesButton, themeButton, canvasHost, gizmoHost, sidebar, sidebarToggleButton } = createLayout(el);
  const renderer = makeRenderer(canvasHost);
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1e-6, 1e3);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.enableRotate = false;
  controls.enablePan = false;
  controls.zoomSpeed = 1.15;
  controls.enableZoom = true;

  const sceneRoot = new THREE.Group();
  scene.add(sceneRoot);
  scene.add(new THREE.AmbientLight(0xffffff, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(3, -5, 8);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xe5e7eb, 0.35);
  fill.position.set(-4, 2, 3);
  scene.add(fill);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const lastHoveredIdRef = { value: null };
  let currentTheme = "dark";
  const gizmo = makeGizmo(gizmoHost, currentTheme);

  let interactables = [];
  let sceneBounds = null;
  let dragRotate = null;
  let content = null;
  let snapAnimation = null;
  let gizmoDrag = null;
  let initialView = null;
  let measurementFrame = null;
  let showMeasurementFrame = true;
  let autoRotateEnabled = false;
  let objectMap = new Map();
  const objectVisibility = new Map();
  const collapsedGroups = { design: false, devices: false, boundaries: false };
  let sidebarVisible = true;
  const gizmoDragThresholdPx = 4;
  const autoRotateSpeedRadPerSec = 0.45;

  function updateRotateButtonLabel() {
    rotateButton.textContent = autoRotateEnabled ? "Stop Rotate" : "Start Rotate";
  }

  function updateAxesButtonLabel() {
    axesButton.textContent = showMeasurementFrame ? "Hide Axes" : "Show Axes";
  }

  function updateThemeButtonLabel() {
    themeButton.textContent = currentTheme === "dark" ? "Light Mode" : "Dark Mode";
  }

  function updateSidebarState() {
    sidebar.dataset.collapsed = sidebarVisible ? "false" : "true";
    sidebarToggleButton.dataset.visible = sidebarVisible ? "false" : "true";
    if (measurementFrame) {
      updateMeasurementFrameLayout(measurementFrame, camera, renderer);
      updateScreenSizedSprites(measurementFrame, camera, renderer);
    }
  }

  function updateObjectOutlineTheme() {
    const outlineColor = THEMES[currentTheme].objectOutline;
    for (const object of objectMap.values()) {
      object.traverse((node) => {
        const outlineMaterial = node.userData?.zviewOutlineMaterial;
        if (outlineMaterial?.color) {
          outlineMaterial.color.set(colorValue(outlineColor));
          outlineMaterial.needsUpdate = true;
        }
      });
    }
  }

  function classifyObject(spec) {
    const kind = spec.metadata?.kind;
    if (kind === "monitor" || kind === "source" || kind === "source_direction") {
      return "devices";
    }
    if (kind === "domain" || kind === "boundary" || spec.label?.startsWith("PML ")) {
      return "boundaries";
    }
    return "design";
  }

  function objectMetaSummary(spec) {
    const parts = [];
    if (spec.metadata?.type) {
      parts.push(String(spec.metadata.type));
    } else {
      parts.push(String(spec.kind));
    }
    if (Number(spec.metadata?.structure_count || 1) > 1) {
      parts.push(`${Number(spec.metadata.structure_count)} merged`);
    }
    if (spec.metadata?.material?.permittivity !== undefined && spec.metadata?.kind === "structure") {
      parts.push(`eps ${Number(spec.metadata.material.permittivity).toFixed(2)}`);
    }
    if (spec.metadata?.plane_normal) {
      parts.push(`normal ${spec.metadata.plane_normal}`);
    }
    return parts.join(" · ");
  }

  function applyObjectVisibility(objectId, visible) {
    objectVisibility.set(objectId, visible);
    const object = objectMap.get(objectId);
    if (object) {
      object.visible = visible;
    }
    const spec = (sceneSpec.objects || []).find((entry) => entry.id === objectId);
    if (spec) {
      spec.visible = visible;
    }
  }

  function renderSidebar() {
    const objects = sceneSpec.objects || [];
    const groups = {
      design: objects.filter((spec) => classifyObject(spec) === "design"),
      devices: objects.filter((spec) => classifyObject(spec) === "devices"),
      boundaries: objects.filter((spec) => classifyObject(spec) === "boundaries"),
    };
    sidebar.replaceChildren();

    const header = document.createElement("div");
    header.className = "zview-sidebar-header";
    const heading = document.createElement("div");
    const title = document.createElement("div");
    title.className = "zview-sidebar-title";
    title.textContent = sceneSpec.title || "Scene Overview";
    heading.appendChild(title);

    const subtitle = document.createElement("div");
    subtitle.className = "zview-sidebar-subtitle";
    subtitle.textContent = `${objects.length} objects · visibility controls`;
    heading.appendChild(subtitle);
    header.appendChild(heading);

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "zview-button zview-sidebar-close";
    closeButton.textContent = "Hide";
    closeButton.addEventListener("click", () => {
      sidebarVisible = false;
      updateSidebarState();
    });
    header.appendChild(closeButton);
    sidebar.appendChild(header);

    for (const [groupKey, groupLabel] of [["design", "Design"], ["devices", "Devices"], ["boundaries", "Boundaries"]]) {
      const groupObjects = groups[groupKey];
      const groupEl = document.createElement("div");
      groupEl.className = "zview-sidebar-group";

      const headerButton = document.createElement("button");
      headerButton.type = "button";
      headerButton.className = "zview-sidebar-group-button";
      headerButton.addEventListener("click", () => {
        collapsedGroups[groupKey] = !collapsedGroups[groupKey];
        renderSidebar();
      });

      const headerTitle = document.createElement("div");
      headerTitle.className = "zview-sidebar-group-title";
      headerTitle.textContent = groupLabel;
      headerButton.appendChild(headerTitle);

      const headerCount = document.createElement("div");
      headerCount.className = "zview-sidebar-group-count";
      headerCount.textContent = `${groupObjects.length} ${collapsedGroups[groupKey] ? "▸" : "▾"}`;
      headerButton.appendChild(headerCount);
      groupEl.appendChild(headerButton);

      if (!collapsedGroups[groupKey]) {
        const itemsEl = document.createElement("div");
        itemsEl.className = "zview-sidebar-items";
        if (groupObjects.length === 0) {
          const emptyEl = document.createElement("div");
          emptyEl.className = "zview-sidebar-empty";
          emptyEl.textContent = "No items";
          itemsEl.appendChild(emptyEl);
        } else {
          for (const spec of groupObjects) {
            const itemEl = document.createElement("label");
            itemEl.className = "zview-sidebar-item";

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = objectVisibility.get(spec.id) ?? spec.visible !== false;
            checkbox.addEventListener("change", () => applyObjectVisibility(spec.id, checkbox.checked));
            itemEl.appendChild(checkbox);

            const itemCopy = document.createElement("div");
            const itemLabel = document.createElement("div");
            itemLabel.className = "zview-sidebar-item-label";
            itemLabel.textContent = spec.label || spec.kind;
            itemCopy.appendChild(itemLabel);

            const itemMeta = document.createElement("div");
            itemMeta.className = "zview-sidebar-item-meta";
            itemMeta.textContent = objectMetaSummary(spec);
            itemCopy.appendChild(itemMeta);

            itemEl.appendChild(itemCopy);
            itemsEl.appendChild(itemEl);
          }
        }
        groupEl.appendChild(itemsEl);
      }

      sidebar.appendChild(groupEl);
    }
  }

  function applyTheme() {
    el.dataset.theme = currentTheme;
    scene.background = colorValue(THEMES[currentTheme].sceneBackground);
    gizmo.setTheme(currentTheme);
    updateObjectOutlineTheme();
    if (sceneBounds) {
      if (measurementFrame) {
        sceneRoot.remove(measurementFrame);
      }
      measurementFrame = makeMeasurementFrame(sceneBounds.box, THEMES[currentTheme]);
      measurementFrame.visible = showMeasurementFrame;
      sceneRoot.add(measurementFrame);
      updateMeasurementFrameLayout(measurementFrame, camera, renderer);
      updateScreenSizedSprites(measurementFrame, camera, renderer);
    }
    updateThemeButtonLabel();
  }

  sidebarToggleButton.addEventListener("click", () => {
    sidebarVisible = true;
    updateSidebarState();
  });

  function currentDistance() {
    return sceneBounds ? camera.position.distanceTo(sceneBounds.center) : 1;
  }

  function startSnapToView(view) {
    if (!sceneBounds) {
      return;
    }
    const center = sceneBounds.center.clone();
    const distance = currentDistance();
    const dir = view.direction.clone().normalize();
    const targetPosition = center.clone().add(dir.multiplyScalar(distance));
    const targetUp = view.up.clone().normalize();
    if (
      camera.position.distanceToSquared(targetPosition) < 1e-10 &&
      camera.up.clone().normalize().distanceToSquared(targetUp) < 1e-10 &&
      Math.abs((view.zoom ?? camera.zoom) - camera.zoom) < 1e-10
    ) {
      controls.target.copy(center);
      camera.lookAt(center);
      return;
    }
    snapAnimation = {
      startTime: performance.now(),
      durationMs: 220,
      startPos: camera.position.clone(),
      endPos: targetPosition,
      startUp: camera.up.clone(),
      endUp: targetUp,
      startZoom: camera.zoom,
      endZoom: view.zoom ?? camera.zoom,
      center,
    };
    controls.target.copy(center);
  }

  function hydrate(nextScene) {
    snapAnimation = null;
    scene.background = colorValue(THEMES[currentTheme].sceneBackground);
    resize(renderer, camera, canvasHost);
    const rendered = renderObjects(sceneRoot, nextScene, currentTheme);
    content = rendered.content;
    objectMap = rendered.objectMap;
    renderer.localClippingEnabled = rendered.planes.length > 0;

    sceneBounds = fitCamera(camera, controls, content);
    if (sceneBounds) {
      measurementFrame = makeMeasurementFrame(sceneBounds.box, THEMES[currentTheme]);
      measurementFrame.visible = showMeasurementFrame;
      sceneRoot.add(measurementFrame);
      applyCameraSpec(camera, controls, nextScene, sceneBounds.center);
      controls.target.copy(sceneBounds.center);
      camera.lookAt(sceneBounds.center);
      controls.update();
      updateMeasurementFrameLayout(measurementFrame, camera, renderer);
      updateScreenSizedSprites(measurementFrame, camera, renderer);
      initialView = {
        center: sceneBounds.center.clone(),
        position: camera.position.clone(),
        up: camera.up.clone(),
        zoom: camera.zoom,
      };
    }

    interactables = [];
    content.traverse((node) => {
      if ((node.isMesh || node.isLine) && node.userData?.zview) {
        interactables.push(node);
      }
    });
    for (const spec of nextScene.objects || []) {
      if (!objectVisibility.has(spec.id)) {
        objectVisibility.set(spec.id, spec.visible !== false);
      }
      applyObjectVisibility(spec.id, objectVisibility.get(spec.id));
    }
    renderSidebar();
  }

  hydrate(sceneSpec);

  renderer.domElement.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !sceneBounds) {
      return;
    }
    const hit = pickHit(raycaster, pointer, event, camera, interactables, renderer.domElement);
    if (!hit) {
      return;
    }
    const center = sceneBounds.center.clone();
    const hitWorldVector = hit.point.clone().sub(center);
    if (hitWorldVector.lengthSq() < 1e-12) {
      return;
    }
    const rect = renderer.domElement.getBoundingClientRect();
    const cursorVec = projectArcballVector(
      event.clientX,
      event.clientY,
      center,
      screenRadiusForSphere(center, sceneBounds.radius, camera, rect),
      camera,
      rect,
    );
    const hitCameraVec = worldVectorToCameraTrackball(hitWorldVector, camera);
    dragRotate = {
      pointerId: event.pointerId,
      center,
      radiusPx: screenRadiusForSphere(center, sceneBounds.radius, camera, rect),
      alignQuat: quaternionBetweenVectors(cursorVec, hitCameraVec),
      lastVec: hitCameraVec,
      moved: false,
    };
    renderer.domElement.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  renderer.domElement.addEventListener("pointermove", (event) => {
    if (dragRotate && event.pointerId === dragRotate.pointerId) {
      const rect = renderer.domElement.getBoundingClientRect();
      const nextVec = projectArcballVector(
        event.clientX,
        event.clientY,
        dragRotate.center,
        dragRotate.radiusPx,
        camera,
        rect,
      ).applyQuaternion(dragRotate.alignQuat).normalize();
      applyArcballRotation(camera, controls, dragRotate.center, dragRotate.lastVec, nextVec);
      dragRotate.lastVec = nextVec;
      dragRotate.moved = true;
      syncSelection(
        lastHoveredIdRef,
        pickHit(raycaster, pointer, event, camera, interactables, renderer.domElement),
        onHover,
        onSelect,
        false,
      );
      event.preventDefault();
      return;
    }
    syncSelection(
      lastHoveredIdRef,
      pickHit(raycaster, pointer, event, camera, interactables, renderer.domElement),
      onHover,
      onSelect,
      false,
    );
  });

  renderer.domElement.addEventListener("pointerup", (event) => {
    if (!dragRotate || event.pointerId !== dragRotate.pointerId) {
      return;
    }
    renderer.domElement.releasePointerCapture(event.pointerId);
    if (!dragRotate.moved) {
      syncSelection(
        lastHoveredIdRef,
        pickHit(raycaster, pointer, event, camera, interactables, renderer.domElement),
        onHover,
        onSelect,
        true,
      );
    }
    dragRotate = null;
  });

  renderer.domElement.addEventListener("pointercancel", (event) => {
    if (!dragRotate || event.pointerId !== dragRotate.pointerId) {
      return;
    }
    renderer.domElement.releasePointerCapture(event.pointerId);
    dragRotate = null;
  });

  renderer.domElement.addEventListener("pointerleave", () => {
    if (dragRotate) {
      return;
    }
    syncSelection(lastHoveredIdRef, null, onHover, onSelect, false);
  });

  gizmo.domElement.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !sceneBounds) {
      return;
    }
    const faceHit = gizmo.pickFace(event);
    const hit = gizmo.pickDragSurface(event);
    if (!hit) {
      return;
    }
    const rect = gizmo.domElement.getBoundingClientRect();
    const gizmoCenter = new THREE.Vector3(0, 0, 0);
    const radiusPx = Math.max(Math.min(rect.width, rect.height) * 0.38, 34);
    const cursorVec = projectArcballVector(
      event.clientX,
      event.clientY,
      gizmoCenter,
      radiusPx,
      gizmo.camera,
      rect,
    );
    const hitVecWorld = hit.point.clone();
    if (hitVecWorld.lengthSq() < 1e-12) {
      return;
    }
    const hitVec = worldVectorToCameraTrackball(hitVecWorld, gizmo.camera);
    gizmoDrag = {
      pointerId: event.pointerId,
      radiusPx,
      alignQuat: quaternionBetweenVectors(cursorVec, hitVec),
      lastVec: hitVec,
      clickedView: faceHit
        ? {
            direction: faceHit.direction.clone(),
            up: faceHit.up.clone(),
          }
        : null,
      startClientX: event.clientX,
      startClientY: event.clientY,
      moved: false,
    };
    gizmo.domElement.setPointerCapture(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
  });
  gizmo.domElement.addEventListener("pointermove", (event) => {
    if (!gizmoDrag || event.pointerId !== gizmoDrag.pointerId || !sceneBounds) {
      return;
    }
    const dragDistance = Math.hypot(event.clientX - gizmoDrag.startClientX, event.clientY - gizmoDrag.startClientY);
    if (!gizmoDrag.moved && dragDistance < gizmoDragThresholdPx) {
      return;
    }
    const rect = gizmo.domElement.getBoundingClientRect();
    const nextVec = projectArcballVector(
      event.clientX,
      event.clientY,
      new THREE.Vector3(0, 0, 0),
      gizmoDrag.radiusPx,
      gizmo.camera,
      rect,
    ).applyQuaternion(gizmoDrag.alignQuat).normalize();
    applyArcballRotation(camera, controls, sceneBounds.center, gizmoDrag.lastVec, nextVec);
    gizmoDrag.lastVec = nextVec;
    gizmoDrag.moved = true;
    event.preventDefault();
    event.stopPropagation();
  });
  gizmo.domElement.addEventListener("pointerup", (event) => {
    if (!gizmoDrag || event.pointerId !== gizmoDrag.pointerId) {
      return;
    }
    gizmo.domElement.releasePointerCapture(event.pointerId);
    if (!gizmoDrag.moved && gizmoDrag.clickedView) {
      startSnapToView(gizmoDrag.clickedView);
    }
    gizmoDrag = null;
    event.preventDefault();
    event.stopPropagation();
  });
  gizmo.domElement.addEventListener("pointercancel", (event) => {
    if (!gizmoDrag || event.pointerId !== gizmoDrag.pointerId) {
      return;
    }
    gizmo.domElement.releasePointerCapture(event.pointerId);
    gizmoDrag = null;
  });

  resetButton.textContent = "Reset Fit";
  resetButton.addEventListener("click", () => {
    if (!initialView) {
      hydrate(sceneSpec);
      return;
    }
    snapAnimation = {
      startTime: performance.now(),
      durationMs: 220,
      startPos: camera.position.clone(),
      endPos: initialView.position.clone(),
      startUp: camera.up.clone(),
      endUp: initialView.up.clone(),
      startZoom: camera.zoom,
      endZoom: initialView.zoom,
      center: initialView.center.clone(),
    };
    controls.target.copy(initialView.center);
  });
  updateRotateButtonLabel();
  updateAxesButtonLabel();
  updateThemeButtonLabel();
  updateSidebarState();
  rotateButton.addEventListener("click", () => {
    autoRotateEnabled = !autoRotateEnabled;
    updateRotateButtonLabel();
  });
  axesButton.addEventListener("click", () => {
    showMeasurementFrame = !showMeasurementFrame;
    if (measurementFrame) {
      measurementFrame.visible = showMeasurementFrame;
    }
    updateAxesButtonLabel();
  });
  themeButton.addEventListener("click", () => {
    currentTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme();
  });

  const observer = new ResizeObserver(() => resize(renderer, camera, canvasHost));
  observer.observe(canvasHost);

  let active = true;
  let lastFrameTime = performance.now();
  function frame() {
    if (!active) {
      return;
    }
    const now = performance.now();
    const deltaSeconds = Math.min(Math.max((now - lastFrameTime) / 1000, 0), 0.05);
    lastFrameTime = now;
    if (snapAnimation) {
      const t = Math.min((now - snapAnimation.startTime) / snapAnimation.durationMs, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      camera.position.lerpVectors(snapAnimation.startPos, snapAnimation.endPos, eased);
      camera.up.copy(snapAnimation.startUp).lerp(snapAnimation.endUp, eased).normalize();
      camera.zoom = THREE.MathUtils.lerp(snapAnimation.startZoom ?? camera.zoom, snapAnimation.endZoom ?? camera.zoom, eased);
      camera.updateProjectionMatrix();
      controls.target.copy(snapAnimation.center);
      camera.lookAt(snapAnimation.center);
      if (t >= 1) {
        camera.up.copy(snapAnimation.endUp);
        camera.zoom = snapAnimation.endZoom ?? camera.zoom;
        camera.updateProjectionMatrix();
        camera.lookAt(snapAnimation.center);
        snapAnimation = null;
      }
    }
    if (autoRotateEnabled && sceneBounds && !dragRotate && !gizmoDrag && !snapAnimation) {
      rotateCameraAroundCenter(
        camera,
        controls,
        sceneBounds.center,
        new THREE.Vector3(0, 0, 1),
        autoRotateSpeedRadPerSec * deltaSeconds,
      );
    }
    controls.update();
    if (showMeasurementFrame && measurementFrame) {
      measurementFrame.visible = true;
      updateMeasurementFrameLayout(measurementFrame, camera, renderer);
      updateScreenSizedSprites(measurementFrame, camera, renderer);
    } else if (measurementFrame) {
      measurementFrame.visible = false;
    }
    renderer.render(scene, camera);
    gizmo.syncFromCamera(camera);
    gizmo.render();
    requestAnimationFrame(frame);
  }
  resize(renderer, camera, canvasHost);
  frame();

  return {
    setScene(nextScene) {
      sceneSpec = nextScene;
      dragRotate = null;
      hydrate(nextScene);
    },
    destroy() {
      active = false;
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      gizmo.destroy();
      el.replaceChildren();
    },
  };
}

export { mountZView };
