import * as THREE from "three";
import { useRef, useEffect, useState, useCallback } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import {
  mergeGeometries,
  mergeVertices,
} from "three/examples/jsm/utils/BufferGeometryUtils.js";
import GPGPU from "./GPGPU";
import GPGPUUtils from "./GPGPUUtils";

const MODEL_ROTATION_INTERVAL = 3200;
const MODEL_RETIRE_DURATION_MS = 700;
const INCOMING_START_DELAY_MS = 260;
const INCOMING_FADE_DURATION_MS = 1100;
const MAX_RETIRING_SYSTEMS = 1;
const RETIRING_COMPUTE_INTERVAL_MS = 50;
const PREWARM_LEAD_MS = 2400;
const PREWARM_WARMUP_STEPS = 2;
const PREWARM_CHECK_INTERVAL_MS = 120;
const PERF_LOG_INTERVAL_MS = 2000;
const PERF_JANK_THRESHOLD_MS = 19.5;
const PERF_ENABLED_QUERY_PARAM = "particlePerf";
const PERF_ENABLED_STORAGE_KEY = "particlePerf";
const PERF_DEFAULT_ENABLED = Boolean(import.meta.env?.DEV);
const PERF_HISTORY_LIMIT = 240;

function createPerfMetricsWindow(startAt = 0) {
  return {
    windowStartAt: startAt,
    frameCount: 0,
    frameDeltaMsSum: 0,
    frameDeltaMsMax: 0,
    jankFrameCount: 0,
    cpuFrameMsSum: 0,
    cpuFrameMsMax: 0,
    warmupComputeCount: 0,
    warmupComputeMsSum: 0,
    warmupComputeMsMax: 0,
    activeComputeCount: 0,
    activeComputeMsSum: 0,
    activeComputeMsMax: 0,
    retiringComputeCount: 0,
    retiringComputeMsSum: 0,
    retiringComputeMsMax: 0,
    pointerMoveEvents: 0,
    pointerEnterEvents: 0,
    pointerDownEvents: 0,
    pointerUpEvents: 0,
    pointerLeaveEvents: 0,
    pointerRayUpdatesFromEvents: 0,
    pointerRayUpdatesFromFrame: 0,
    retiringSystemsSum: 0,
    retiringSystemsMax: 0,
    transitionFrameCount: 0,
    renderCallsSum: 0,
    renderPointsSum: 0,
    renderTrianglesSum: 0,
  };
}

function resetPerfMetricsWindow(metrics, startAt = performance.now()) {
  Object.assign(metrics, createPerfMetricsWindow(startAt));
}

function safeAverage(sum, count) {
  return count > 0 ? sum / count : 0;
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function buildPerfSnapshot({
  metrics,
  endAt,
  renderInfo,
  memoryInfo,
  activeSystemCount,
  retiringSystemCount,
  transitionActive,
  pointerActive,
  computeTextureTypeLabel,
}) {
  const windowMs = Math.max(endAt - metrics.windowStartAt, 1);
  const windowSeconds = windowMs / 1000;
  const frameCount = Math.max(metrics.frameCount, 1);
  const totalPointerEvents =
    metrics.pointerMoveEvents +
    metrics.pointerEnterEvents +
    metrics.pointerDownEvents +
    metrics.pointerUpEvents +
    metrics.pointerLeaveEvents;

  return {
    windowMs: round(windowMs, 0),
    fpsAvg: round(metrics.frameCount / windowSeconds, 1),
    frameMsAvg: round(
      safeAverage(metrics.frameDeltaMsSum, metrics.frameCount),
      2
    ),
    frameMsMax: round(metrics.frameDeltaMsMax, 2),
    frameJankPct: round(
      safeAverage(metrics.jankFrameCount * 100, metrics.frameCount),
      1
    ),
    cpuFrameMsAvg: round(
      safeAverage(metrics.cpuFrameMsSum, metrics.frameCount),
      3
    ),
    cpuFrameMsMax: round(metrics.cpuFrameMsMax, 3),
    activeComputeMsAvg: round(
      safeAverage(metrics.activeComputeMsSum, metrics.activeComputeCount),
      3
    ),
    activeComputeMsMax: round(metrics.activeComputeMsMax, 3),
    retiringComputeMsAvg: round(
      safeAverage(metrics.retiringComputeMsSum, metrics.retiringComputeCount),
      3
    ),
    retiringComputeMsMax: round(metrics.retiringComputeMsMax, 3),
    warmupComputeMsAvg: round(
      safeAverage(metrics.warmupComputeMsSum, metrics.warmupComputeCount),
      3
    ),
    warmupComputeMsMax: round(metrics.warmupComputeMsMax, 3),
    activeComputeCalls: metrics.activeComputeCount,
    retiringComputeCalls: metrics.retiringComputeCount,
    warmupComputeCalls: metrics.warmupComputeCount,
    pointerEventsPerSec: round(totalPointerEvents / windowSeconds, 1),
    pointerMovePerSec: round(metrics.pointerMoveEvents / windowSeconds, 1),
    pointerRayUpdatesFromEvents: metrics.pointerRayUpdatesFromEvents,
    pointerRayUpdatesFromFrame: metrics.pointerRayUpdatesFromFrame,
    retiringSystemsAvg: round(
      safeAverage(metrics.retiringSystemsSum, metrics.frameCount),
      2
    ),
    retiringSystemsMax: metrics.retiringSystemsMax,
    transitionFramePct: round(
      safeAverage(metrics.transitionFrameCount * 100, metrics.frameCount),
      1
    ),
    renderCallsAvg: round(safeAverage(metrics.renderCallsSum, frameCount), 1),
    renderPointsAvg: round(safeAverage(metrics.renderPointsSum, frameCount), 0),
    renderTrianglesAvg: round(
      safeAverage(metrics.renderTrianglesSum, frameCount),
      0
    ),
    renderCallsNow: renderInfo?.calls ?? 0,
    renderPointsNow: renderInfo?.points ?? 0,
    renderTrianglesNow: renderInfo?.triangles ?? 0,
    texturesNow: memoryInfo?.textures ?? 0,
    geometriesNow: memoryInfo?.geometries ?? 0,
    activeSystemsNow: activeSystemCount,
    retiringSystemsNow: retiringSystemCount,
    transitionActiveNow: transitionActive,
    pointerActiveNow: pointerActive,
    computeTextureTypeNow: computeTextureTypeLabel ?? "unknown",
  };
}

const models = [
  {
    path: "/models/whale.obj",
    color: [0.3, 0.6, 0.9],
    scale: [4.5, 4.5, 4.5],
    rotation: [Math.PI, 1.1, -1.56],
  },
  {
    path: "/models/dancer.obj",
    color: [1, 0.8, 0.5],
    scale: [4.5, 4.5, 4.5],
    rotation: [Math.PI / 12, 0, 0],
    //rotation: [Math.PI / 0.1, -1.25, 0],
  },
  {
    path: "/models/horse.obj",
    color: [0.5, 0.4, 0.8],
    scale: [4.5, 4.5, 4.5],
    rotation: [Math.PI / 1.9, Math.PI / 1.7, Math.PI],
  },
  {
    path: "/models/elephant.obj",
    color: [0.8, 0.2, 0.4],
    scale: [4.5, 4.5, 4.5],
    rotation: [Math.PI / 2.1, -0.9, 3.15],
  },
  {
    path: "/models/air-balloon.obj",
    color: [0.4, 0.6, 0.5],
    scale: [4.5, 4.5, 4.5],
    rotation: [Math.PI, 1.8, -1.56],
  },
  // {
  //   path: "/models/dog.obj",
  //   color: [1, 0.6, 0.5],
  //   scale: [4.5, 4.5, 4.5],
  //   rotation: [Math.PI, 1.8, -1.56, 0],
  // },
];

function disposeParticleData(particleData) {
  particleData?.positionTexture?.dispose?.();
}

function disposeMeshDataList(dataList) {
  for (const entry of dataList) {
    disposeParticleData(entry?.particleData);
  }
}

export function GPGPUParticleMesh({ isMobile, onInitialReady }) {
  const particleSize = isMobile ? 512 : 1024;
  const gpgpuRef = useRef(null);
  const prewarmedGpgpuRef = useRef(null);
  const recyclableGpgpusRef = useRef([]);
  const prewarmWarmupQueueRef = useRef([]);
  const prewarmTimeoutRef = useRef(null);
  const activeTransitionRef = useRef(null);
  const retiringGpgpusRef = useRef([]);
  const isUnmountingRef = useRef(false);
  const hasNotifiedInitialReady = useRef(false);
  const pointerActiveRef = useRef(false);
  const pointerUpdatePendingRef = useRef(false);
  const activeTouchPointersRef = useRef(new Set());
  const mouseStateRef = useRef({
    cursorPosition: new THREE.Vector3(9999, 9999, 9999),
    rayOrigin: new THREE.Vector3(9999, 9999, 9999),
    rayDirection: new THREE.Vector3(0, 0, 0),
    previousRayOrigin: new THREE.Vector3(9999, 9999, 9999),
    previousRayDirection: new THREE.Vector3(0, 0, 0),
    speed: 0,
  });
  const targetRayOriginRef = useRef(new THREE.Vector3(9999, 9999, 9999));
  const targetRayDirectionRef = useRef(new THREE.Vector3(0, 0, 0));
  const raycasterRef = useRef(new THREE.Raycaster());
  const pointerNdcRef = useRef(new THREE.Vector2());
  const canvasRectRef = useRef({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });
  const hitPointRef = useRef(new THREE.Vector3());
  const planeNormalRef = useRef(new THREE.Vector3());
  const interactionPlaneRef = useRef(new THREE.Plane());
  const planeOriginRef = useRef(new THREE.Vector3(0, 0, 0));
  const previousMouseWorldRef = useRef(null);
  const initialResolutionRef = useRef(new THREE.Vector2());
  const meshDataListRef = useRef([]);
  const perfEnabledRef = useRef(false);
  const perfMetricsRef = useRef(createPerfMetricsWindow(0));
  const perfLatestSnapshotRef = useRef(null);
  const perfHistoryRef = useRef([]);
  const { camera, gl, scene, size } = useThree();
  const [meshDataList, setMeshDataList] = useState([]);
  const [modelIndex, setModelIndex] = useState(0);

  const getComputeTextureTypeLabel = useCallback((gpgpu) => {
    if (!gpgpu) return "none";
    if (gpgpu.computeTextureType === THREE.HalfFloatType) {
      return "HalfFloatType";
    }
    if (gpgpu.computeTextureType === THREE.FloatType) {
      return "FloatType";
    }
    return "unknown";
  }, []);

  useEffect(() => {
    meshDataListRef.current = meshDataList;
  }, [meshDataList]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    let enabled = PERF_DEFAULT_ENABLED;

    try {
      const params = new URLSearchParams(window.location.search);
      const queryValue = params.get(PERF_ENABLED_QUERY_PARAM);
      if (queryValue === "0" || queryValue === "1") {
        enabled = queryValue === "1";
      } else {
        const storedValue = window.localStorage.getItem(
          PERF_ENABLED_STORAGE_KEY
        );
        if (storedValue === "0" || storedValue === "1") {
          enabled = storedValue === "1";
        }
      }
    } catch {
      // Ignore storage/query failures and keep default behavior.
    }

    perfEnabledRef.current = enabled;
    resetPerfMetricsWindow(perfMetricsRef.current, performance.now());

    const setEnabled = (nextEnabled) => {
      const normalized = Boolean(nextEnabled);
      perfEnabledRef.current = normalized;
      resetPerfMetricsWindow(perfMetricsRef.current, performance.now());
      perfLatestSnapshotRef.current = null;
      perfHistoryRef.current = [];

      try {
        window.localStorage.setItem(
          PERF_ENABLED_STORAGE_KEY,
          normalized ? "1" : "0"
        );
      } catch {
        // Ignore storage failures; monitoring still works for this session.
      }

      console.info(
        `[ParticlePerf] monitor ${normalized ? "enabled" : "disabled"}.`
      );
    };

    const perfApi = {
      enable: () => setEnabled(true),
      disable: () => setEnabled(false),
      setEnabled,
      reset: () => {
        resetPerfMetricsWindow(perfMetricsRef.current, performance.now());
        perfLatestSnapshotRef.current = null;
        perfHistoryRef.current = [];
      },
      getLatestSnapshot: () => perfLatestSnapshotRef.current,
      getHistory: () => perfHistoryRef.current.slice(),
      getWorstSnapshots: (limit = 5) =>
        perfHistoryRef.current
          .slice()
          .sort((a, b) => (b.frameMsMax ?? 0) - (a.frameMsMax ?? 0))
          .slice(0, Math.max(1, Math.floor(limit))),
    };

    window.__particlePerf = perfApi;

    if (enabled) {
      console.info(
        "[ParticlePerf] monitor enabled. Use window.__particlePerf.getLatestSnapshot() for the latest report."
      );
    }

    return () => {
      if (window.__particlePerf === perfApi) {
        delete window.__particlePerf;
      }
    };
  }, []);

  const createGpgpuSystem = useCallback(
    (color, particleData) => {
      const initialResolution = initialResolutionRef.current;
      gl.getSize(initialResolution);

      return new GPGPU({
        size: particleSize,
        camera,
        renderer: gl,
        mouse: mouseStateRef.current,
        scene,
        model: null,
        color,
        sizes: { width: initialResolution.x, height: initialResolution.y },
        isMobile: isMobile,
        precomputedData: particleData,
      });
    },
    [camera, gl, isMobile, particleSize, scene]
  );

  const removeWarmupTask = useCallback((gpgpu) => {
    if (!gpgpu) return;
    prewarmWarmupQueueRef.current = prewarmWarmupQueueRef.current.filter(
      (task) => task.gpgpu !== gpgpu
    );
  }, []);

  const queueWarmup = useCallback(
    (gpgpu, steps) => {
      if (!gpgpu) return;

      const warmupSteps = Math.max(0, Math.floor(steps));
      if (warmupSteps <= 0) return;

      removeWarmupTask(gpgpu);
      prewarmWarmupQueueRef.current.push({
        gpgpu,
        remainingSteps: warmupSteps,
      });
    },
    [removeWarmupTask]
  );

  const recycleGpgpuSystem = useCallback(
    (gpgpu) => {
      if (!gpgpu) return;
      if (recyclableGpgpusRef.current.includes(gpgpu)) return;

      removeWarmupTask(gpgpu);
      gpgpu.setVisible(false);
      gpgpu.setOpacity(0);
      recyclableGpgpusRef.current.push(gpgpu);
    },
    [removeWarmupTask]
  );

  const acquireGpgpuSystem = useCallback(
    (color, particleData) => {
      const recycled = recyclableGpgpusRef.current.pop();
      if (recycled) {
        recycled.reseed({ precomputedData: particleData, color });
        return recycled;
      }

      return createGpgpuSystem(color, particleData);
    },
    [createGpgpuSystem]
  );

  useEffect(() => {
    const canvas = gl.domElement;
    const activeTouchPointers = activeTouchPointersRef.current;
    const canvasRect = canvasRectRef.current;
    const recordPerfCounter = (counterKey) => {
      if (!perfEnabledRef.current) return;
      perfMetricsRef.current[counterKey] += 1;
    };

    function updateCanvasRect() {
      const rect = canvas.getBoundingClientRect();
      canvasRect.left = rect.left;
      canvasRect.top = rect.top;
      canvasRect.width = rect.width;
      canvasRect.height = rect.height;
    }

    updateCanvasRect();

    function deactivatePointer() {
      const mouseState = mouseStateRef.current;
      pointerActiveRef.current = false;
      pointerUpdatePendingRef.current = false;
      targetRayOriginRef.current.set(9999, 9999, 9999);
      targetRayDirectionRef.current.set(0, 0, 0);
      mouseState.previousRayOrigin.copy(mouseState.rayOrigin);
      mouseState.previousRayDirection.copy(mouseState.rayDirection);
      mouseState.speed = 0;
      previousMouseWorldRef.current = null;
    }

    function updatePointerTarget(event) {
      if (event.pointerType === "touch") {
        const activeTouches = activeTouchPointers.size;
        if (!event.isPrimary || activeTouches !== 1) {
          deactivatePointer();
          return;
        }
      }

      const rect = canvasRect;
      if (rect.width === 0 || rect.height === 0) return;

      pointerNdcRef.current.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      const wasActive = pointerActiveRef.current;
      pointerActiveRef.current = true;

      if (!wasActive) {
        const raycaster = raycasterRef.current;
        raycaster.setFromCamera(pointerNdcRef.current, camera);
        targetRayOriginRef.current.copy(raycaster.ray.origin);
        targetRayDirectionRef.current.copy(raycaster.ray.direction);
        const mouseState = mouseStateRef.current;
        mouseState.rayOrigin.copy(raycaster.ray.origin);
        mouseState.rayDirection.copy(raycaster.ray.direction);
        mouseState.previousRayOrigin.copy(raycaster.ray.origin);
        mouseState.previousRayDirection.copy(raycaster.ray.direction);
        mouseState.speed = 0;
        previousMouseWorldRef.current = null;
        pointerUpdatePendingRef.current = false;
        recordPerfCounter("pointerRayUpdatesFromEvents");
      } else {
        pointerUpdatePendingRef.current = true;
      }
    }

    function handlePointerMove(event) {
      recordPerfCounter("pointerMoveEvents");
      updatePointerTarget(event);
    }

    function handlePointerEnter(event) {
      recordPerfCounter("pointerEnterEvents");
      updateCanvasRect();
      updatePointerTarget(event);
    }

    function handlePointerDown(event) {
      recordPerfCounter("pointerDownEvents");
      updateCanvasRect();
      if (event.pointerType === "touch") {
        activeTouchPointers.add(event.pointerId);
      }
      updatePointerTarget(event);
    }

    function handlePointerLeave(event) {
      recordPerfCounter("pointerLeaveEvents");
      if (event?.pointerType === "touch") {
        activeTouchPointers.delete(event.pointerId);
      }
      deactivatePointer();
    }

    function handleWindowBlur() {
      activeTouchPointers.clear();
      handlePointerLeave();
    }

    function handleVisibilityChange() {
      if (document.hidden) {
        activeTouchPointers.clear();
        handlePointerLeave();
      }
    }

    function handlePointerUp(event) {
      recordPerfCounter("pointerUpEvents");
      if (event.pointerType === "touch") {
        activeTouchPointers.delete(event.pointerId);
      }
      deactivatePointer();
    }

    canvas.addEventListener("pointermove", handlePointerMove, {
      passive: true,
    });
    canvas.addEventListener("pointerenter", handlePointerEnter, {
      passive: true,
    });
    canvas.addEventListener("pointerdown", handlePointerDown, {
      passive: true,
    });
    canvas.addEventListener("pointerup", handlePointerUp, { passive: true });
    canvas.addEventListener("pointercancel", handlePointerUp, {
      passive: true,
    });
    canvas.addEventListener("pointerleave", handlePointerLeave, {
      passive: true,
    });
    window.addEventListener("blur", handleWindowBlur);
    window.addEventListener("resize", updateCanvasRect, { passive: true });
    window.addEventListener("scroll", updateCanvasRect, { passive: true });
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerenter", handlePointerEnter);
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointerup", handlePointerUp);
      canvas.removeEventListener("pointercancel", handlePointerUp);
      canvas.removeEventListener("pointerleave", handlePointerLeave);
      window.removeEventListener("blur", handleWindowBlur);
      window.removeEventListener("resize", updateCanvasRect);
      window.removeEventListener("scroll", updateCanvasRect);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      activeTouchPointers.clear();
    };
  }, [camera, gl]);

  // Preload all models on mount
  useEffect(() => {
    let cancelled = false;

    async function preloadModels() {
      const results = [];
      const objLoader = new OBJLoader();
      const gltfLoader = new GLTFLoader();
      const center = new THREE.Vector3();
      const sizeVec = new THREE.Vector3();

      for (const model of models) {
        if (cancelled || isUnmountingRef.current) break;

        const geometries = [];

        try {
          if (model.path.endsWith(".glb") || model.path.endsWith(".gltf")) {
            const gltf = await gltfLoader.loadAsync(model.path);
            gltf.scene.updateMatrixWorld(true);
            gltf.scene.traverse((child) => {
              if (child.isMesh) {
                const geom = child.geometry.clone();
                geom.applyMatrix4(child.matrixWorld);
                geometries.push(geom);
              }
            });
          } else if (model.path.endsWith(".obj")) {
            const object = await objLoader.loadAsync(model.path);
            object.updateMatrixWorld(true);
            object.traverse((child) => {
              if (child.isMesh) {
                const geom = child.geometry.clone();
                geom.applyMatrix4(child.matrixWorld);
                geometries.push(geom);
              }
            });
          }
        } catch (err) {
          console.warn("Failed to load model:", model.path, err);
          continue;
        }

        if (!geometries.length) continue;

        let geometry;
        try {
          geometry = mergeGeometries(geometries, false);
        } catch (err) {
          console.warn("mergeGeometries failed for", model.path, err);
          geometries.forEach((geom) => geom.dispose());
          continue;
        }
        geometries.forEach((geom) => geom.dispose());

        const dedupedGeometry = mergeVertices(geometry, 1e-4);
        if (dedupedGeometry !== geometry) {
          geometry.dispose();
        }
        geometry = dedupedGeometry;

        // Center and scale to uniform size
        geometry.computeBoundingBox();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        geometry.computeBoundingBox();
        geometry.boundingBox.getSize(sizeVec);
        const maxAxis = Math.max(sizeVec.x, sizeVec.y, sizeVec.z);
        geometry.scale(1 / maxAxis, 1 / maxAxis, 1 / maxAxis);

        if (model.scale) geometry.scale(...model.scale);
        if (model.rotation) {
          geometry.rotateX(model.rotation[0]);
          geometry.rotateY(model.rotation[1]);
          geometry.rotateZ(model.rotation[2]);
        }

        const mesh = new THREE.Mesh(geometry);
        mesh.updateMatrixWorld(true);

        const particleData = GPGPUUtils.generateParticleData(
          mesh,
          particleSize
        );

        if (
          particleData.positionTexture &&
          typeof gl.initTexture === "function"
        ) {
          gl.initTexture(particleData.positionTexture);
        }
        mesh.geometry.dispose();
        results.push({ color: model.color, particleData });
      }

      if (cancelled || isUnmountingRef.current) {
        disposeMeshDataList(results);
        return;
      }

      if (results.length === 0 && !hasNotifiedInitialReady.current) {
        hasNotifiedInitialReady.current = true;
        onInitialReady?.();
      }

      setMeshDataList(results);
    }

    preloadModels();
    return () => {
      cancelled = true;
    };
  }, [gl, onInitialReady, particleSize]);

  // Cycle through preloaded models
  useEffect(() => {
    if (meshDataList.length < 2) return;

    let timeoutId = null;
    let cancelled = false;

    const scheduleNextCheck = (delayMs) => {
      timeoutId = setTimeout(() => {
        if (cancelled) return;

        const nextIndex = (modelIndex + 1) % meshDataList.length;
        const prewarmed = prewarmedGpgpuRef.current;
        const isNextReady =
          prewarmed !== null &&
          prewarmed !== undefined &&
          prewarmed.index === nextIndex &&
          prewarmed.ready;

        if (!isNextReady) {
          scheduleNextCheck(PREWARM_CHECK_INTERVAL_MS);
          return;
        }

        setModelIndex(nextIndex);
      }, delayMs);
    };

    scheduleNextCheck(MODEL_ROTATION_INTERVAL);

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    };
  }, [meshDataList, modelIndex]);

  // Load GPGPU when mesh is ready
  useEffect(() => {
    if (!meshDataList.length) return;

    const { color, particleData } = meshDataList[modelIndex];
    const prewarmed = prewarmedGpgpuRef.current;

    let gpgpu;
    if (prewarmed && prewarmed.index === modelIndex) {
      gpgpu = prewarmed.gpgpu;
      removeWarmupTask(gpgpu);
      prewarmedGpgpuRef.current = null;
      gpgpu.setVisible(true);
      gpgpu.resetTiming();
    } else {
      gpgpu = acquireGpgpuSystem(color, particleData);
    }

    gpgpuRef.current = gpgpu;
    gpgpu.setVisible(true);
    gpgpu.setOpacity(0);
    const transitionStart = performance.now() + INCOMING_START_DELAY_MS;
    activeTransitionRef.current = {
      startAt: transitionStart,
      fadeStartAt: transitionStart,
      fadeDuration: INCOMING_FADE_DURATION_MS,
    };

    if (!hasNotifiedInitialReady.current) {
      hasNotifiedInitialReady.current = true;
      onInitialReady?.();
    }

    return () => {
      if (gpgpuRef.current === gpgpu) {
        gpgpuRef.current = null;
        activeTransitionRef.current = null;
      }

      if (isUnmountingRef.current) {
        gpgpu.dispose();
        return;
      }

      gpgpu.setOpacity(1);
      const retiringSystems = retiringGpgpusRef.current;
      const retireStartAt = performance.now() + INCOMING_START_DELAY_MS;
      retiringSystems.push({
        gpgpu,
        retireStartAt,
        retireAt: retireStartAt + MODEL_RETIRE_DURATION_MS,
        lastComputeAt: 0,
      });

      while (retiringSystems.length > MAX_RETIRING_SYSTEMS) {
        const oldestRetired = retiringSystems.shift();
        if (oldestRetired?.gpgpu) {
          recycleGpgpuSystem(oldestRetired.gpgpu);
        }
      }
    };
  }, [
    acquireGpgpuSystem,
    camera,
    gl,
    isMobile,
    meshDataList,
    modelIndex,
    removeWarmupTask,
    onInitialReady,
    particleSize,
    recycleGpgpuSystem,
    scene,
  ]);

  useEffect(() => {
    if (meshDataList.length < 2) return;

    if (prewarmTimeoutRef.current !== null) {
      clearTimeout(prewarmTimeoutRef.current);
      prewarmTimeoutRef.current = null;
    }

    const nextIndex = (modelIndex + 1) % meshDataList.length;
    const prewarmDelay = Math.max(MODEL_ROTATION_INTERVAL - PREWARM_LEAD_MS, 0);

    function createPrewarmed() {
      if (isUnmountingRef.current) return;

      const existing = prewarmedGpgpuRef.current;
      if (existing && existing.index === nextIndex) return;

      if (existing) {
        recycleGpgpuSystem(existing.gpgpu);
        prewarmedGpgpuRef.current = null;
      }

      const { color, particleData } = meshDataList[nextIndex];
      const prewarmedGpgpu = acquireGpgpuSystem(color, particleData);

      prewarmedGpgpu.setOpacity(0);
      prewarmedGpgpu.setVisible(false);
      queueWarmup(prewarmedGpgpu, PREWARM_WARMUP_STEPS);
      prewarmedGpgpuRef.current = {
        index: nextIndex,
        gpgpu: prewarmedGpgpu,
        ready: PREWARM_WARMUP_STEPS <= 0,
      };
    }

    prewarmTimeoutRef.current = setTimeout(createPrewarmed, prewarmDelay);

    return () => {
      if (prewarmTimeoutRef.current !== null) {
        clearTimeout(prewarmTimeoutRef.current);
        prewarmTimeoutRef.current = null;
      }
    };
  }, [
    acquireGpgpuSystem,
    camera,
    gl,
    isMobile,
    meshDataList,
    modelIndex,
    particleSize,
    queueWarmup,
    recycleGpgpuSystem,
    scene,
  ]);

  useEffect(() => {
    return () => {
      isUnmountingRef.current = true;

      if (gpgpuRef.current) {
        gpgpuRef.current.dispose();
        gpgpuRef.current = null;
      }
      activeTransitionRef.current = null;

      if (prewarmTimeoutRef.current !== null) {
        clearTimeout(prewarmTimeoutRef.current);
        prewarmTimeoutRef.current = null;
      }

      if (prewarmedGpgpuRef.current) {
        prewarmedGpgpuRef.current.gpgpu.dispose();
        prewarmedGpgpuRef.current = null;
      }
      prewarmWarmupQueueRef.current = [];

      for (const { gpgpu } of retiringGpgpusRef.current) {
        gpgpu.dispose();
      }
      retiringGpgpusRef.current = [];

      for (const gpgpu of recyclableGpgpusRef.current) {
        gpgpu.dispose();
      }
      recyclableGpgpusRef.current = [];

      disposeMeshDataList(meshDataListRef.current);
      meshDataListRef.current = [];
    };
  }, []);

  useEffect(() => {
    gpgpuRef.current?.setSize(size.width, size.height);
    prewarmedGpgpuRef.current?.gpgpu?.setSize(size.width, size.height);
    for (const { gpgpu } of retiringGpgpusRef.current) {
      gpgpu.setSize(size.width, size.height);
    }
    for (const gpgpu of recyclableGpgpusRef.current) {
      gpgpu.setSize(size.width, size.height);
    }
  }, [size.height, size.width]);

  // Run compute every frame
  useFrame((state, delta) => {
    const perfEnabled = perfEnabledRef.current;
    const perfMetrics = perfMetricsRef.current;
    const frameStartAt = perfEnabled ? performance.now() : 0;
    const mouseState = mouseStateRef.current;

    if (perfEnabled) {
      const deltaMs = delta * 1000;
      perfMetrics.frameCount += 1;
      perfMetrics.frameDeltaMsSum += deltaMs;
      if (deltaMs > perfMetrics.frameDeltaMsMax) {
        perfMetrics.frameDeltaMsMax = deltaMs;
      }
      if (deltaMs >= PERF_JANK_THRESHOLD_MS) {
        perfMetrics.jankFrameCount += 1;
      }
    }

    if (pointerActiveRef.current && pointerUpdatePendingRef.current) {
      const raycaster = raycasterRef.current;
      raycaster.setFromCamera(pointerNdcRef.current, camera);
      targetRayOriginRef.current.copy(raycaster.ray.origin);
      targetRayDirectionRef.current.copy(raycaster.ray.direction);
      pointerUpdatePendingRef.current = false;
      if (perfEnabled) {
        perfMetrics.pointerRayUpdatesFromFrame += 1;
      }
    }

    const cursorSpeed = Math.min(mouseState.speed ?? 0, 1);
    const baseFollowRate = THREE.MathUtils.lerp(44.0, 24.0, cursorSpeed);
    let catchup = 0;

    if (pointerActiveRef.current) {
      const originDelta = mouseState.rayOrigin.distanceTo(
        targetRayOriginRef.current
      );
      const currentDirLenSq = mouseState.rayDirection.lengthSq();
      const targetDirLenSq = targetRayDirectionRef.current.lengthSq();
      let dirDelta = 0;

      if (currentDirLenSq > 1e-10 && targetDirLenSq > 1e-10) {
        const dirDot = THREE.MathUtils.clamp(
          mouseState.rayDirection.dot(targetRayDirectionRef.current) /
            Math.sqrt(currentDirLenSq * targetDirLenSq),
          -1,
          1
        );
        dirDelta = Math.acos(dirDot);
      }

      catchup = Math.min(originDelta * 0.35 + dirDelta * 10.0, 1);
    }

    const followRate = baseFollowRate + catchup * 18.0;
    const follow = 1 - Math.exp(-delta * followRate);

    mouseState.previousRayOrigin.copy(mouseState.rayOrigin);
    mouseState.previousRayDirection.copy(mouseState.rayDirection);

    mouseState.rayOrigin.lerp(targetRayOriginRef.current, follow);
    mouseState.rayDirection.lerp(targetRayDirectionRef.current, follow);

    if (mouseState.rayDirection.lengthSq() > 1e-10) {
      mouseState.rayDirection.normalize();

      camera.getWorldDirection(planeNormalRef.current);
      interactionPlaneRef.current.setFromNormalAndCoplanarPoint(
        planeNormalRef.current,
        planeOriginRef.current
      );

      raycasterRef.current.ray.origin.copy(mouseState.rayOrigin);
      raycasterRef.current.ray.direction.copy(mouseState.rayDirection);
      const worldPoint = raycasterRef.current.ray.intersectPlane(
        interactionPlaneRef.current,
        hitPointRef.current
      );

      if (worldPoint) {
        mouseState.cursorPosition.copy(worldPoint);

        if (previousMouseWorldRef.current) {
          const distance = worldPoint.distanceTo(previousMouseWorldRef.current);
          const speedNow =
            delta > 0 ? Math.min((distance / delta) * 0.04, 1) : 0;
          mouseState.speed = Math.max(mouseState.speed * 0.82, speedNow);
        } else {
          mouseState.speed = Math.max(mouseState.speed, 0.2);
          previousMouseWorldRef.current = worldPoint.clone();
        }

        previousMouseWorldRef.current.copy(worldPoint);
      } else if (!pointerActiveRef.current) {
        mouseState.cursorPosition.set(9999, 9999, 9999);
        mouseState.speed = 0;
        previousMouseWorldRef.current = null;
      }
    } else {
      if (!pointerActiveRef.current) {
        mouseState.cursorPosition.set(9999, 9999, 9999);
        mouseState.speed = 0;
      }
    }

    if (!pointerActiveRef.current) {
      mouseState.speed *= 0.85;
      if (mouseState.speed < 0.0001) {
        mouseState.speed = 0;
      }
    }

    const elapsed = state.clock.elapsedTime;
    const now = performance.now();

    const warmupQueue = prewarmWarmupQueueRef.current;
    if (warmupQueue.length > 0) {
      const task = warmupQueue[0];
      if (!task?.gpgpu || task.gpgpu.disposed) {
        warmupQueue.shift();
      } else {
        const warmupStartAt = perfEnabled ? performance.now() : 0;
        task.gpgpu.warmup(1);
        if (perfEnabled) {
          const warmupDuration = performance.now() - warmupStartAt;
          perfMetrics.warmupComputeCount += 1;
          perfMetrics.warmupComputeMsSum += warmupDuration;
          if (warmupDuration > perfMetrics.warmupComputeMsMax) {
            perfMetrics.warmupComputeMsMax = warmupDuration;
          }
        }
        task.remainingSteps -= 1;

        if (task.remainingSteps <= 0) {
          warmupQueue.shift();
          if (prewarmedGpgpuRef.current?.gpgpu === task.gpgpu) {
            prewarmedGpgpuRef.current.ready = true;
          }
        }
      }
    }

    if (gpgpuRef.current) {
      const transition = activeTransitionRef.current;
      if (transition && now < transition.startAt) {
        gpgpuRef.current.setOpacity(0);
      } else {
        if (transition) {
          const fadeT =
            transition.fadeDuration > 0
              ? Math.min(
                  Math.max(
                    (now - transition.fadeStartAt) / transition.fadeDuration,
                    0
                  ),
                  1
                )
              : 1;
          gpgpuRef.current.setOpacity(fadeT);

          if (fadeT >= 1) {
            activeTransitionRef.current = null;
          }
        } else {
          gpgpuRef.current.setOpacity(1);
        }

        const activeComputeStartAt = perfEnabled ? performance.now() : 0;
        gpgpuRef.current.compute(elapsed, now);
        if (perfEnabled) {
          const activeComputeDuration =
            performance.now() - activeComputeStartAt;
          perfMetrics.activeComputeCount += 1;
          perfMetrics.activeComputeMsSum += activeComputeDuration;
          if (activeComputeDuration > perfMetrics.activeComputeMsMax) {
            perfMetrics.activeComputeMsMax = activeComputeDuration;
          }
        }
      }
    }

    for (let i = retiringGpgpusRef.current.length - 1; i >= 0; i--) {
      const retired = retiringGpgpusRef.current[i];
      const shouldComputeRetired =
        retired.lastComputeAt === 0 ||
        now - retired.lastComputeAt >= RETIRING_COMPUTE_INTERVAL_MS;

      if (shouldComputeRetired) {
        const retiringComputeStartAt = perfEnabled ? performance.now() : 0;
        retired.gpgpu.compute(elapsed, now);
        if (perfEnabled) {
          const retiringComputeDuration =
            performance.now() - retiringComputeStartAt;
          perfMetrics.retiringComputeCount += 1;
          perfMetrics.retiringComputeMsSum += retiringComputeDuration;
          if (retiringComputeDuration > perfMetrics.retiringComputeMsMax) {
            perfMetrics.retiringComputeMsMax = retiringComputeDuration;
          }
        }
        retired.lastComputeAt = now;
      }

      const retireProgress = THREE.MathUtils.clamp(
        (now - retired.retireStartAt) / MODEL_RETIRE_DURATION_MS,
        0,
        1
      );
      retired.gpgpu.setOpacity(1 - retireProgress);

      if (now >= retired.retireAt) {
        recycleGpgpuSystem(retired.gpgpu);
        retiringGpgpusRef.current.splice(i, 1);
      }
    }

    if (perfEnabled) {
      const retiringCountNow = retiringGpgpusRef.current.length;
      perfMetrics.retiringSystemsSum += retiringCountNow;
      if (retiringCountNow > perfMetrics.retiringSystemsMax) {
        perfMetrics.retiringSystemsMax = retiringCountNow;
      }
      if (activeTransitionRef.current) {
        perfMetrics.transitionFrameCount += 1;
      }

      const renderInfo = gl.info.render;
      perfMetrics.renderCallsSum += renderInfo.calls;
      perfMetrics.renderPointsSum += renderInfo.points;
      perfMetrics.renderTrianglesSum += renderInfo.triangles;

      const frameCpuDuration = performance.now() - frameStartAt;
      perfMetrics.cpuFrameMsSum += frameCpuDuration;
      if (frameCpuDuration > perfMetrics.cpuFrameMsMax) {
        perfMetrics.cpuFrameMsMax = frameCpuDuration;
      }

      const windowDuration = now - perfMetrics.windowStartAt;
      if (windowDuration >= PERF_LOG_INTERVAL_MS) {
        const snapshot = buildPerfSnapshot({
          metrics: perfMetrics,
          endAt: now,
          renderInfo,
          memoryInfo: gl.info.memory,
          activeSystemCount: gpgpuRef.current ? 1 : 0,
          retiringSystemCount: retiringCountNow,
          transitionActive: Boolean(activeTransitionRef.current),
          pointerActive: pointerActiveRef.current,
          computeTextureTypeLabel: getComputeTextureTypeLabel(gpgpuRef.current),
        });

        perfLatestSnapshotRef.current = snapshot;
        const history = perfHistoryRef.current;
        history.push(snapshot);
        if (history.length > PERF_HISTORY_LIMIT) {
          history.splice(0, history.length - PERF_HISTORY_LIMIT);
        }
        console.log("[ParticlePerf]", snapshot);
        console.log("[ParticlePerfJSON]", JSON.stringify(snapshot));
        resetPerfMetricsWindow(perfMetrics, now);
      }
    }
  });

  return null;
}
