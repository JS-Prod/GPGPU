import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { GPGPUParticleMesh } from "./gpgpu/GPGPUParticleMesh.js";
import "./App.css";

function ShiftCameraView({ isMobile }) {
  const { camera, size } = useThree();

  useEffect(() => {
    if (!isMobile) {
      const { width, height } = size;
      camera.setViewOffset(width, height, -width * 0.15, 0, width, height);
      camera.updateProjectionMatrix();
    }
  }, [camera, size, isMobile]);

  return null;
}

function checkDevice() {
  if (typeof navigator === "undefined") return false;

  const userAgent = navigator.userAgent || "";
  const uaMobile =
    /Mobi|Android|iPhone|iPad|iPod|BlackBerry|Windows Phone/i.test(userAgent);
  const uaDataMobile = Boolean(navigator.userAgentData?.mobile);
  const iPadDesktopUa =
    /Macintosh/i.test(userAgent) && navigator.maxTouchPoints > 1;
  const coarsePointer =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches;
  const touchCapable = navigator.maxTouchPoints > 0;

  return (
    uaMobile || uaDataMobile || iPadDesktopUa || (touchCapable && coarsePointer)
  );
}

const CAMERA_DAMPING_FACTOR = 0.08;
const CAMERA_ROTATE_SPEED = 0.65;
const CAMERA_ZOOM_SPEED = 0.75;
const CAMERA_PAN_SPEED = 0.75;
const TOUCH_NONE = -1;

export default function App() {
  const [isMobile, setIsMobile] = useState(() => checkDevice());
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [controlsInstance, setControlsInstance] = useState(null);
  const controlsRef = useRef(null);
  const resetAnimationFrameRef = useRef(null);

  const setControlsRef = useCallback((controls) => {
    controlsRef.current = controls;
    setControlsInstance(controls);
  }, []);

  const handleInitialReady = useCallback(() => {
    setIsInitialLoading(false);
  }, []);

  const cancelResetAnimation = useCallback(() => {
    if (resetAnimationFrameRef.current !== null) {
      cancelAnimationFrame(resetAnimationFrameRef.current);
      resetAnimationFrameRef.current = null;
    }

    const controls = controlsRef.current;
    if (controls) {
      controls.enabled = true;
    }
  }, []);

  const handleResetCamera = useCallback(() => {
    cancelResetAnimation();

    const controls = controlsRef.current;
    if (!controls) return;

    const camera = controls.object;
    const width = window.innerWidth;
    const height = window.innerHeight;
    const startPosition = camera.position.clone();
    const startTarget = controls.target.clone();
    const endPosition = new THREE.Vector3(0, 0, -5.5);
    const endTarget = new THREE.Vector3(0, 0, 0);
    const animationDurationMs = 700;

    if (!isMobile) {
      camera.setViewOffset(width, height, -width * 0.15, 0, width, height);
    } else if (camera.view !== null) {
      camera.clearViewOffset();
    }
    camera.updateProjectionMatrix();

    controls.enabled = false;
    const animationStart = performance.now();

    const animateReset = (now) => {
      const progress = Math.min(
        (now - animationStart) / animationDurationMs,
        1
      );
      const easedProgress = 1 - Math.pow(1 - progress, 3);

      camera.position.lerpVectors(startPosition, endPosition, easedProgress);
      controls.target.lerpVectors(startTarget, endTarget, easedProgress);
      controls.update();

      if (progress < 1) {
        resetAnimationFrameRef.current = requestAnimationFrame(animateReset);
        return;
      }

      controls.enabled = true;
      resetAnimationFrameRef.current = null;
    };

    resetAnimationFrameRef.current = requestAnimationFrame(animateReset);
  }, [cancelResetAnimation, isMobile]);

  useEffect(() => cancelResetAnimation, [cancelResetAnimation]);

  useEffect(() => {
    const updateMobileState = () => {
      setIsMobile(checkDevice());
    };

    updateMobileState();
    window.addEventListener("resize", updateMobileState, { passive: true });
    window.addEventListener("orientationchange", updateMobileState);

    return () => {
      window.removeEventListener("resize", updateMobileState);
      window.removeEventListener("orientationchange", updateMobileState);
    };
  }, []);

  useEffect(() => {
    if (!isMobile) return;

    const controls = controlsInstance;
    if (!controls) return;

    const canvas = controls.domElement;
    const gestureState = {
      lastCentroidX: 0,
      lastCentroidY: 0,
    };
    const panOffset = new THREE.Vector3();
    const axisX = new THREE.Vector3();
    const axisY = new THREE.Vector3();
    const cameraToTarget = new THREE.Vector3();

    const getTouchCentroid = (touchList) => {
      let x = 0;
      let y = 0;
      const count = touchList.length;
      if (count === 0) return { x: 0, y: 0 };

      for (let i = 0; i < count; i += 1) {
        x += touchList[i].clientX;
        y += touchList[i].clientY;
      }

      return {
        x: x / count,
        y: y / count,
      };
    };

    const panWithTouchDelta = (deltaX, deltaY) => {
      const camera = controls.object;
      if (!camera || !controls.target) return;

      panOffset.set(0, 0, 0);

      if (camera.isPerspectiveCamera) {
        cameraToTarget.subVectors(camera.position, controls.target);
        let targetDistance = cameraToTarget.length();
        targetDistance *= Math.tan((camera.fov * Math.PI) / 360);

        const panDeltaX = (2 * deltaX * targetDistance) / canvas.clientHeight;
        const panDeltaY = (2 * deltaY * targetDistance) / canvas.clientHeight;

        axisX.setFromMatrixColumn(camera.matrix, 0);
        axisY.setFromMatrixColumn(camera.matrix, 1);

        panOffset.addScaledVector(axisX, -panDeltaX);
        panOffset.addScaledVector(axisY, panDeltaY);
      } else if (camera.isOrthographicCamera) {
        const panDeltaX =
          (deltaX * (camera.right - camera.left)) /
          camera.zoom /
          canvas.clientWidth;
        const panDeltaY =
          (deltaY * (camera.top - camera.bottom)) /
          camera.zoom /
          canvas.clientHeight;

        axisX.setFromMatrixColumn(camera.matrix, 0);
        axisY.setFromMatrixColumn(camera.matrix, 1);

        panOffset.addScaledVector(axisX, -panDeltaX);
        panOffset.addScaledVector(axisY, panDeltaY);
      } else {
        return;
      }

      camera.position.add(panOffset);
      controls.target.add(panOffset);
      controls.update();
    };

    const setGestureMode = (touchCount) => {
      controls.touches.ONE = TOUCH_NONE;
      controls.touches.TWO = THREE.TOUCH.DOLLY_ROTATE;

      if (touchCount === 3) {
        controls.enabled = false;
        controls.enableRotate = false;
        controls.enableZoom = false;
        controls.enablePan = false;
        return;
      }

      controls.enabled = true;
      controls.enableRotate = true;
      controls.enableZoom = true;
      controls.enablePan = false;
    };

    const handleTouchStart = (event) => {
      const touchCount = event.touches.length;
      if (touchCount === 3) {
        const centroid = getTouchCentroid(event.touches);
        gestureState.lastCentroidX = centroid.x;
        gestureState.lastCentroidY = centroid.y;
        event.preventDefault();
      }

      setGestureMode(touchCount);
    };

    const handleTouchMove = (event) => {
      const touchCount = event.touches.length;

      if (touchCount === 3) {
        const centroid = getTouchCentroid(event.touches);
        const deltaX = centroid.x - gestureState.lastCentroidX;
        const deltaY = centroid.y - gestureState.lastCentroidY;

        gestureState.lastCentroidX = centroid.x;
        gestureState.lastCentroidY = centroid.y;
        panWithTouchDelta(deltaX, deltaY);
        event.preventDefault();
      }

      setGestureMode(touchCount);
    };

    const handleTouchEnd = (event) => {
      const touchCount = event.touches.length;
      if (touchCount === 3) {
        const centroid = getTouchCentroid(event.touches);
        gestureState.lastCentroidX = centroid.x;
        gestureState.lastCentroidY = centroid.y;
      }
      setGestureMode(touchCount);
    };

    const handleTouchCancel = (event) => {
      setGestureMode(event.touches.length);
    };

    setGestureMode(0);
    canvas.addEventListener("touchstart", handleTouchStart, {
      passive: false,
      capture: true,
    });
    canvas.addEventListener("touchmove", handleTouchMove, {
      passive: false,
      capture: true,
    });
    canvas.addEventListener("touchend", handleTouchEnd, {
      passive: false,
      capture: true,
    });
    canvas.addEventListener("touchcancel", handleTouchCancel, {
      passive: false,
      capture: true,
    });

    return () => {
      canvas.removeEventListener("touchstart", handleTouchStart, true);
      canvas.removeEventListener("touchmove", handleTouchMove, true);
      canvas.removeEventListener("touchend", handleTouchEnd, true);
      canvas.removeEventListener("touchcancel", handleTouchCancel, true);

      controls.enabled = true;
      controls.enableRotate = true;
      controls.enableZoom = true;
      controls.enablePan = true;
      controls.touches.ONE = THREE.TOUCH.ROTATE;
      controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
    };
  }, [controlsInstance, isMobile]);

  const cameraGestures = isMobile
    ? [
        { gesture: "Pinch", action: "Zoom" },
        { gesture: "2 Finger Drag", action: "Rotate" },
        { gesture: "3 Finger Drag", action: "Pan" },
      ]
    : [
        { gesture: "Scroll", action: "Zoom" },
        { gesture: "Left Drag", action: "Rotate" },
        { gesture: "Right Drag", action: "Pan" },
      ];

  return (
    <div className="app-container">
      <div className="left-panel">
        <div className="panel-shell">
          <div className="panel-shell-content">
            <p className="panel-kicker">GPGPU Particles</p>
            <h1 className="site-title">Dreamfield Particles</h1>
            <p className="hero-copy">
              {isMobile
                ? "250,000 particles are sampled from mesh surfaces and simulated on the GPU in real time."
                : "1,000,000 particles are sampled from mesh surfaces and simulated on the GPU in real time."}
            </p>
            <p className="hero-subcopy">
              Desktop runs 1,000,000 particles while mobile runs 250,000. A
              forward wind carries the cloud toward camera while lateral damping
              keeps each silhouette readable instead of blowing apart.
            </p>

            <div className="stats-row">
              <div className="stat-card">
                <span className="stat-value">
                  {isMobile ? "250,000" : "1,000,000"}
                </span>
                <span className="stat-label">Live Particles</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">GPGPU</span>
                <span className="stat-label">Simulation Pipeline</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">Real-Time</span>
                <span className="stat-label">
                  {isMobile ? "Touch Response" : "Cursor Response"}
                </span>
              </div>
            </div>

            <h2 className="section-title">Interaction</h2>
            {isMobile ? (
              <p>
                Your touch input is treated like a moving ray through the
                system. Dragging through the particles carves channels that
                reseal.
              </p>
            ) : (
              <p>
                Your pointer is treated like a moving ray through the system.
                Moving the cursor through the particles carves channels that
                reseal.
              </p>
            )}

            <div className="nav-hud" aria-label="Camera controls">
              <div className="nav-hud-header">
                <h3 className="nav-hud-title">Camera</h3>
                <button
                  type="button"
                  className="reset-camera-btn"
                  onClick={handleResetCamera}
                  aria-label="Reset camera view"
                  title="Reset camera view"
                >
                  Reset View
                </button>
              </div>

              <ul className="nav-hud-list">
                {cameraGestures.map(({ gesture, action }) => (
                  <li className="nav-hud-item" key={`${gesture}-${action}`}>
                    <span className="nav-gesture">{gesture}</span>
                    <span className="nav-action">{action}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="canvas-container">
        <Canvas
          camera={{ position: [0, 0, -5.5], fov: 60 }}
          className="fullscreen-canvas"
          dpr={isMobile ? [1, 1.25] : [1, 1.5]}
          gl={{
            antialias: false,
            powerPreference: "high-performance",
            stencil: false,
          }}
        >
          <ShiftCameraView isMobile={isMobile} />
          <GPGPUParticleMesh
            isMobile={isMobile}
            onInitialReady={handleInitialReady}
          />
          <OrbitControls
            ref={setControlsRef}
            enableDamping
            dampingFactor={CAMERA_DAMPING_FACTOR}
            enablePan={!isMobile}
            enableRotate
            enableZoom
            rotateSpeed={CAMERA_ROTATE_SPEED}
            zoomSpeed={CAMERA_ZOOM_SPEED}
            panSpeed={CAMERA_PAN_SPEED}
            touches={
              isMobile
                ? {
                    ONE: TOUCH_NONE,
                    TWO: THREE.TOUCH.DOLLY_ROTATE,
                  }
                : {
                    ONE: THREE.TOUCH.ROTATE,
                    TWO: THREE.TOUCH.DOLLY_PAN,
                  }
            }
            mouseButtons={{
              LEFT: THREE.MOUSE.ROTATE,
              MIDDLE: THREE.MOUSE.DOLLY,
              RIGHT: THREE.MOUSE.PAN,
            }}
          />
        </Canvas>

        {isInitialLoading && (
          <div className="loading-overlay" role="status" aria-live="polite">
            <div className="loading-overlay-content">
              <div className="loading-spinner" aria-hidden="true" />
              <p className="loading-text">Loading particles...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
