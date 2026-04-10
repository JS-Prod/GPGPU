import * as THREE from "three";
import GPGPUUtils from "./GPGPUUtils.js";
import { GPUComputationRenderer } from "three/examples/jsm/misc/GPUComputationRenderer.js";
import simFragmentPositionShader from "./shaders/simFragment.glsl?raw";
import simFragmentVelocityShader from "./shaders/simFragmentVelocity.glsl?raw";
import fragmentShader from "./shaders/fragment.glsl?raw";
import vertexShader from "./shaders/vertex.glsl?raw";

export default class GPGPU {
  constructor({
    size,
    camera,
    renderer,
    mouse,
    scene,
    model,
    color,
    sizes,
    isMobile,
    precomputedData,
  }) {
    this.camera = camera;
    this.renderer = renderer;
    this.mouse = mouse;
    this.scene = scene;
    this.sizes = sizes;
    this.size = size;
    this.model = model;
    this.color = color;
    this.isMobile = Boolean(isMobile);
    this.velocityScale = 60.0;
    this.precomputedData = precomputedData ?? null;
    this.disposed = false;
    this.computeTextureType = THREE.FloatType;
    this.currentOpacity = 1.0;
    this.currentVisible = true;
    this.currentWidth = this.sizes.width;
    this.currentHeight = this.sizes.height;

    this.lastFrameTime = performance.now();

    this.init();
  }

  init() {
    this.utils = new GPGPUUtils(this.model, this.size, this.precomputedData);
    this.initGPGPU();
    this.createParticles();
  }

  initGPGPU() {
    this.gpgpuCompute = new GPUComputationRenderer(
      this.size,
      this.size,
      this.renderer
    );

    const canUseHalfFloatCompute =
      Boolean(this.renderer?.capabilities?.isWebGL2) &&
      Boolean(this.renderer?.extensions?.has?.("EXT_color_buffer_float"));
    if (canUseHalfFloatCompute) {
      this.gpgpuCompute.setDataType(THREE.HalfFloatType);
      this.computeTextureType = THREE.HalfFloatType;
    } else {
      this.computeTextureType = THREE.FloatType;
    }

    const positionTexture = this.utils.getPositionTexture();
    const velocityTexture = this.utils.getVelocityTexture();

    this.positionVariable = this.gpgpuCompute.addVariable(
      "uCurrentPosition",
      simFragmentPositionShader,
      positionTexture
    );

    this.velocityVariable = this.gpgpuCompute.addVariable(
      "uCurrentVelocity",
      simFragmentVelocityShader,
      velocityTexture
    );

    this.gpgpuCompute.setVariableDependencies(this.positionVariable, [
      this.positionVariable,
      this.velocityVariable,
    ]);

    this.gpgpuCompute.setVariableDependencies(this.velocityVariable, [
      this.positionVariable,
      this.velocityVariable,
    ]);

    // uniforms
    this.uniforms = {
      positionUniforms: this.positionVariable.material.uniforms,
      velocityUniforms: this.velocityVariable.material.uniforms,
    };

    // Properly define deltaTime and velocityScale
    this.uniforms.positionUniforms.uDeltaTime = { value: 0 };
    this.uniforms.positionUniforms.uVelocityScale = { value: 60.0 };

    this.uniforms.velocityUniforms.uDeltaTime = { value: 0 };
    this.uniforms.velocityUniforms.uVelocityScale = { value: 60.0 };
    this.uniforms.velocityUniforms.uDeltaVelocityScale = { value: 0 };
    this.uniforms.velocityUniforms.uVelocityDamping = { value: 1.0 };

    this.uniforms.velocityUniforms.uMouseRayOrigin = {
      value: this.mouse.rayOrigin,
    };
    this.uniforms.velocityUniforms.uMouseRayDirection = {
      value: this.mouse.rayDirection,
    };
    this.uniforms.velocityUniforms.uPrevMouseRayOrigin = {
      value: this.mouse.previousRayOrigin,
    };
    this.uniforms.velocityUniforms.uPrevMouseRayDirection = {
      value: this.mouse.previousRayDirection,
    };
    this.uniforms.velocityUniforms.uMouseSpeed = { value: 0 };
    this.uniforms.velocityUniforms.uPositionTexture = {
      value: positionTexture,
    };
    this.uniforms.velocityUniforms.uVelocityTexture = {
      value: velocityTexture,
    };
    this.uniforms.velocityUniforms.uOriginalPosition = {
      value: positionTexture,
    };
    this.uniforms.velocityUniforms.uTime = { value: 0 };
    this.uniforms.velocityUniforms.uWind = {
      value: new THREE.Vector3(0.0, 0.0, -1.0),
    };

    const error = this.gpgpuCompute.init();
    if (error) {
      throw new Error(`GPGPU init failed: ${error}`);
    }
  }

  createParticles() {
    const geometry = new THREE.BufferGeometry();

    const positions = this.utils.getPositions();
    const uvs = this.utils.getUVs();

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uPositionTexture: {
          value: this.gpgpuCompute.getCurrentRenderTarget(this.positionVariable)
            .texture,
        },
        uResolution: {
          value: new THREE.Vector2(this.sizes.width, this.sizes.height),
        },
        uColor: { value: this.color },
        uOpacity: { value: 1.0 },
        uParticleSize: { value: 3 },
      },
      vertexShader: vertexShader,
      fragmentShader: fragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.mesh = new THREE.Points(geometry, this.material);
    this.scene.add(this.mesh);
  }

  setSize(width, height) {
    if (!this.material) return;
    if (this.currentWidth === width && this.currentHeight === height) return;
    this.currentWidth = width;
    this.currentHeight = height;
    this.material.uniforms.uResolution.value.set(width, height);
  }

  setOpacity(opacity) {
    if (!this.material) return;
    const safeOpacity = Math.max(0, Math.min(opacity, 1));
    if (this.currentOpacity === safeOpacity) return;
    this.currentOpacity = safeOpacity;
    this.material.uniforms.uOpacity.value = safeOpacity;
  }

  setVisible(visible) {
    if (!this.mesh) return;
    const safeVisible = Boolean(visible);
    if (this.currentVisible === safeVisible) return;
    this.currentVisible = safeVisible;
    this.mesh.visible = safeVisible;
  }

  resetTiming() {
    this.lastFrameTime = performance.now();
  }

  warmup(steps = 1) {
    if (this.disposed || !this.gpgpuCompute || !this.uniforms) return;

    const safeSteps = Math.max(1, Math.min(Math.floor(steps), 4));
    this.uniforms.positionUniforms.uDeltaTime.value = 0;
    this.uniforms.positionUniforms.uVelocityScale.value = this.velocityScale;
    this.uniforms.velocityUniforms.uDeltaTime.value = 0;
    this.uniforms.velocityUniforms.uVelocityScale.value = this.velocityScale;
    this.uniforms.velocityUniforms.uMouseSpeed.value = 0;

    for (let i = 0; i < safeSteps; i++) {
      this.gpgpuCompute.compute();
    }

    this.material.uniforms.uPositionTexture.value =
      this.gpgpuCompute.getCurrentRenderTarget(this.positionVariable).texture;
    this.resetTiming();
  }

  reseed({ precomputedData, color }) {
    if (this.disposed || !this.gpgpuCompute || !this.uniforms) return;
    if (!precomputedData?.positionTexture) return;

    const positionTexture = precomputedData.positionTexture;
    const velocityTexture = GPGPUUtils.getSharedZeroVelocityTexture(this.size);

    this.positionVariable.initialValueTexture = positionTexture;
    this.velocityVariable.initialValueTexture = velocityTexture;

    const positionTargets = this.positionVariable.renderTargets;
    const velocityTargets = this.velocityVariable.renderTargets;

    this.gpgpuCompute.renderTexture(positionTexture, positionTargets[0]);
    this.gpgpuCompute.renderTexture(positionTexture, positionTargets[1]);
    this.gpgpuCompute.renderTexture(velocityTexture, velocityTargets[0]);
    this.gpgpuCompute.renderTexture(velocityTexture, velocityTargets[1]);
    this.gpgpuCompute.currentTextureIndex = 0;

    this.precomputedData = precomputedData;
    this.color = color;
    this.material.uniforms.uColor.value = color;
    this.material.uniforms.uPositionTexture.value = positionTargets[0].texture;

    this.uniforms.velocityUniforms.uPositionTexture.value = positionTargets[0].texture;
    this.uniforms.velocityUniforms.uVelocityTexture.value = velocityTargets[0].texture;
    this.uniforms.velocityUniforms.uOriginalPosition.value = positionTexture;
    this.uniforms.velocityUniforms.uMouseSpeed.value = 0;
    this.uniforms.velocityUniforms.uMouseRayOrigin.value = this.mouse.rayOrigin;
    this.uniforms.velocityUniforms.uMouseRayDirection.value =
      this.mouse.rayDirection;
    this.uniforms.velocityUniforms.uPrevMouseRayOrigin.value =
      this.mouse.previousRayOrigin;
    this.uniforms.velocityUniforms.uPrevMouseRayDirection.value =
      this.mouse.previousRayDirection;

    this.resetTiming();
  }

  compute(time, nowMs = performance.now()) {
    if (this.disposed) return;

    let delta = (nowMs - this.lastFrameTime) / 1000;
    this.lastFrameTime = nowMs;

    delta = Math.max(0.001, Math.min(delta, 0.1));

    this.uniforms.positionUniforms.uDeltaTime.value = delta;
    this.uniforms.positionUniforms.uVelocityScale.value = this.velocityScale;

    this.uniforms.velocityUniforms.uDeltaTime.value = delta;
    this.uniforms.velocityUniforms.uVelocityScale.value = this.velocityScale;
    this.uniforms.velocityUniforms.uDeltaVelocityScale.value =
      delta * this.velocityScale;
    this.uniforms.velocityUniforms.uVelocityDamping.value = Math.pow(
      0.98,
      delta * 60.0
    );
    this.uniforms.velocityUniforms.uMouseSpeed.value = this.mouse.speed ?? 0;

    const currentPositionTarget =
      this.gpgpuCompute.getCurrentRenderTarget(this.positionVariable);
    const currentVelocityTarget =
      this.gpgpuCompute.getCurrentRenderTarget(this.velocityVariable);

    this.uniforms.velocityUniforms.uTime.value = time;
    this.uniforms.velocityUniforms.uPositionTexture.value = currentPositionTarget.texture;
    this.uniforms.velocityUniforms.uVelocityTexture.value = currentVelocityTarget.texture;

    this.gpgpuCompute.compute();

    this.material.uniforms.uPositionTexture.value =
      this.gpgpuCompute.getCurrentRenderTarget(this.positionVariable).texture;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;

    if (this.mesh?.parent) {
      this.mesh.parent.remove(this.mesh);
    }

    if (this.mesh?.geometry) {
      this.mesh.geometry.dispose();
    }

    if (this.material) {
      this.material.uniforms.uPositionTexture.value = null;
      this.material.dispose();
    }

    if (this.gpgpuCompute?.variables) {
      for (const variable of this.gpgpuCompute.variables) {
        // These input textures can be shared across model instances.
        // Prevent GPUComputationRenderer.dispose() from disposing them
        // so they stay resident and reusable between cycles.
        variable.initialValueTexture = null;
        variable.material?.dispose?.();
      }
    }

    this.gpgpuCompute?.dispose?.();
    this.utils?.dispose?.();

    this.mesh = null;
    this.material = null;
    this.gpgpuCompute = null;
    this.positionVariable = null;
    this.velocityVariable = null;
    this.uniforms = null;
    this.utils = null;
  }
}
