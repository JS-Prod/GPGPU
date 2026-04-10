import * as THREE from "three";
import { MeshSurfaceSampler } from "three/examples/jsm/Addons.js";

export default class GPGPUUtils {
  constructor(mesh, size, precomputedData = null) {
    this.size = size;
    this.precomputedData = precomputedData;

    this.number = this.size * this.size;
    this.ownsPositionTexture = false;
    this.ownsVelocityTexture = false;

    if (precomputedData) {
      this.setupDataFromPrecomputed(precomputedData);
    } else {
      this.mesh = mesh;
      this.sampler = new MeshSurfaceSampler(this.mesh).build();
      this._position = new THREE.Vector3();
      this.setupDataFromMesh();
    }

    this.setupVelocitiesData();
  }

  static getSharedUVs(size) {
    if (!this._sharedUVsBySize) {
      this._sharedUVsBySize = new Map();
    }

    if (!this._sharedUVsBySize.has(size)) {
      const number = size * size;
      const uvs = new Float32Array(2 * number);

      for (let index = 0; index < number; index++) {
        const i = Math.floor(index / size);
        const j = index % size;
        uvs[2 * index] = (j + 0.5) / size;
        uvs[2 * index + 1] = (i + 0.5) / size;
      }

      this._sharedUVsBySize.set(size, uvs);
    }

    return this._sharedUVsBySize.get(size);
  }

  static getSharedZeroPositions(size) {
    if (!this._sharedZeroPositionsBySize) {
      this._sharedZeroPositionsBySize = new Map();
    }

    if (!this._sharedZeroPositionsBySize.has(size)) {
      this._sharedZeroPositionsBySize.set(size, new Float32Array(3 * size * size));
    }

    return this._sharedZeroPositionsBySize.get(size);
  }

  static getSharedZeroVelocityData(size) {
    if (!this._sharedZeroVelocityBySize) {
      this._sharedZeroVelocityBySize = new Map();
    }

    if (!this._sharedZeroVelocityBySize.has(size)) {
      const data = new Float32Array(4 * size * size);
      for (let i = 3; i < data.length; i += 4) {
        data[i] = 0.04;
      }
      this._sharedZeroVelocityBySize.set(size, data);
    }

    return this._sharedZeroVelocityBySize.get(size);
  }

  static createDataTexture(data, size) {
    const texture = new THREE.DataTexture(
      data,
      size,
      size,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;
    return texture;
  }

  static getSharedZeroVelocityTexture(size) {
    if (!this._sharedZeroVelocityTextureBySize) {
      this._sharedZeroVelocityTextureBySize = new Map();
    }

    if (!this._sharedZeroVelocityTextureBySize.has(size)) {
      const zeroVelocityTexture = this.createDataTexture(
        this.getSharedZeroVelocityData(size),
        size
      );
      this._sharedZeroVelocityTextureBySize.set(size, zeroVelocityTexture);
    }

    return this._sharedZeroVelocityTextureBySize.get(size);
  }

  static generateParticleData(mesh, size) {
    const number = size * size;
    const sampler = new MeshSurfaceSampler(mesh).build();
    const sampledPosition = new THREE.Vector3();

    const positionData = new Float32Array(4 * number);
    const uvs = this.getSharedUVs(size);

    for (let index = 0; index < number; index++) {
      let tries = 0;
      do {
        sampler.sample(sampledPosition);
        tries++;
      } while (sampledPosition.lengthSq() < 1e-6 && tries < 5);

      positionData[4 * index] = sampledPosition.x;
      positionData[4 * index + 1] = sampledPosition.y;
      positionData[4 * index + 2] = sampledPosition.z;
      positionData[4 * index + 3] = 1.0;
    }

    return {
      positionData,
      positionTexture: this.createDataTexture(positionData, size),
      uvs,
    };
  }

  setupDataFromPrecomputed(precomputedData) {
    if (precomputedData.positionTexture) {
      this.positionTexture = precomputedData.positionTexture;
      this.ownsPositionTexture = false;
    } else {
      this.positionTexture = GPGPUUtils.createDataTexture(
        precomputedData.positionData,
        this.size
      );
      this.ownsPositionTexture = true;
    }

    this.positions =
      precomputedData.positions ?? GPGPUUtils.getSharedZeroPositions(this.size);
    this.uvs = precomputedData.uvs;
  }

  setupDataFromMesh() {
    const data = new Float32Array(4 * this.number);
    const positions = new Float32Array(3 * this.number);
    const uvs = GPGPUUtils.getSharedUVs(this.size);

    for (let index = 0; index < this.number; index++) {
      let tries = 0;
      do {
        this.sampler.sample(this._position);
        tries++;
      } while (this._position.lengthSq() < 1e-6 && tries < 5);

      // Position
      data[4 * index] = this._position.x;
      data[4 * index + 1] = this._position.y;
      data[4 * index + 2] = this._position.z;
      data[4 * index + 3] = 1.0;

      positions[3 * index] = this._position.x;
      positions[3 * index + 1] = this._position.y;
      positions[3 * index + 2] = this._position.z;
    }

    this.positions = positions;
    this.positionTexture = GPGPUUtils.createDataTexture(data, this.size);
    this.ownsPositionTexture = true;
    this.uvs = uvs;
  }

  setupVelocitiesData() {
    this.velocityTexture = GPGPUUtils.getSharedZeroVelocityTexture(this.size);
    this.ownsVelocityTexture = false;
  }

  getPositions() {
    return this.positions;
  }

  getUVs() {
    return this.uvs;
  }

  getPositionTexture() {
    return this.positionTexture;
  }

  getVelocityTexture() {
    return this.velocityTexture;
  }

  dispose() {
    if (this.ownsPositionTexture) {
      this.positionTexture?.dispose?.();
    }
    if (this.ownsVelocityTexture) {
      this.velocityTexture?.dispose?.();
    }

    this.mesh = null;
    this.sampler = null;
    this._position = null;
    this.positions = null;
    this.uvs = null;
    this.positionTexture = null;
    this.velocityTexture = null;
    this.precomputedData = null;
    this.ownsPositionTexture = null;
    this.ownsVelocityTexture = null;
  }
}
