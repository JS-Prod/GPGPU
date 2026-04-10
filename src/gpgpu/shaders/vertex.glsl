varying vec2 vUv;

uniform float uParticleSize;
uniform sampler2D uPositionTexture;

void main() {
    vUv = uv;

    vec3 newpos = position;

    vec4 color = texture2D( uPositionTexture, vUv );



    newpos.xyz = color.xyz;

    vec4 mvPosition = modelViewMatrix * vec4( newpos, 1.0 );

    gl_PointSize = ( uParticleSize / -mvPosition.z );

    gl_Position = projectionMatrix * mvPosition;
}
