varying vec2 vUv;

uniform sampler2D uVelocityTexture;

uniform vec3 uColor;
uniform float uOpacity;



void main() {
    vec2 centered = gl_PointCoord - vec2(0.5);
    if (dot(centered, centered) > 0.25) {
        discard;
    }

    float velocityAlpha = texture2D( uVelocityTexture, vUv ).w;

    gl_FragColor = vec4(uColor, velocityAlpha * uOpacity);
}
