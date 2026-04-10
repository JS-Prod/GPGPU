
precision highp float;

uniform sampler2D uVelocityTexture;
uniform sampler2D uPositionTexture;
uniform sampler2D uOriginalPosition;
uniform vec3 uWind;
uniform vec3 uMouseRayOrigin;
uniform vec3 uMouseRayDirection;
uniform vec3 uPrevMouseRayOrigin;
uniform vec3 uPrevMouseRayDirection;
uniform float uMouseSpeed;
uniform float uTime;
uniform float uDeltaTime;
uniform float uVelocityScale;
uniform float uDeltaVelocityScale;
uniform float uVelocityDamping;

vec3 rayRepel(
    vec3 position,
    vec3 rayOrigin,
    vec3 rayDirection,
    float radius,
    float force
) {
    float rayLengthSq = dot(rayDirection, rayDirection);
    if (rayLengthSq <= 1e-10) return vec3(0.0);

    vec3 dir = rayDirection * inversesqrt(rayLengthSq);
    vec3 toParticle = position - rayOrigin;
    float rayT = max(dot(toParticle, dir), 0.0);
    vec3 closestOnRay = rayOrigin + dir * rayT;
    vec3 rayOffset = position - closestOnRay;

    // Stronger partial depth compensation so edge enlargement is clearly reduced
    // while still preserving a slight perspective variation.
    float depthScale = clamp(rayT / 4.2, 0.48, 1.55);
    float effectiveRadius = radius * mix(1.0, depthScale, 0.72);
    float effectiveRadiusSq = effectiveRadius * effectiveRadius;
    float rayDistanceSq = dot(rayOffset, rayOffset);

    if (rayDistanceSq >= effectiveRadiusSq) return vec3(0.0);

    float invRayDistance = inversesqrt(max(rayDistanceSq, 1e-10));
    float rayDistance = rayDistanceSq * invRayDistance;
    vec3 pushSeed = rayOffset + vec3(0.0001);
    vec3 pushDirection = pushSeed * inversesqrt(dot(pushSeed, pushSeed));
    float falloff = 1.0 - (rayDistance / effectiveRadius);
    // Smooth radial falloff gives a softer, more fluid displacement edge.
    falloff = falloff * falloff * (3.0 - 2.0 * falloff);
    return pushDirection * force * falloff;
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;

    vec3 velocity = texture2D(uVelocityTexture, uv).xyz;
    vec3 position = texture2D(uPositionTexture, uv).xyz;
    vec3 original = texture2D(uOriginalPosition, uv).xyz;

    vec3 wind = uWind;
    if (dot(wind, wind) > 0.0) {
        wind *= sin(uTime * 0.5 + uv.x * 10.0) * 0.5 + 0.5;
        velocity += wind * 0.01 * uDeltaVelocityScale;
    }

    vec3 windAxisSeed = uWind + vec3(0.00001);
    vec3 windAxis =
        windAxisSeed * inversesqrt(dot(windAxisSeed, windAxisSeed));
    vec3 toOriginal = original - position;
    float alongWind = dot(toOriginal, windAxis);
    vec3 returnToFlowLine = toOriginal - (windAxis * alongWind);
    vec3 lateralVelocity = velocity - windAxis * dot(velocity, windAxis);
    velocity += returnToFlowLine * 0.035 * uDeltaVelocityScale;
    velocity -= lateralVelocity * 0.22 * uDeltaVelocityScale;

    float baseMouseForce = 0.0085;
    float motionBoost = 0.0012 * min(uMouseSpeed, 1.0);
    float mouseForce = baseMouseForce + motionBoost;
    float rayRadius = 0.145;

    vec3 mouseDisplacement = rayRepel(
        position,
        uMouseRayOrigin,
        uMouseRayDirection,
        rayRadius,
        mouseForce
    );

    float motionActivation = 0.0;
    float prevRayLengthSq = dot(uPrevMouseRayDirection, uPrevMouseRayDirection);
    if (prevRayLengthSq > 0.0) {
        vec3 prevDirNorm =
            uPrevMouseRayDirection * inversesqrt(prevRayLengthSq);
        vec3 currDirSeed = uMouseRayDirection + vec3(0.0001);
        vec3 currDirNorm =
            currDirSeed * inversesqrt(max(dot(currDirSeed, currDirSeed), 1e-10));
        float dirDot = clamp(dot(prevDirNorm, currDirNorm), -1.0, 1.0);
        float speedFactor = min(uMouseSpeed, 1.0);
        float directionFast = 1.0 - smoothstep(0.9800666, 0.999992, dirDot);
        float fastMove = max(directionFast, speedFactor * 0.75);
        motionActivation = smoothstep(0.05, 0.85, fastMove);
        float bridgeCount = mix(4.0, 20.0, fastMove);
        float bridgeForce = mouseForce * mix(0.58, 0.76, fastMove);
        int bridgeSteps = int(floor(bridgeCount));
        float bridgeTScale = 1.0 / (bridgeCount + 1.0);
        vec3 bridgeOriginDelta = uMouseRayOrigin - uPrevMouseRayOrigin;
        vec3 bridgeDirDelta = currDirNorm - prevDirNorm;
        mouseDisplacement += rayRepel(
            position,
            uPrevMouseRayOrigin,
            prevDirNorm,
            rayRadius,
            bridgeForce
        );

        const int MAX_BRIDGE_SAMPLES = 20;
        for (int i = 1; i <= MAX_BRIDGE_SAMPLES; i++) {
            if (i > bridgeSteps) {
                break;
            }

            float t = float(i) * bridgeTScale;
            vec3 bridgeSample = rayRepel(
                position,
                uPrevMouseRayOrigin + bridgeOriginDelta * t,
                prevDirNorm + bridgeDirDelta * t + vec3(0.0001),
                rayRadius,
                bridgeForce
            );
            mouseDisplacement += bridgeSample;
        }
    }

    float displacementMagnitude = length(mouseDisplacement);
    if (displacementMagnitude > 0.0) {
        float speedFactor = min(uMouseSpeed, 1.0);
        vec3 flowAxisSeed =
            uMouseRayDirection + (uPrevMouseRayDirection * 0.8) + vec3(0.0001);
        vec3 flowAxis =
            flowAxisSeed * inversesqrt(dot(flowAxisSeed, flowAxisSeed));
        vec3 displacementDir = mouseDisplacement / (displacementMagnitude + 0.0001);
        vec3 swirlSeed = cross(flowAxis, displacementDir) + vec3(0.0001);
        vec3 swirlDir = swirlSeed * inversesqrt(dot(swirlSeed, swirlSeed));

        float response = smoothstep(0.0, 0.06, displacementMagnitude);
        float swirlGain = (0.02 + speedFactor * 0.18) * response * motionActivation;
        float streamGain = (0.015 + speedFactor * 0.12) * response * motionActivation;

        mouseDisplacement += swirlDir * displacementMagnitude * swirlGain;
        mouseDisplacement += flowAxis * displacementMagnitude * streamGain;
        displacementMagnitude = length(mouseDisplacement);
    }

    // Keep interaction reversible by removing wind-axis displacement drift.
    // This prevents particles from being permanently advanced/retarded.
    if (displacementMagnitude > 0.0) {
        mouseDisplacement -= windAxis * dot(mouseDisplacement, windAxis);
        displacementMagnitude = length(mouseDisplacement);
    }

    if (displacementMagnitude > 0.06) {
        mouseDisplacement *= 0.06 / displacementMagnitude;
    }

    velocity += mouseDisplacement * uDeltaVelocityScale;

    // Additional settle pass only when cursor influence is gone/weak.
    float idleCursor = 1.0 - smoothstep(0.03, 0.18, min(uMouseSpeed, 1.0));
    float settleFactor =
        idleCursor * (1.0 - smoothstep(0.0006, 0.008, displacementMagnitude));
    vec3 lateralVelocityPost = velocity - windAxis * dot(velocity, windAxis);
    velocity += returnToFlowLine * (0.028 * settleFactor) * uDeltaVelocityScale;
    velocity -= lateralVelocityPost * (0.11 * settleFactor) * uDeltaVelocityScale;

    float maxSpeed = 1.5;
    float speed = length(velocity);
    if (speed > maxSpeed) {
        velocity *= maxSpeed / speed;
        speed = maxSpeed;
    }

    velocity *= uVelocityDamping;
    speed *= uVelocityDamping;

    // Store render alpha in velocity texture .w so fragment shading
    // avoids recomputing speed length per covered pixel.
    float velocityAlpha = clamp(speed, 0.04, 0.8);
    gl_FragColor = vec4(velocity, velocityAlpha);
}
