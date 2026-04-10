import React, { useRef, useEffect } from "react";

const PARTICLE_RADIUS = 4;
const PARTICLE_SPEED = 1.5;
const PARTICLE_LIFETIME = 80; // frames

function randomRange(min, max) {
  return Math.random() * (max - min) + min;
}

export default function LeftPanelParticles() {
  const canvasRef = useRef(null);
  const animationFrameId = useRef(null);
  const particles = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Resize canvas to fill parent container
    function resize() {
      const parent = canvas.parentElement;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    }
    resize();

    // Particle constructor
    function createParticle() {
      return {
        x: canvas.width - PARTICLE_RADIUS, // start at right edge
        y: randomRange(PARTICLE_RADIUS, canvas.height - PARTICLE_RADIUS),
        radius: PARTICLE_RADIUS,
        speed: PARTICLE_SPEED * randomRange(0.7, 1.3),
        lifetime: PARTICLE_LIFETIME,
        alpha: 1,
      };
    }

    // Add new particle every few frames
    let frameCount = 0;

    function animate() {
      frameCount++;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Emit a new particle every 6 frames (~10 per second at 60fps)
      if (frameCount % 6 === 0) {
        particles.current.push(createParticle());
      }

      // Update and draw particles
      particles.current = particles.current.filter((p) => p.lifetime > 0);

      particles.current.forEach((p) => {
        p.x -= p.speed;
        p.lifetime--;
        p.alpha = p.lifetime / PARTICLE_LIFETIME;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${p.alpha.toFixed(2)})`; // white, fading out
        ctx.fill();
      });

      animationFrameId.current = requestAnimationFrame(animate);
    }

    animate();

    // Handle window resize to keep canvas sized properly
    window.addEventListener("resize", resize);

    return () => {
      cancelAnimationFrame(animationFrameId.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        pointerEvents: "none",
        userSelect: "none",
        width: "100%",
        height: "100%",
        display: "block",
      }}
    />
  );
}
