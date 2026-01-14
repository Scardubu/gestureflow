'use client';

import { useRef, useState, useEffect } from 'react';

interface SwipeCanvasProps {
  onGestureComplete: (trajectory: number[][], timestamps: number[]) => void;
  disabled?: boolean;
}

export default function SwipeCanvas({ onGestureComplete, disabled }: SwipeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [trajectory, setTrajectory] = useState<number[][]>([]);
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [startTime, setStartTime] = useState<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw trajectory
    if (trajectory.length > 0) {
      ctx.strokeStyle = '#667eea';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(trajectory[0][0] * canvas.width, trajectory[0][1] * canvas.height);
      
      for (let i = 1; i < trajectory.length; i++) {
        ctx.lineTo(trajectory[i][0] * canvas.width, trajectory[i][1] * canvas.height);
      }
      
      ctx.stroke();

      // Draw start point
      ctx.fillStyle = '#4CAF50';
      ctx.beginPath();
      ctx.arc(trajectory[0][0] * canvas.width, trajectory[0][1] * canvas.height, 6, 0, Math.PI * 2);
      ctx.fill();

      // Draw end point
      if (trajectory.length > 1) {
        ctx.fillStyle = '#F44336';
        ctx.beginPath();
        const last = trajectory[trajectory.length - 1];
        ctx.arc(last[0] * canvas.width, last[1] * canvas.height, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [trajectory]);

  const handleStart = (x: number, y: number) => {
    if (disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const normalizedX = (x - rect.left) / rect.width;
    const normalizedY = (y - rect.top) / rect.height;

    setIsDrawing(true);
    setStartTime(Date.now());
    setTrajectory([[normalizedX, normalizedY]]);
    setTimestamps([0]);
  };

  const handleMove = (x: number, y: number) => {
    if (!isDrawing || disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const normalizedX = (x - rect.left) / rect.width;
    const normalizedY = (y - rect.top) / rect.height;

    const currentTime = (Date.now() - startTime) / 1000;

    setTrajectory(prev => [...prev, [normalizedX, normalizedY]]);
    setTimestamps(prev => [...prev, currentTime]);
  };

  const handleEnd = () => {
    if (!isDrawing || disabled) return;

    setIsDrawing(false);

    if (trajectory.length >= 2) {
      onGestureComplete(trajectory, timestamps);
    }
  };

  const handleClear = () => {
    setTrajectory([]);
    setTimestamps([]);
    setIsDrawing(false);
  };

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        onMouseDown={(e) => handleStart(e.clientX, e.clientY)}
        onMouseMove={(e) => handleMove(e.clientX, e.clientY)}
        onMouseUp={handleEnd}
        onMouseLeave={handleEnd}
        onTouchStart={(e) => {
          e.preventDefault();
          const touch = e.touches[0];
          handleStart(touch.clientX, touch.clientY);
        }}
        onTouchMove={(e) => {
          e.preventDefault();
          const touch = e.touches[0];
          handleMove(touch.clientX, touch.clientY);
        }}
        onTouchEnd={(e) => {
          e.preventDefault();
          handleEnd();
        }}
      />
      {trajectory.length > 0 && (
        <button
          onClick={handleClear}
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            padding: '8px 16px',
            background: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Clear
        </button>
      )}
    </div>
  );
}
