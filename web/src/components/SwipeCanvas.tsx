'use client';

import React, { useRef, useState, useEffect } from 'react';

interface Point {
  x: number;
  y: number;
  timestamp: number;
}

interface SwipeCanvasProps {
  onGestureComplete: (points: Point[]) => void;
  disabled?: boolean;
}

export default function SwipeCanvas({ onGestureComplete, disabled = false }: SwipeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [points, setPoints] = useState<Point[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw keyboard layout
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    
    // Draw simple QWERTY layout
    const keys = [
      ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
      ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
      ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ];
    
    const keyWidth = 35;
    const keyHeight = 45;
    const startY = 50;
    
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    keys.forEach((row, rowIndex) => {
      const startX = (canvas.width - row.length * keyWidth) / 2;
      row.forEach((key, keyIndex) => {
        const x = startX + keyIndex * keyWidth;
        const y = startY + rowIndex * keyHeight;
        
        ctx.strokeRect(x, y, keyWidth - 2, keyHeight - 2);
        ctx.fillStyle = '#000';
        ctx.fillText(key, x + keyWidth / 2, y + keyHeight / 2);
      });
    });

    // Draw gesture path
    if (points.length > 0) {
      ctx.strokeStyle = '#2196F3';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      
      ctx.stroke();
      
      // Draw points
      ctx.fillStyle = '#2196F3';
      points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }, [points]);

  const getCoordinates = (event: React.MouseEvent | React.TouchEvent): Point | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    let clientX: number, clientY: number;

    if ('touches' in event) {
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      clientX = event.clientX;
      clientY = event.clientY;
    }

    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
      timestamp: Date.now()
    };
  };

  const handleStart = (event: React.MouseEvent | React.TouchEvent) => {
    if (disabled) return;
    
    event.preventDefault();
    setIsDrawing(true);
    
    const point = getCoordinates(event);
    if (point) {
      setPoints([point]);
    }
  };

  const handleMove = (event: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || disabled) return;
    
    event.preventDefault();
    const point = getCoordinates(event);
    
    if (point) {
      setPoints(prev => [...prev, point]);
    }
  };

  const handleEnd = () => {
    if (!isDrawing || disabled) return;
    
    setIsDrawing(false);
    
    if (points.length > 2) {
      onGestureComplete(points);
    }
    
    setPoints([]);
  };

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={400}
        height={250}
        className="border border-gray-300 rounded-lg cursor-crosshair touch-none"
        onMouseDown={handleStart}
        onMouseMove={handleMove}
        onMouseUp={handleEnd}
        onMouseLeave={handleEnd}
        onTouchStart={handleStart}
        onTouchMove={handleMove}
        onTouchEnd={handleEnd}
      />
      <p className="mt-2 text-sm text-gray-600">
        Draw a gesture by swiping across the keyboard
      </p>
    </div>
  );
}
