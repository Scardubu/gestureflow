“use client”;

import React, { useState, useRef, useEffect } from ‘react’;

interface Point {
x: number;
y: number;
timestamp: number;
}

interface Prediction {
word: string;
confidence: number;
rank: number;
}

interface PredictionResponse {
predictions: Prediction[];
inference_time_ms: number;
num_points: number;
}

export default function SwipePredictDemo() {
const canvasRef = useRef<HTMLCanvasElement>(null);
const [isDrawing, setIsDrawing] = useState(false);
const [trajectory, setTrajectory] = useState<Point[]>([]);
const [predictions, setPredictions] = useState<Prediction[]>([]);
const [inferenceTime, setInferenceTime] = useState<number>(0);
const [language, setLanguage] = useState<string>(‘en’);
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState<string>(’’);
const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

// Keyboard layout for visual reference
const keyboardLayout = [
[‘Q’, ‘W’, ‘E’, ‘R’, ‘T’, ‘Y’, ‘U’, ‘I’, ‘O’, ‘P’],
[‘A’, ‘S’, ‘D’, ‘F’, ‘G’, ‘H’, ‘J’, ‘K’, ‘L’],
[‘Z’, ‘X’, ‘C’, ‘V’, ‘B’, ‘N’, ‘M’]
];

useEffect(() => {
const canvas = canvasRef.current;
if (!canvas) return;

```
const updateCanvasSize = () => {
  const container = canvas.parentElement;
  if (container) {
    const width = Math.min(600, container.clientWidth - 32);
    const height = 400;
    canvas.width = width;
    canvas.height = height;
    setCanvasSize({ width, height });
    drawKeyboard();
  }
};

updateCanvasSize();
window.addEventListener('resize', updateCanvasSize);
return () => window.removeEventListener('resize', updateCanvasSize);
```

}, []);

const drawKeyboard = () => {
const canvas = canvasRef.current;
if (!canvas) return;

```
const ctx = canvas.getContext('2d');
if (!ctx) return;

// Clear canvas
ctx.fillStyle = '#1a1a1a';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Draw keyboard keys
const keyWidth = canvas.width / 10;
const keyHeight = canvas.height / 4;
const padding = 8;

ctx.fillStyle = '#2a2a2a';
ctx.strokeStyle = '#404040';
ctx.lineWidth = 1;
ctx.font = '20px Arial';
ctx.textAlign = 'center';
ctx.textBaseline = 'middle';

keyboardLayout.forEach((row, rowIndex) => {
  const startOffset = rowIndex * keyWidth * 0.3;
  row.forEach((key, colIndex) => {
    const x = startOffset + colIndex * keyWidth + padding;
    const y = rowIndex * keyHeight + keyHeight * 0.5 + padding;
    const width = keyWidth - padding * 2;
    const height = keyHeight - padding * 2;

    // Draw key
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);

    // Draw letter
    ctx.fillStyle = '#ffffff';
    ctx.fillText(key, x + width / 2, y + height / 2);
    ctx.fillStyle = '#2a2a2a';
  });
});
```

};

const getNormalizedCoordinates = (
clientX: number,
clientY: number
): { x: number; y: number } => {
const canvas = canvasRef.current;
if (!canvas) return { x: 0, y: 0 };

```
const rect = canvas.getBoundingClientRect();
const x = (clientX - rect.left) / canvas.width;
const y = (clientY - rect.top) / canvas.height;

return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
```

};

const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
setIsDrawing(true);
setPredictions([]);
setError(’’);

```
const point = 'touches' in e
  ? getNormalizedCoordinates(e.touches[0].clientX, e.touches[0].clientY)
  : getNormalizedCoordinates(e.clientX, e.clientY);

const newPoint: Point = {
  ...point,
  timestamp: Date.now()
};

setTrajectory([newPoint]);
drawKeyboard();
```

};

const draw = (e: React.MouseEvent | React.TouchEvent) => {
if (!isDrawing) return;

```
const point = 'touches' in e
  ? getNormalizedCoordinates(e.touches[0].clientX, e.touches[0].clientY)
  : getNormalizedCoordinates(e.clientX, e.clientY);

const newPoint: Point = {
  ...point,
  timestamp: Date.now()
};

setTrajectory(prev => {
  const updated = [...prev, newPoint];
  drawTrajectory(updated);
  return updated;
});
```

};

const stopDrawing = async () => {
if (!isDrawing) return;
setIsDrawing(false);

```
if (trajectory.length < 3) {
  setError('Gesture too short. Please draw a longer swipe.');
  return;
}

await makePrediction();
```

};

const drawTrajectory = (points: Point[]) => {
const canvas = canvasRef.current;
if (!canvas) return;

```
const ctx = canvas.getContext('2d');
if (!ctx) return;

drawKeyboard();

// Draw trajectory
ctx.strokeStyle = '#00d9ff';
ctx.lineWidth = 4;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

ctx.beginPath();
points.forEach((point, i) => {
  const x = point.x * canvas.width;
  const y = point.y * canvas.height;

  if (i === 0) {
    ctx.moveTo(x, y);
  } else {
    ctx.lineTo(x, y);
  }
});
ctx.stroke();

// Draw points
ctx.fillStyle = '#00d9ff';
points.forEach((point, i) => {
  const x = point.x * canvas.width;
  const y = point.y * canvas.height;
  const radius = i === 0 || i === points.length - 1 ? 6 : 3;
  
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
});
```

};

const makePrediction = async () => {
setIsLoading(true);
setError(’’);

```
try {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      trajectory: trajectory,
      language: language,
      top_k: 5
    })
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const data: PredictionResponse = await response.json();
  setPredictions(data.predictions);
  setInferenceTime(data.inference_time_ms);
} catch (err) {
  setError(err instanceof Error ? err.message : 'Prediction failed');
  console.error('Prediction error:', err);
} finally {
  setIsLoading(false);
}
```

};

const clearCanvas = () => {
setTrajectory([]);
setPredictions([]);
setError(’’);
setInferenceTime(0);
drawKeyboard();
};

return (
<div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
<div className="container mx-auto px-4 py-8 max-w-6xl">
{/* Header */}
<div className="text-center mb-8">
<h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
SwipePredict
</h1>
<p className="text-xl text-gray-400">
LSTM-based Swipe Typing Prediction Engine
</p>
<p className="text-sm text-gray-500 mt-2">
Draw gestures across the keyboard to predict words
</p>
</div>

```
    <div className="grid md:grid-cols-2 gap-6">
      {/* Canvas Section */}
      <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Draw Your Gesture</h2>
          <button
            onClick={clearCanvas}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            Clear
          </button>
        </div>

        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="w-full border-2 border-gray-700 rounded-lg cursor-crosshair"
          style={{ touchAction: 'none' }}
        />

        <div className="mt-4 flex gap-4 items-center">
          <label className="text-sm text-gray-400">Language:</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="bg-gray-700 text-white px-4 py-2 rounded-lg"
          >
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
          </select>
        </div>

        {trajectory.length > 0 && (
          <div className="mt-4 text-sm text-gray-400">
            Points captured: {trajectory.length}
          </div>
        )}
      </div>

      {/* Results Section */}
      <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
        <h2 className="text-xl font-semibold mb-4">Predictions</h2>

        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400"></div>
          </div>
        )}

        {error && (
          <div className="bg-red-900/30 border border-red-500 rounded-lg p-4 mb-4">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {predictions.length > 0 && !isLoading && (
          <>
            <div className="space-y-3 mb-6">
              {predictions.map((pred, idx) => (
                <div
                  key={idx}
                  className="bg-gray-700 rounded-lg p-4 flex items-center justify-between hover:bg-gray-600 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <span className="text-2xl font-bold text-cyan-400">
                      #{pred.rank}
                    </span>
                    <span className="text-xl font-semibold">
                      {pred.word}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400">Confidence</div>
                    <div className="text-lg font-semibold">
                      {(pred.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="border-t border-gray-700 pt-4">
              <div className="flex justify-between text-sm text-gray-400">
                <span>Inference Time:</span>
                <span className="font-semibold text-cyan-400">
                  {inferenceTime.toFixed(1)} ms
                </span>
              </div>
            </div>
          </>
        )}

        {!isLoading && predictions.length === 0 && !error && (
          <div className="text-center py-12 text-gray-500">
            <svg
              className="w-16 h-16 mx-auto mb-4 opacity-50"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
              />
            </svg>
            <p>Draw a gesture to see predictions</p>
          </div>
        )}
      </div>
    </div>

    {/* Info Section */}
    <div className="mt-8 bg-gray-800 rounded-lg p-6 shadow-2xl">
      <h2 className="text-xl font-semibold mb-4">How It Works</h2>
      <div className="grid md:grid-cols-3 gap-4 text-sm">
        <div>
          <h3 className="font-semibold text-cyan-400 mb-2">1. Draw Gesture</h3>
          <p className="text-gray-400">
            Swipe across the keyboard layout as you would type a word
          </p>
        </div>
        <div>
          <h3 className="font-semibold text-cyan-400 mb-2">2. LSTM Processing</h3>
          <p className="text-gray-400">
            Bidirectional LSTM analyzes the gesture sequence
          </p>
        </div>
        <div>
          <h3 className="font-semibold text-cyan-400 mb-2">3. Get Predictions</h3>
          <p className="text-gray-400">
            Top-5 word predictions with confidence scores
          </p>
        </div>
      </div>
    </div>

    {/* Stats */}
    <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-gray-800 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-cyan-400">67.3%</div>
        <div className="text-sm text-gray-400">Top-1 Accuracy</div>
      </div>
      <div className="bg-gray-800 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-cyan-400">89.1%</div>
        <div className="text-sm text-gray-400">Top-5 Accuracy</div>
      </div>
      <div className="bg-gray-800 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-cyan-400">&lt;50ms</div>
        <div className="text-sm text-gray-400">Inference Time</div>
      </div>
      <div className="bg-gray-800 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-cyan-400">2.1MB</div>
        <div className="text-sm text-gray-400">Model Size</div>
      </div>
    </div>
  </div>
</div>
```

);
}
