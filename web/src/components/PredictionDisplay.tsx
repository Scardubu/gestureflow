'use client';

import React from 'react';

interface Prediction {
  word: string;
  confidence: number;
}

interface PredictionDisplayProps {
  predictions: Prediction[];
  isLoading?: boolean;
}

export default function PredictionDisplay({ predictions, isLoading = false }: PredictionDisplayProps) {
  if (isLoading) {
    return (
      <div className="p-6 bg-white rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Predictions</h3>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div className="p-6 bg-white rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Predictions</h3>
        <p className="text-gray-500 text-center py-8">
          Draw a gesture to see predictions
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h3 className="text-lg font-semibold mb-4">Predictions</h3>
      <div className="space-y-3">
        {predictions.map((prediction, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <span className="text-lg font-medium">{prediction.word}</span>
            <div className="flex items-center gap-2">
              <div className="w-32 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${prediction.confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-sm text-gray-600 min-w-[3rem] text-right">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
