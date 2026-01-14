'use client';

interface Prediction {
  word: string;
  confidence: number;
}

interface PredictionDisplayProps {
  predictions: Prediction[];
  isLoading: boolean;
}

export default function PredictionDisplay({ predictions, isLoading }: PredictionDisplayProps) {
  if (isLoading) {
    return (
      <div className="predictions">
        <h3>Predictions</h3>
        <div className="loading">Analyzing gesture</div>
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div className="predictions">
        <h3>Predictions</h3>
        <p style={{ textAlign: 'center', color: '#999', padding: '2rem' }}>
          Draw a gesture to see predictions
        </p>
      </div>
    );
  }

  return (
    <div className="predictions">
      <h3>Predictions</h3>
      <div className="prediction-list">
        {predictions.map((pred, index) => (
          <div key={index} className="prediction-item">
            <span className="prediction-word">
              {index + 1}. {pred.word}
            </span>
            <span className="prediction-confidence">
              {(pred.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
