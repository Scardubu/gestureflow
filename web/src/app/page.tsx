'use client';

import SwipeCanvas from '@/components/SwipeCanvas';
import PredictionDisplay from '@/components/PredictionDisplay';
import LanguageSelector from '@/components/LanguageSelector';
import { useState } from 'react';

export default function Home() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [language, setLanguage] = useState('en');
  const [isLoading, setIsLoading] = useState(false);

  const handleGestureComplete = async (trajectory: number[][], timestamps: number[]) => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          trajectory,
          timestamps,
          top_k: 5,
          language,
        }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data.predictions || []);
    } catch (error) {
      console.error('Error:', error);
      setPredictions([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>GestureFlow</h1>
        <p>Swipe to type</p>
      </header>

      <main>
        <LanguageSelector 
          selected={language} 
          onChange={setLanguage}
        />
        
        <SwipeCanvas 
          onGestureComplete={handleGestureComplete}
          disabled={isLoading}
        />
        
        <PredictionDisplay 
          predictions={predictions}
          isLoading={isLoading}
        />
      </main>

      <footer>
        <p>Powered by GestureFlow ML</p>
      </footer>
    </div>
  );
}
