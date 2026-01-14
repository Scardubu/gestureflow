import type { Metadata } from ‘next’;
import { Inter } from ‘next/font/google’;
import ‘./globals.css’;

const inter = Inter({ subsets: [‘latin’] });

export const metadata: Metadata = {
title: ‘GestureFlow | LSTM Swipe Typing Prediction’,
description: ‘Production-ready LSTM sequence model for swipe typing prediction. Multi-language support with <50ms inference latency.’,
keywords: [‘machine learning’, ‘LSTM’, ‘NLP’, ‘swipe typing’, ‘gesture recognition’, ‘deep learning’],
authors: [{ name: ‘Oscar Ndugbu’, url: ‘https://www.scardubu.dev’ }],
openGraph: {
title: ‘GestureFlow - AI Swipe Typing Engine’,
description: ‘LSTM-based swipe typing prediction with 67% top-1 and 89% top-5 accuracy’,
url: ‘https://gestureflow.scardubu.dev’,
siteName: ‘GestureFlow’,
type: ‘website’,
},
twitter: {
card: ‘summary_large_image’,
title: ‘GestureFlow - AI Swipe Typing Engine’,
description: ‘LSTM sequence model achieving 89% top-5 accuracy for swipe prediction’,
creator: ‘@scardubu’,
},
};

export default function RootLayout({
children,
}: {
children: React.ReactNode;
}) {
return (
<html lang="en">
<body className={inter.className}>
{children}
<footer className="bg-gray-900 border-t border-gray-800 py-6 mt-12">
<div className="container mx-auto px-4 text-center text-gray-400 text-sm">
<p>
Built by{’ ‘}
<a
href="https://www.scardubu.dev"
target="_blank"
rel="noopener noreferrer"
className="text-cyan-400 hover:text-cyan-300 transition-colors"
>
Oscar Ndugbu
</a>
{’ | ’}
<a
href="https://github.com/scardubu/gestureflow"
target="_blank"
rel="noopener noreferrer"
className="text-cyan-400 hover:text-cyan-300 transition-colors"
>
View on GitHub
</a>
</p>
<p className="mt-2 text-xs text-gray-500">
Production ML Engineer • Building AI that works
</p>
</div>
</footer>
</body>
</html>
);
}
