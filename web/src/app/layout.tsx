import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'GestureFlow',
  description: 'Gesture-based text input system',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
