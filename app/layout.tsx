import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://locus-campus-spatial-viewer.ryannchenoo.chatgpt.site'),
  title: 'Habitat Navigation Learner',
  description: 'Explore Habitat trajectories and predict the next navigation action with a trained MLP.',
  openGraph: {
    title: 'Habitat Navigation Learner',
    description: 'A trajectory-first annotation and neural prediction workbench for spatial intelligence research.',
    images: [{ url: '/og.png', width: 1730, height: 909, alt: 'Habitat navigation learner interface' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Habitat Navigation Learner',
    description: 'Next-action prediction plus frame-level and segment-level trajectory annotation.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
