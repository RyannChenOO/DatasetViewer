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
  metadataBase: new URL('https://locus-campus-spatial-viewer.cream-owlet-2009.chatgpt.site'),
  title: 'LOCUS — Campus Spatial Episode Viewer',
  description: 'An interactive viewer for embodied spatial intelligence research data from Lost on Campus.',
  openGraph: {
    title: 'LOCUS — Campus Spatial Episode Viewer',
    description: 'Explore, inspect, and annotate embodied spatial intelligence research data.',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'LOCUS campus spatial episode viewer' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'LOCUS — Campus Spatial Episode Viewer',
    description: 'Explore, inspect, and annotate embodied spatial intelligence research data.',
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
