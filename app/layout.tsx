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
  title: 'Habitat Trajectory Annotator',
  description: 'Annotate long Habitat trajectories at both frame and segment level.',
  openGraph: {
    title: 'Habitat Trajectory Annotator',
    description: 'A trajectory-first annotation workbench for spatial intelligence research.',
    images: [{ url: '/og.png', width: 1730, height: 909, alt: 'Habitat trajectory annotator interface' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Habitat Trajectory Annotator',
    description: 'Frame-level and segment-level labels for long navigation trajectories.',
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
