import type { Metadata } from "next";
import { Urbanist } from "next/font/google";
import "./globals.css";

// next/font self-hosts the files at build time, so there is no runtime request to Google and no
// flash of unstyled text.
const urbanist = Urbanist({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-urbanist",
});

export const metadata: Metadata = {
  metadataBase: new URL("http://localhost:3000"),
  title: "Bilingual Voice Dubbing",
  description:
    "Bilingual voice transcription and synthesis preserving speaker identity, using NLP and deep learning.",
  openGraph: {
    title: "Bilingual Voice Dubbing",
    description: "English ↔ Spanish speech translation preserving speaker identity.",
    images: [{ url: "/og.png", width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Bilingual Voice Dubbing",
    description: "English ↔ Spanish speech translation preserving speaker identity.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={urbanist.variable}>
      <body>{children}</body>
    </html>
  );
}
