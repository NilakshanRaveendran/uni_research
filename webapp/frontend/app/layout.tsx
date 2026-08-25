import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bilingual Voice Dubbing",
  description:
    "Bilingual voice transcription and synthesis preserving speaker identity, using NLP and deep learning.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
