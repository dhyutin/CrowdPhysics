import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CrowdPhysics — AI Crowd Safety Platform",
  description:
    "Plan safe. Monitor live. Never react. Crowd fluid dynamics + AI safety platform.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="h-screen overflow-hidden bg-void text-text1">{children}</body>
    </html>
  );
}
