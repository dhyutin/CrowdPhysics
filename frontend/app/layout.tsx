import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CrowdPhysics — AI Crowd Safety",
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
      <body className="min-h-screen bg-void text-text1">{children}</body>
    </html>
  );
}
