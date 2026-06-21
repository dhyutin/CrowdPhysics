import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        void:    "#060a12",
        ground:  "#0a1020",
        surface: "#0f1a2e",
        raised:  "#132030",
        border:  "#1a2840",
        amber:   "#f59e0b",
        crimson: "#dc2626",
        teal:    "#0ea5e9",
        emerald: "#10b981",
        text1:   "#e2e8f4",
        text2:   "#8a9ab8",
        text3:   "#4a5878",
      },
      fontFamily: {
        display: ["Space Grotesk", "system-ui", "sans-serif"],
        body:    ["Inter", "system-ui", "sans-serif"],
        mono:    ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
