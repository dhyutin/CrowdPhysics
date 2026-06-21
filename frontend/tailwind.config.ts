import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Backgrounds — neutral charcoal (no blue tint)
        void:    "#0D1117",
        ground:  "#10151D",
        surface: "#161B22",
        raised:  "#1C2230",
        border:  "#21262D",

        // Accent / status
        teal:    "#4493F8",   // professional blue (primary CTA)
        emerald: "#3FB950",   // safe green
        amber:   "#D29922",   // warning amber
        crimson: "#F85149",   // danger red

        // Typography
        text1:   "#E6EDF3",
        text2:   "#8B949E",
        text3:   "#656D76",
      },
      fontFamily: {
        display: ["Space Grotesk", "system-ui", "sans-serif"],
        body:    ["Inter", "system-ui", "sans-serif"],
        mono:    ["JetBrains Mono", "Fira Code", "monospace"],
      },
      borderRadius: {
        lg: "10px",
        xl: "14px",
      },
      animation: {
        "fade-in":  "fade-in 0.2s ease-out",
        "spin-fast": "spin 0.6s linear infinite",
      },
      keyframes: {
        "fade-in": {
          "0%":   { opacity: "0", transform: "translateY(5px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
