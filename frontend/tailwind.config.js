/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#1a1d29",
          hover: "#252836",
          dark: "#0d0f15",
        },
        background: "#0f1117",
        border: "#2d3148",
        profit: "#22c55e",
        loss: "#ef4444",
      },
    },
  },
  plugins: [],
};
