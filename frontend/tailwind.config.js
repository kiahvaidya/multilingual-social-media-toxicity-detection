/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        void: '#050608',
        obsidian: '#0a0c10',
        charcoal: '#12151c',
        graphite: '#1c202a',
        slate: {
          muted: '#282f3d',
          base: '#3e4758',
          light: '#65728a',
        },
        steel: '#8d98ad',
        silver: '#c8cfdb',
        bone: '#f2ede4',
        chalk: '#faf8f5',
        forensic: {
          safe: '#2dd4bf',       // Calm, cool turquoise/steel-cyan
          safeMuted: 'rgba(45, 212, 191, 0.15)',
          review: '#f59e0b',     // Amber ambiguity warning
          reviewMuted: 'rgba(245, 158, 11, 0.15)',
          risk: '#f43f5e',       // Controlled crimson / high-risk alert
          riskMuted: 'rgba(244, 63, 94, 0.15)',
          beam: '#38bdf8',       // Precision laser / vector beam
          beamMuted: 'rgba(56, 189, 248, 0.15)',
        }
      },
      fontFamily: {
        display: ['Syne', 'Space Grotesk', 'sans-serif'],
        sans: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      letterSpacing: {
        ultra: '0.25em',
        widest: '0.15em',
      },
      boxShadow: {
        'forensic-risk': '0 0 35px -8px rgba(244, 63, 94, 0.4)',
        'forensic-safe': '0 0 35px -8px rgba(45, 212, 191, 0.35)',
        'forensic-review': '0 0 35px -8px rgba(245, 158, 11, 0.35)',
        'forensic-beam': '0 0 30px -5px rgba(56, 189, 248, 0.35)',
        'subtle-depth': '0 20px 40px -15px rgba(0, 0, 0, 0.7)',
      },
      animation: {
        'reticle-spin': 'spin 12s linear infinite',
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
