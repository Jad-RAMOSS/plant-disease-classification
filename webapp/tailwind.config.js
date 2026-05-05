/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'ecu-red': '#C41230',
        'ecu-red-dark': '#9E0E27',
        'ecu-red-light': '#F9E8EB',
        'ecu-gray': '#5A5A5A',
        'ecu-gray-light': '#9A9A9A',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
