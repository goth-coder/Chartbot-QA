// Empty PostCSS config, on purpose. Vite/PostCSS otherwise searches UP the directory
// tree for a config and can pick up an unrelated one from a parent folder (e.g. a
// Tailwind config in D:\Projects\), which then fails the build with
// "Cannot find module 'tailwindcss'". This project uses plain CSS (see src/*.css) and
// needs no PostCSS plugins — declaring an empty config here stops the upward search and
// keeps the build hermetic regardless of what sits above the repo.
export default {
  plugins: {},
}
