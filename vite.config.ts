import { defineConfig } from 'vite'

// The page lives in web/ and builds to web/dist, which `ntk serve` serves beside its routes, so the page and the
// routes share one origin and the page's calls need no cross origin grant. `npm run dev` hands the routes to that
// server, on its port.
const ROUTES = ['/session', '/roles', '/search', '/ask', '/doc', '/stats', '/events']

export default defineConfig({
  root: 'web',
  base: './',
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2022' },
  server: { proxy: Object.fromEntries(ROUTES.map((route) => [route, 'http://127.0.0.1:6435'])) },
})
