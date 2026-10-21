/**
 * The built page for the browser checks: `web/dist`, under a path of its own as a published page has one, at
 * 127.0.0.2, an address the page does not take for its local server, so it replays its recording as it would on any
 * other host. The workspace's page gates import `serve` from here as from every project's tools, and
 * `tools/check-page.mjs` does too, with the routes of a stand in for `ntk serve` where it asks for them.
 *
 * The port comes from the operating system and the server is this process's own, so no gate measures a page some
 * earlier run left answering.
 */

import { createReadStream, existsSync, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

export const dist = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'web', 'dist')
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
}

// The built files under `base`, after whatever `routes` answers first; nothing else answers.
export async function serve({ host = '127.0.0.2', base = '/need-to-know/', routes = async () => false } = {}) {
  if (!existsSync(resolve(dist, 'index.html'))) throw new Error('no web/dist: npm run build makes it')
  const server = createServer(async (request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://page').pathname)
    if (await routes(request, response, path)) return
    let file = path.startsWith(base) ? resolve(dist, `.${path.slice(base.length - 1)}`) : ''
    if (file && existsSync(file) && statSync(file).isDirectory()) file = resolve(file, 'index.html')
    if (!file || !(file === dist || file.startsWith(dist + sep)) || !existsSync(file)) {
      response.writeHead(404)
      response.end()
      return
    }
    response.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' })
    createReadStream(file).pipe(response)
  })
  await new Promise((done) => server.listen(0, host, done))
  const port = server.address().port
  return { url: `http://${host}:${port}${base}`, port, stop: () => server.close() }
}

// A page some other process serves, measured as it is: nothing to stop.
export async function useShared(url) {
  return { url, port: new URL(url).port, stop: () => {} }
}
