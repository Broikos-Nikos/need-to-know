/**
 * The one program this repository runs that npm did not install: ffmpeg, which turns the page's recording into the
 * picture at the top of the README (`tools/capture.mjs`).
 *
 * Copied from the workspace's other projects, which learned it the hard way: called bare, whatever ffmpeg comes first
 * on PATH wins, and a missing one threw after a browser had launched and a recording had run. So it is named, resolved
 * once before anything starts, checked by running `-version`, and the version that made the committed picture is
 * written to `docs/capture.json`. `FFMPEG` set to a full path wins over PATH. A version of a program on someone
 * else's machine is not something a repository can pin; which one made the picture, it can record.
 */

import { execFileSync } from 'node:child_process'

export const FFMPEG_ENV = 'FFMPEG'

/** How a reader is told to get it. Named here so the README, the gate and the
    error all quote the same sentence rather than three similar ones. */
export const FFMPEG_HOW =
  'Install ffmpeg and put it on PATH (winget install Gyan.FFmpeg, brew install ffmpeg, apt install ffmpeg), or set FFMPEG to its full path.'

/**
 * @returns {{ path: string, version: string } | { path: string, error: string }}
 */
export function resolveFfmpeg() {
  const path = process.env[FFMPEG_ENV] || 'ffmpeg'
  try {
    const out = execFileSync(path, ['-version'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
    /* The first line is `ffmpeg version 8.1.2-essentials_build-www.gyan.dev
       Copyright ...`, and the version is what comes between. */
    /* Trimmed before the replace, not after. ffmpeg's output is CRLF here, and
       `.*$` in a pattern without the m flag will not cross the `\r`, so the
       first version of this printed the whole copyright line into the gate's
       ok line and into docs/capture.json. */
    const first = out.split('\n')[0].trim()
    const version = first.replace(/^ffmpeg version\s*/, '').replace(/\s*Copyright.*$/, '').trim()
    return { path, version: version || first }
  } catch (err) {
    const e = /** @type {any} */ (err)
    return { path, error: e.code === 'ENOENT' ? `not found as ${JSON.stringify(path)}` : String(e.message).split('\n')[0] }
  }
}

/** The same, and it exits rather than returning a problem. */
export function requireFfmpeg() {
  const found = resolveFfmpeg()
  if ('error' in found) {
    console.error(`FAIL  ffmpeg: ${found.error}`)
    console.error(`      ${FFMPEG_HOW}`)
    console.error('      This is checked before the browser starts, because it used to throw ENOENT after one.')
    process.exit(1)
  }
  return found
}
