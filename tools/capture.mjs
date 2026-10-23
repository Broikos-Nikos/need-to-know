/**
 * Record the page doing what the README's first sentence says: one question, asked as every department, answered
 * only from what each may read; then one department's answer, a stage at a time.
 *
 *   npm run build && npm run capture
 *
 * The page is served where no server answers (`tools/serve.mjs`), so it replays the recording committed with it and
 * the picture shows what any visitor can make it show. Choreographed against geometry measured at 1280 by 780:
 *
 *   the form        377 to 483
 *   the state line  586
 *   the nine cards  three rows, 619 to 969, 982 to 1,236 and 1,249 to 1,600, drawn at load as every role
 *
 * So the picture opens on the cards, as the page does (RC-F3, RC-F4): the frame at 300 holds the question, the box
 * checked and the first row, sales answered beside legal and logistics refused; at 940, the other two rows, admin
 * answered among six refused. The cards are held four seconds or more before slow motion plays sales's trace one
 * stage at a time, each brought into view, and `docs/capture.json` records how long they were held, by this tool's
 * clock from the picture's first frame to the click on sales's card.
 *
 * Built as the workspace's other capture tools are, for the same reasons: ffmpeg is resolved before a browser starts
 * (`tools/ffmpeg.mjs`); the encoder writes a draft inside `.capture`, which git ignores, and the draft is renamed onto
 * `docs/need-to-know.gif` only once ffmpeg exits cleanly, since ffmpeg truncates its output before it validates its
 * filters; the browser is closed in a `finally`, since Playwright finishes a video only when its context closes, so a
 * failure keeps a whole recording and says where it is; and `CAPTURE_FAIL_AT=start` and `CAPTURE_FAIL_AT=encode` are
 * the seams the workspace's gate drives the failure path through.
 */

import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { basename, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

import { requireFfmpeg } from './ffmpeg.mjs'
import { serve } from './serve.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const OUT = resolve(root, 'docs/need-to-know.gif')
const WORK = resolve(root, '.capture')

const SIZE = { width: 1280, height: 780 }
const SCROLLS = [300, 940]  // the question and the first row of cards, then the other two rows
const HOLD = 2500  // each of them, in milliseconds: the cards held 4 s or more before the trace
const CROP = 'crop=1280:720:0:0'
const FPS = 8
const WIDTH = 880
const COLOURS = 24  // the accent, the measure's blue, the sync's green and the greys: sixteen banded the blue
const TURN = 1200  // web/src/main.ts, each stage's turn in slow motion
const FAIL_AT = process.env.CAPTURE_FAIL_AT ?? ''

rmSync(WORK, { recursive: true, force: true })
mkdirSync(WORK, { recursive: true })

// Before the browser and before the try: requireFfmpeg exits, and inside the try that would read as a failed recording.
const FFMPEG = requireFfmpeg()
console.log(`ffmpeg ${FFMPEG.version}${process.env.FFMPEG ? ` (FFMPEG=${FFMPEG.path})` : ''}`)

const server = await serve()
let browser = null
let context = null
let failure = null
let looked = null
let startedAt = 0
let videoStart = 0

try {
  browser = await chromium.launch()
  // Reduced motion: the stages and the scroll jump rather than glide, so a frame changes only when the page does,
  // which is what a GIF pays for, and the picture is no less true: the page offers exactly this to anyone who asks.
  context = await browser.newContext({ viewport: SIZE, deviceScaleFactor: 1, reducedMotion: 'reduce', recordVideo: { dir: WORK, size: SIZE } })
  videoStart = Date.now()
  const page = await context.newPage()
  await page.goto(server.url, { waitUntil: 'domcontentloaded' })
  if (FAIL_AT === 'start') {
    await page.waitForTimeout(500)
    throw new Error('CAPTURE_FAIL_AT=start, the seam the failure path is tested through')
  }
  // The same question as every department, as the page draws it at load: two answers, seven refusals in the same words.
  await page.waitForSelector('#compare [data-role="admin"]', { timeout: 30_000 })
  const cards = await page.$$eval('#compare [data-role]', (all) => all.map((card) => [card.dataset.role,
    card.querySelector('[data-said]')?.dataset.said ?? '']))
  await page.evaluate((y) => window.scrollTo({ top: y, behavior: 'instant' }), SCROLLS[0])
  await page.waitForTimeout(800)
  startedAt = Date.now()  // the picture opens 0.4 s before this, on the cards
  await page.waitForTimeout(HOLD - 400)
  await page.evaluate((y) => window.scrollTo({ top: y, behavior: 'instant' }), SCROLLS[1])
  await page.waitForTimeout(HOLD)

  // One department's answer, opened from its card as a visitor opens it, a stage at a time, each scrolled into view.
  // Clicked in the page rather than through Playwright, which would first scroll the controls into view and flash the
  // top of the page between the cards and the trace.
  const held = (Date.now() - startedAt) / 1000 + 0.4  // the cards, from the picture's first frame to the trace
  await page.evaluate(() => {
    document.getElementById('slow').checked = true
    document.querySelector('#compare [data-role="sales"] button').click()  // its one button, Its stages
  })
  const stages = await page.$$eval('#trace [data-stage]', (all) => all.length)
  await page.waitForTimeout(stages * TURN + 2500)
  looked = {
    question: await page.$eval('#pick', (pick) => pick.value),
    every_role: Object.fromEntries(cards),
    every_role_held: Number(held.toFixed(2)),
    stages_played: stages,
  }

  await context.close()
  await browser.close()

  const video = readdirSync(WORK).find((name) => name.endsWith('.webm'))
  if (!video) throw new Error('playwright wrote no video')
  const webm = resolve(WORK, video)
  mkdirSync(resolve(root, 'docs'), { recursive: true })
  // `basename`, not a split on a slash: resolve() gives backslashes on Windows, and the draft would be the picture.
  const draft = resolve(WORK, basename(OUT))
  const ff = (args) => execFileSync(FFMPEG.path, ['-y', '-loglevel', 'error', ...args], { stdio: 'inherit' })
  const palette = resolve(WORK, 'palette.png')
  // A frame no different from the one before is dropped, its time given to the one before (mpdecimate, vfr).
  const filters = `${CROP},fps=${FPS},mpdecimate,scale=${WIDTH}:-1:flags=lanczos`
  const trim = ['-ss', String(Math.max(0, (startedAt - videoStart) / 1000 - 0.4))]
  if (FAIL_AT === 'encode') {
    const kept = readdirSync(WORK).filter((name) => name.endsWith('.webm'))
    const bytes = kept.reduce((total, name) => total + statSync(resolve(WORK, name)).size, 0)
    console.error('FAIL  CAPTURE_FAIL_AT=encode, the seam that proves the success path runs')
    console.error(`      the recording is in ${WORK}, ${bytes} bytes, finished and kept, for looking at`)
    console.error('      .capture is in .gitignore, so it cannot reach a commit. Delete it when you are done.')
    server.stop()
    process.exit(1)
  }
  ff([...trim, '-i', webm, '-vf', `${filters},palettegen=max_colors=${COLOURS}:stats_mode=diff`, palette])
  ff([...trim, '-i', webm, '-i', palette, '-lavfi', `${filters}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle`,
    '-fps_mode', 'vfr', '-loop', '0', draft])
  renameSync(draft, OUT)
  rmSync(WORK, { recursive: true, force: true })  // the webm with it: nothing here uses it, and npm run capture makes it again
  writeFileSync(resolve(root, 'docs/capture.json'), JSON.stringify({
    recorded: new Date().toISOString().slice(0, 10), ffmpeg: FFMPEG.version, viewport: SIZE, fps: FPS, width: WIDTH, looked,
  }, null, 2) + String.fromCharCode(10))
} catch (error) {
  failure = error
} finally {
  await context?.close().catch(() => {})
  await browser?.close().catch(() => {})
  server.stop()
}
if (failure) {
  console.error(`FAIL  ${failure.message}`)
  const kept = existsSync(WORK) ? readdirSync(WORK).filter((name) => name.endsWith('.webm')) : []
  if (kept.length) {
    const bytes = kept.reduce((total, name) => total + statSync(resolve(WORK, name)).size, 0)
    console.error(`      the recording is in ${WORK}, ${bytes} bytes, finished and kept, for looking at`)
    console.error('      .capture is in .gitignore, so it cannot reach a commit. Delete it when you are done.')
  } else {
    console.error(`      nothing was recorded, and ${WORK} is empty`)
  }
  process.exit(1)
}

const { size } = statSync(OUT)
console.log(`docs/need-to-know.gif   ${(size / 1e6).toFixed(2)} MB at ${FPS} fps, ${WIDTH}px wide`)
console.log(`${looked.question} as every role: ${Object.values(looked.every_role).filter((said) => said === 'answered').length} ` +
  `answered of ${Object.keys(looked.every_role).length}, held ${looked.every_role_held} s by the capture's clock; then ` +
  `${looked.stages_played} stages played as sales`)
