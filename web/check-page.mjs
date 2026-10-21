/**
 * The page as a visitor meets it where no server of its own answers, as on the project's published page: the built
 * `web/dist`, served under a path of its own at 127.0.0.2, an address the page does not take for its local server, so
 * it replays the recorded questions. What it draws is checked against the recording it fetched:
 *
 *   npm run build && npm run check:page
 *
 * 1. The provenance line says when, at which commit and with which models the recording was made.
 * 2. Each question, replayed as the role that asks it, draws that role's filter, each search's first rows with how
 *    many it held, and the answer as recorded, and draws no unit the role's recorded trace does not hold.
 * 3. As every role, each role has its card with its answer or its refusal as recorded.
 * 4. The licence and the source are on the page's face; no request fails and the console stays quiet.
 *
 * Static checks of the same files, the trace's labels and the rows the recorder keeps, are in tests/test_page.py; this
 * is the one that runs the page.
 */

import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dist = resolve(root, 'web', 'dist')
const traces = resolve(root, 'web', 'public', 'recorded', 'traces')
const HOST = '127.0.0.2'
const BASE = '/need-to-know/'
const SOURCE = 'https://github.com/Broikos-Nikos/need-to-know'
const SEARCHES = ['dense', 'sparse', 'fused', 'reranked']
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
}
const SHOWN = Number(/^const SHOWN = (\d+)$/m.exec(readFileSync(resolve(root, 'web', 'src', 'main.ts'), 'utf8'))[1])
const number = (n) => n.toLocaleString('en')
const flat = (text) => text.replace(/\s+/g, ' ').trim()

let failed = 0
const fail = (what, detail) => {
  failed++
  console.error(`FAIL  ${what}`)
  if (detail) console.error(`      ${detail}`)
}
const ok = (what) => console.log(`  ok      ${what}`)

if (!existsSync(resolve(dist, 'index.html'))) {
  console.error('check:page: no web/dist; npm run build makes it')
  process.exit(1)
}
const index = JSON.parse(readFileSync(resolve(traces, 'index.json'), 'utf8'))
for (const name of ['index.json', ...index.questions.map((q) => `${q.id}.json`)]) {
  const shipped = resolve(dist, 'recorded', 'traces', name)
  if (!existsSync(shipped) || !readFileSync(shipped).equals(readFileSync(resolve(traces, name)))) {
    console.error(`check:page: web/dist holds another ${name} than web/public; npm run build makes it again`)
    process.exit(1)
  }
}
const asked = Object.fromEntries(
  index.questions.map((q) => [q.id, JSON.parse(readFileSync(resolve(traces, `${q.id}.json`), 'utf8')).traces]),
)

// The built files, under a path of their own as a published page has one; nothing else answers.
const server = createServer((request, response) => {
  const path = decodeURIComponent(new URL(request.url, 'http://page').pathname)
  let file = path.startsWith(BASE) ? resolve(dist, `.${path.slice(BASE.length - 1)}`) : ''
  if (file && existsSync(file) && statSync(file).isDirectory()) file = resolve(file, 'index.html')
  if (!file || !(file === dist || file.startsWith(dist + sep)) || !existsSync(file)) {
    response.writeHead(404)
    response.end()
    return
  }
  response.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' })
  createReadStream(file).pipe(response)
})
await new Promise((done) => server.listen(0, HOST, done))
const url = `http://${HOST}:${server.address().port}${BASE}`

const browser = await chromium.launch()
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' })
  const errors = []
  page.on('console', (message) => message.type() === 'error' && errors.push(message.text()))
  page.on('pageerror', (error) => errors.push(String(error)))
  page.on('requestfailed', (request) => errors.push(`${request.url()} failed`))
  page.on('response', (response) => response.status() >= 400 && errors.push(`${response.url()} ${response.status()}`))
  await page.goto(url)
  await page.waitForFunction(() => document.getElementById('mode')?.textContent?.startsWith('Replayed'), null, {
    timeout: 30_000,
  })

  // 1. Where the recording came from.
  const tree = index.clean ? '' : ', the tree holding changes'
  const said = `Replayed, as no server answers here: recorded ${index.recorded} at ${index.commit}${tree}; answers by ` +
    `${index.models.join(', ')}, retrieval by ${index.embedder} and ${index.reranker}.`
  const mode = await page.textContent('#mode')
  if (mode === said) ok(`the provenance line names the recording: ${index.recorded}, ${index.commit}`)
  else fail('the provenance line is not the recording’s', `${JSON.stringify(mode)} against ${JSON.stringify(said)}`)
  const roles = await page.$$eval('#role option', (options) => options.map((option) => option.value))
  const picks = await page.$$eval('#pick option', (options) => options.map((option) => option.value))
  if (JSON.stringify(roles) === JSON.stringify(index.roles) && JSON.stringify(picks) === JSON.stringify(index.questions.map((q) => q.id))) {
    ok(`the pickers offer the recording's ${roles.length} roles and ${picks.length} questions`)
  } else {
    fail('the pickers are not the recording’s', `${roles} and ${picks}`)
  }

  // 2. Each question as the role that asks it.
  await page.setChecked('#every', false)
  await page.setChecked('#slow', false)
  for (const question of index.questions) {
    const trace = asked[question.id][question.role]
    await page.selectOption('#role', question.role)
    await page.selectOption('#pick', question.id)
    await page.click('#replay')
    // Drawn when the trace holds this question's answer, not when the state line reads as the last replay's did.
    await page.waitForFunction(
      ([role, answer]) => document.getElementById('state')?.textContent === `Replayed as ${role}, as the app answered it ` +
        'when recorded.' && (document.getElementById('trace')?.textContent ?? '').replace(/\s+/g, ' ').includes(answer),
      [question.role, flat(trace.answer.answer)],
      { timeout: 10_000 },
    ).catch(() => {})
    const text = flat(await page.textContent('#trace'))
    const lists = await page.$$eval('#trace ol.hits', (all) => all.map((list) => list.children.length))
    const drawn = await page.$$eval('#trace .hit', (rows) => rows.map((row) => row.dataset.unit))
    const wrong = []
    const { labels, passes, holds_back } = trace.filter
    if (!text.includes(`${trace.role} reads ${labels.join(', ')}`)) wrong.push('the labels the role reads')
    if (!text.includes(`${number(passes)} units pass the filter and ${number(holds_back)} are held back.`)) {
      wrong.push('the filter’s counts')
    }
    SEARCHES.forEach((stage, at) => {
      const rows = Math.min(SHOWN, trace[stage].length)
      const held = trace.counts[stage]
      if (lists[at] !== rows) wrong.push(`${stage}: ${lists[at]} rows drawn of ${rows}`)
      const count = held > rows ? `${number(held)} in all, the first ${rows} shown.` : `${number(held)} in all.`
      if (!text.includes(count)) wrong.push(`${stage}: no "${count}"`)
    })
    const foreign = drawn.filter((unit) => !(unit in trace.units))
    if (foreign.length) wrong.push(`${foreign.length} drawn units the role's trace does not hold`)
    if (!text.includes(flat(trace.answer.answer))) wrong.push('the recorded answer')
    if (wrong.length) fail(`${question.id} as ${question.role}: ${wrong.length} thing${wrong.length === 1 ? '' : 's'} drawn otherwise than recorded`, wrong.join('; '))
    else ok(`${question.id} as ${question.role}: the filter, ${SEARCHES.length} searches' rows and counts, and the answer, as recorded`)
  }

  // 3. Each question as every role.
  await page.setChecked('#every', true)
  for (const question of index.questions) {
    const by = asked[question.id]
    const expected = index.roles.filter((role) => by[role]).map((role) => [role, flat(by[role].answer.answer)])
    await page.selectOption('#pick', question.id)
    await page.click('#replay')
    const read = () => [...document.querySelectorAll('#compare .role-card')].map((card) =>
      [card.querySelector('h3')?.textContent ?? '', (card.querySelector('.said')?.textContent ?? '').replace(/\s+/g, ' ').trim()])
    // Drawn when the cards are this question's: two questions can open with the same refusal. The page's policy
    // allows no eval, so the reading is written out again here.
    await page.waitForFunction(
      (want) => JSON.stringify([...document.querySelectorAll('#compare .role-card')].map((card) =>
        [card.querySelector('h3')?.textContent ?? '', (card.querySelector('.said')?.textContent ?? '').replace(/\s+/g, ' ').trim()]),
      ) === JSON.stringify(want),
      expected,
      { timeout: 10_000 },
    ).catch(() => {})
    const cards = await page.evaluate(read)
    if (JSON.stringify(cards) === JSON.stringify(expected)) ok(`${question.id} as every role: ${cards.length} cards, each answer or refusal as recorded`)
    else fail(`${question.id} as every role: the cards are not the recording's`, JSON.stringify(cards).slice(0, 300))
  }

  // 4. The licence and the source, and nothing that failed.
  const foot = await page.$eval('footer.foot', (node) => ({ text: node.textContent ?? '', href: node.querySelector('a')?.href }))
  if (foot.href === SOURCE && flat(foot.text).includes('MIT licence')) ok(`the licence and the source are on the page: ${SOURCE}`)
  else fail('the page does not name its licence and source', JSON.stringify(foot))
  if (errors.length) fail(`${errors.length} console errors or failed requests`, errors.slice(0, 5).join('; '))
  else ok('no console error and no failed request')
} finally {
  await browser.close()
  server.close()
}

if (failed > 0) {
  console.error(`\npage: ${failed} checks failed; a recording can be right and the page that replays it wrong`)
  process.exit(1)
}
console.log('page: replayed where no server answers, every figure drawn is the recording’s')
