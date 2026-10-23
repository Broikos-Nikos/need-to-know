/**
 * The page as a visitor meets it where no server of its own answers, as on the project's published page: the built
 * `web/dist`, served under a path of its own at 127.0.0.2, an address the page does not take for its local server, so
 * it replays the recorded questions. What it draws is checked against the recording it fetched:
 *
 *   npm run build && npm run check:page
 *
 * 0. At load, before any click, the first question is drawn as every role, each card with its answer or refusal as
 *    recorded, in the first screen, an answer larger than the body and white, a refusal not (RC-F3).
 * 1. The provenance line says what it is, a recording of the app, then when, at which commit and with which models the
 *    recording was made (RC-F5).
 * 2. Each question, replayed as the role that asks it, draws that role's filter, each search's first rows with how
 *    many it held, and the answer as recorded, and draws no unit the role's recorded trace does not hold. The answer
 *    is the trace's loudest text, larger than the excerpts and the body, in the body's face and white, the excerpts
 *    not, and each citation links to the excerpt it names (DE-F1).
 * 3. As every role, each role has its card with its answer or its refusal as recorded.
 * 4. The licence and the source are on the page's face; no request fails and the console stays quiet.
 * 5. A replay whose file comes late draws nothing over the one picked after it (DR-F39).
 * 6. Live, against a stand in for `ntk serve` on the loopback that forgets its sessions as a restart does, the page
 *    signs in again and asks again (DR-F38).
 * 7. With JavaScript off, the notice says where the figures come from and no line says a server is being looked for
 *    (HS-F24, HS-F27).
 * 8. Live, a question the server refuses clears the trace below it, and a server that stops is named as stopped in
 *    the state line and in the feed (HS-F25, HS-F26).
 *
 * Static checks of the same files, the trace's labels and the rows the recorder keeps, are in tests/test_page.py; this
 * is the one that runs the page.
 */

import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

import { dist, serve } from './serve.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const traces = resolve(root, 'web', 'public', 'recorded', 'traces')
const SOURCE = 'https://github.com/Broikos-Nikos/need-to-know'
const SEARCHES = ['dense', 'sparse', 'fused', 'reranked']
const SHOWN = Number(/^const SHOWN = (\d+)$/m.exec(readFileSync(resolve(root, 'web', 'src', 'main.ts'), 'utf8'))[1])
const STOPPED = /^const STOPPED = '([^']+)'$/m.exec(readFileSync(resolve(root, 'web', 'src', 'main.ts'), 'utf8'))[1]
const RECORDING = 'A recording of the app'  // how the mode line opens where the page replays
const number = (n) => n.toLocaleString('en')
const flat = (text) => text.replace(/\s+/g, ' ').trim()
const later = (ms) => new Promise((done) => setTimeout(done, ms))

let failed = 0
const fail = (what, detail) => {
  failed++
  console.error(`FAIL  ${what}`)
  if (detail) console.error(`      ${detail}`)
}
const ok = (what) => console.log(`  ok      ${what}`)

// Every visible run of two words or more whose letters are mostly of one language, under a declared language that is
// not it: a screen reader reads what the markup says the language is, not what the letters are.
const misread = (page) => page.evaluate(() => {
  const wrong = []
  for (const node of document.body.querySelectorAll('*')) {
    if (node.closest('[hidden]') || ['SCRIPT', 'STYLE'].includes(node.tagName)) continue
    const own = [...node.childNodes].filter((child) => child.nodeType === Node.TEXT_NODE).map((child) => child.textContent)
      .join(' ').trim()
    const letters = own.match(/\p{L}/gu) ?? []
    if (own.split(/\s+/).length < 2 || !letters.length) continue
    const greek = letters.filter((letter) => /\p{Script=Greek}/u.test(letter)).length
    const is = greek * 2 > letters.length ? 'el' : 'en'
    const said = (node.closest('[lang]')?.getAttribute('lang') ?? '').split('-')[0]
    if (said !== is) wrong.push(`<${node.tagName.toLowerCase()} lang=${said}> ${own.slice(0, 40)}`)
  }
  return wrong
})

// Each visible control's boundary, the better of its border and its fill against the colour behind it, as WCAG
// 1.4.11 asks of a control, 3:1; colours read through a canvas, since the browser hands some back as lab() or oklch().
// A control drawn as underlined text, with no fill and no border, is a link in prose to a reader, as the workspace's
// contrast gate counts links: its text is held to the rule for text, 4.5:1 (1.4.3).
const bounded = (page) => page.evaluate(() => {
  const probe = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
  const rgb = (css) => {
    probe.clearRect(0, 0, 1, 1)
    probe.fillStyle = '#000'
    probe.fillRect(0, 0, 1, 1)
    probe.fillStyle = css
    probe.fillRect(0, 0, 1, 1)
    return [...probe.getImageData(0, 0, 1, 1).data]
  }
  const opaque = (css) => {  // painted over nothing, so a transparent fill reads as none, not as the black under it
    probe.clearRect(0, 0, 1, 1)
    probe.fillStyle = css
    probe.fillRect(0, 0, 1, 1)
    return probe.getImageData(0, 0, 1, 1).data[3] === 255
  }
  const light = ([r, g, b]) => [r, g, b].map((c) => c / 255).map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
    .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0)
  const ratio = (a, b) => (Math.max(light(a), light(b)) + 0.05) / (Math.min(light(a), light(b)) + 0.05)
  const behind = (node) => {
    for (let up = node.parentElement; up; up = up.parentElement) {
      const fill = getComputedStyle(up).backgroundColor
      if (opaque(fill)) return rgb(fill)
    }
    return rgb(getComputedStyle(document.body).backgroundColor)
  }
  return [...document.querySelectorAll('select, button, input:not([type="checkbox"]), textarea')]
    .filter((node) => !node.closest('[hidden]') && node.getClientRects().length)
    .map((node) => {
      const style = getComputedStyle(node)
      const outside = behind(node)
      const bordered = parseFloat(style.borderTopWidth) > 0
      const filled = opaque(style.backgroundColor)
      const name = `${node.tagName.toLowerCase()}${node.id ? `#${node.id}` : node.className ? `.${node.className.split(' ')[0]}` : ''}`
      if (!bordered && !filled && style.textDecorationLine.includes('underline')) {
        return [`${name}, as text`, ratio(rgb(style.color), outside), 4.5]
      }
      const border = bordered ? ratio(rgb(style.borderTopColor), outside) : 1
      const fill = filled ? ratio(rgb(style.backgroundColor), outside) : 1
      if (Math.max(border, fill) >= 3) return [name, Math.max(border, fill), 3]
      // Or its group draws it, as the workspace gate allows: an ancestor holding two controls or more, with a
      // boundary of its own by the same rule, as a list of rows that are one control's states.
      for (let up = node.parentElement; up && up !== document.body; up = up.parentElement) {
        if (up.querySelectorAll('select, button, input, textarea').length < 2) continue
        const around = getComputedStyle(up)
        const edge = parseFloat(around.borderTopWidth) > 0 ? ratio(rgb(around.borderTopColor), behind(up)) : 1
        const ground = opaque(around.backgroundColor) ? ratio(rgb(around.backgroundColor), behind(up)) : 1
        if (Math.max(edge, ground) >= 3) return [`${name}, by its group`, Math.max(edge, ground), 3]
      }
      return [name, Math.max(border, fill), 3]
    })
})

// The answer the loudest text of its trace (DE-F1): larger than the excerpts it was drawn from and than the body, in
// the body's face and white, the excerpts not; and each citation a link to the excerpt it names.
const loudest = (page) => page.evaluate(() => {
  const look = (selector) => {
    const node = document.querySelector(selector)
    return node && getComputedStyle(node)
  }
  const said = look('#trace [data-stage="answer"] [data-said]')
  const text = look('#trace [data-units] [data-text]') ?? { fontSize: '0px', color: 'none' }  // a trace can show no excerpt
  const body = getComputedStyle(document.body)
  if (!said) return ['no answer drawn']
  const wrong = []
  const size = (style) => parseFloat(style.fontSize)
  if (!(size(said) > size(text) && size(said) > size(body))) {
    wrong.push(`the answer at ${said.fontSize}, the excerpts at ${text.fontSize}, the body at ${body.fontSize}`)
  }
  if (said.fontFamily !== body.fontFamily) wrong.push(`the answer in ${said.fontFamily.split(',')[0]}`)
  if (said.color !== 'rgb(255, 255, 255)' || text.color === said.color) {
    wrong.push(`the answer in ${said.color}, the excerpts in ${text.color}`)
  }
  for (const item of document.querySelectorAll('#trace [data-cite]')) {
    const link = item.querySelector('a')
    const number = /^#excerpt-(\d+)$/.exec(link?.getAttribute('href') ?? '')?.[1]
    const where = number && document.getElementById(`excerpt-${number}`)?.querySelector('[data-where]')?.textContent
    if (where !== `${number}. ${item.textContent}`) wrong.push(`the citation ${item.textContent.slice(0, 40)} links to no excerpt it names`)
  }
  return wrong
})

// The design the design eye's pass asked for (DE-F2 to DE-F5): every excerpt but the one the answer cites closed to a
// line, and that one whole and edged in the flame; every label a chip, in the filter, on each excerpt and on each card;
// five type sizes on the page, each a fifth or more above the one under it, and one grey; the disk's bytes to three
// figures, the exact count beneath.
const designed = (page) => page.evaluate(() => {
  const wrong = []
  for (const block of document.querySelectorAll('#trace [data-units]')) {
    const text = block.querySelector('[data-text]')
    const style = getComputedStyle(text)
    const lines = text.clientHeight / parseFloat(style.lineHeight)
    const whole = text.scrollHeight <= text.clientHeight + 1
    if (block.dataset.cited !== undefined) {
      if (!whole || !getComputedStyle(block).boxShadow.includes('255, 106, 0')) {
        wrong.push(`the cited ${block.id} not whole and edged in the flame`)
      }
    } else if (!whole && Math.round(lines) !== 1) wrong.push(`${block.id} closed to ${lines.toFixed(1)} lines, not one`)
  }
  const chips = [...document.querySelectorAll('#trace [data-reads] [data-label], #trace [data-units] [data-label], #compare [data-label]')]
  const holders = [...document.querySelectorAll('#trace [data-reads], #trace [data-units], #compare [data-role]')]
  if (holders.some((holder) => !holder.querySelector('[data-label]'))) wrong.push('a filter, an excerpt or a card shows no label')
  if (chips.some((chip) => !(parseFloat(getComputedStyle(chip).borderTopWidth) > 0))) wrong.push('a label drawn without its chip')
  const sizes = new Set()
  const greys = new Set()
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const element = node.parentElement
    if (!node.textContent.trim() || !element || element.closest('[hidden], noscript') || !element.getClientRects().length) continue
    const style = getComputedStyle(element)
    sizes.add(Math.round(parseFloat(style.fontSize) * 100) / 100)
    const [r, g, b] = (style.color.match(/\d+(\.\d+)?/g) ?? []).map(Number)
    if (r === g && g === b) continue
    if (Math.abs(r - g) < 20 && Math.abs(g - b) < 20 && r < 200) greys.add(style.color)  // a warm grey: one only
  }
  const steps = [...sizes].sort((a, b) => a - b)
  if (steps.length > 5) wrong.push(`${steps.length} type sizes: ${steps.join(', ')} px`)
  steps.slice(1).forEach((step, at) => {
    if (step / steps[at] < 1.199) wrong.push(`type at ${steps[at]} and ${step} px, less than a fifth apart`)
  })
  if (greys.size > 1) wrong.push(`${greys.size} greys: ${[...greys].join(', ')}`)
  const tiles = [...document.querySelectorAll('#figures [data-tile]')].slice(-3)
  for (const tile of tiles) {
    const value = tile.querySelector('[data-value]')?.textContent ?? ''
    const under = tile.querySelector('[data-under]')?.textContent ?? ''
    if (!/^\d+(\.\d+)? (GB|MB|kB)$/.test(value) || !/^[\d,]+ bytes$/.test(under)) {
      wrong.push(`the tile ${value} over ${under}, not bytes to three figures over the exact count`)
    }
  }
  return wrong
})

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

// A stand in for `ntk serve`: its routes, a recorded trace for a search, and sessions it can forget as a restart does.
async function standIn() {
  const figures = JSON.parse(readFileSync(resolve(root, 'web', 'public', 'recorded', 'scale.json'), 'utf8'))
  const traced = asked[index.questions[0].id]
  const held = new Map()
  const seen = { signins: 0, refused: 0 }
  let gone = false  // a server stopped: every route's connection dropped, as a refused one reaches the page
  let garbled = false  // the next search answered by something in the way, a proxy's page rather than the server's JSON
  const send = (response, status, body) => {
    response.writeHead(status, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(body))
  }
  const body = async (request) => {
    let text = ''
    for await (const chunk of request) text += chunk
    return JSON.parse(text || '{}')
  }
  const { url, stop } = await serve({ host: '127.0.0.1', base: '/', routes: async (request, response, path) => {
    if (gone && ['/roles', '/session', '/search', '/ask', '/events', '/stats'].includes(path)) {
      request.socket.destroy()
      return true
    }
    if (path === '/roles') return send(response, 200, { roles: index.roles }) ?? true
    if (path === '/session') {
      const { role } = await body(request)
      seen.signins++
      const token = `token-${seen.signins}`
      held.set(token, role)
      return send(response, 200, { token, role }) ?? true
    }
    if (!['/search', '/ask', '/events', '/stats'].includes(path)) return false
    const role = held.get((request.headers.authorization ?? '').replace(/^Bearer /, ''))
    if (!role) {
      seen.refused++
      return send(response, 401, { detail: 'no session: sign in with POST /session and send its token' }) ?? true
    }
    if (path === '/events') return send(response, 200, { events: [] }) ?? true
    if (path === '/stats') return send(response, 200, { ...figures, measured: '2026-10-21T10:00:00+00:00' }) ?? true
    const { question } = await body(request)
    if (garbled) {
      garbled = false
      response.writeHead(502, { 'Content-Type': 'text/html' })
      response.end('<html><body>Bad gateway</body></html>')
      return true
    }
    if (!/[\p{L}\p{N}]/u.test(question ?? '')) {  // as the retrieval API refuses a question of no word (HS-F13)
      return send(response, 400, { detail: 'a question needs at least one word' }) ?? true
    }
    const { answer, verdict, counts, ...trace } = traced[role]  // a search's trace holds no answer and every row
    return send(response, 200, trace) ?? true
  } })
  return { url, stop, seen, restart: () => held.clear(), down: () => { gone = true }, garble: () => { garbled = true } }
}

const replayed = await serve()
const live = await standIn()
const browser = await chromium.launch()
const opened = async (url) => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' })
  await page.goto(url)
  return page
}
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' })
  const errors = []
  page.on('console', (message) => message.type() === 'error' && errors.push(message.text()))
  page.on('pageerror', (error) => errors.push(String(error)))
  page.on('requestfailed', (request) => errors.push(`${request.url()} failed`))
  page.on('response', (response) => response.status() >= 400 && errors.push(`${response.url()} ${response.status()}`))
  await page.goto(replayed.url)
  await page.waitForFunction((opens) => document.getElementById('mode')?.textContent?.startsWith(opens), RECORDING, {
    timeout: 30_000,
  })

  // 0. Before any click, the first recorded question as every role, the box checked (RC-F3): each role's card with its
  // answer or its refusal as recorded, in the first screen, an answer the loudest text of its card and a refusal not.
  const opening = index.questions[0]
  const openingCards = index.roles.filter((role) => asked[opening.id][role])
    .map((role) => [role, flat(asked[opening.id][role].answer.answer)])
  const drawnFirst = await page.waitForFunction(
    (want) => JSON.stringify([...document.querySelectorAll('#compare [data-role]')].map((card) =>
      [card.dataset.role ?? '', (card.querySelector('[data-said]')?.textContent ?? '').replace(/\s+/g, ' ').trim()]),
    ) === JSON.stringify(want),
    openingCards,
    { timeout: 10_000 },
  ).then(() => true, () => false)
  const atLoad = await page.evaluate(() => {
    const look = (node) => node && { size: parseFloat(getComputedStyle(node).fontSize), colour: getComputedStyle(node).color }
    return { every: document.getElementById('every').checked, top: Math.round(document.getElementById('compare').getBoundingClientRect().top),
      body: look(document.body), answered: look(document.querySelector('#compare [data-said="answered"]')),
      refused: look(document.querySelector('#compare [data-said="not_found"]')) }
  })
  const loud = atLoad.answered && atLoad.answered.size > atLoad.body.size && atLoad.answered.colour === 'rgb(255, 255, 255)' &&
    (!atLoad.refused || atLoad.refused.colour !== atLoad.answered.colour)
  if (drawnFirst && atLoad.every && atLoad.top < 900 && loud) {
    ok(`at load, ${opening.id} drawn as every role: ${openingCards.length} cards from ${atLoad.top} px, the answers loudest`)
  } else {
    fail(`at load, ${opening.id} is not drawn as every role in the first screen, its answers loudest`, JSON.stringify(atLoad))
  }

  // 1. Where the recording came from, what it is said first (RC-F5).
  const tree = index.clean ? '' : ', the tree holding changes'
  const said = `${RECORDING}, made ${index.recorded} at ${index.commit}${tree}; answers by ${index.models.join(', ')}, ` +
    `retrieval by ${index.embedder} and ${index.reranker}. No server answers here, so the page replays it.`
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
    const lists = await page.$$eval('#trace [data-hits]', (all) => all.map((list) => list.children.length))
    const drawn = await page.$$eval('#trace [data-unit]', (rows) => rows.map((row) => row.dataset.unit))
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
    wrong.push(...await loudest(page))
    wrong.push(...await designed(page))
    if (wrong.length) {
      fail(`${question.id} as ${question.role}: ${wrong.length} thing${wrong.length === 1 ? '' : 's'} drawn otherwise than recorded`, wrong.join('; '))
    } else {
      ok(`${question.id} as ${question.role}: the filter, ${SEARCHES.length} searches' rows and counts, and the answer as ` +
        'recorded, the loudest text, each citation a link to its excerpt')
    }
  }

  // The last of them in Greek: its trace's Greek marked so, the page's English not.
  const traceEdges = await bounded(page)  // the trace's controls, its rows among them, before the cards replace it
  const traced = await misread(page)
  if (traced.length) fail(`${traced.length} run${traced.length === 1 ? '' : 's'} of a drawn trace in another language than declared`, traced.slice(0, 3).join('; '))
  else ok(`${index.questions.at(-1).id}'s trace: every run of text declares the language it is in`)

  // 3. Each question as every role.
  await page.setChecked('#every', true)
  for (const question of index.questions) {
    const by = asked[question.id]
    const expected = index.roles.filter((role) => by[role]).map((role) => [role, flat(by[role].answer.answer)])
    await page.selectOption('#pick', question.id)
    await page.click('#replay')
    const read = () => [...document.querySelectorAll('#compare [data-role]')].map((card) =>
      [card.dataset.role ?? '', (card.querySelector('[data-said]')?.textContent ?? '').replace(/\s+/g, ' ').trim()])
    // Drawn when the cards are this question's: two questions can open with the same refusal. The page's policy
    // allows no eval, so the reading is written out again here.
    await page.waitForFunction(
      (want) => JSON.stringify([...document.querySelectorAll('#compare [data-role]')].map((card) =>
        [card.dataset.role ?? '', (card.querySelector('[data-said]')?.textContent ?? '').replace(/\s+/g, ' ').trim()]),
      ) === JSON.stringify(want),
      expected,
      { timeout: 10_000 },
    ).catch(() => {})
    const cards = await page.evaluate(read)
    if (JSON.stringify(cards) === JSON.stringify(expected)) ok(`${question.id} as every role: ${cards.length} cards, each answer or refusal as recorded`)
    else fail(`${question.id} as every role: the cards are not the recording's`, JSON.stringify(cards).slice(0, 300))
  }

  const carded = await misread(page)
  if (carded.length) fail(`${carded.length} run${carded.length === 1 ? '' : 's'} of the cards in another language than declared`, carded.slice(0, 3).join('; '))
  else ok(`${index.questions.at(-1).id} as every role: every card's text declares the language it is in`)
  const edges = [...traceEdges, ...await bounded(page)]
  const faint = edges.filter(([, seen, needed]) => seen < needed)
  const worst = (needed) => Math.min(...edges.filter((edge) => edge[2] === needed).map(([, seen]) => seen)).toFixed(2)
  if (faint.length) {
    fail(`${faint.length} of ${edges.length} controls below their contrast`, faint.map(([what, seen, needed]) =>
      `${what} ${seen.toFixed(2)}:1 of ${needed}:1`).join('; '))
  } else {
    const texts = edges.filter((edge) => edge[2] === 4.5).length
    ok(`${edges.length} controls: boundaries at ${worst(3)}:1 or better` +
      (texts ? `, the ${texts} drawn as text at ${worst(4.5)}:1 or better` : ', none drawn as text alone'))
  }

  // 4. The licence and the source, and nothing that failed.
  const foot = await page.$eval('footer', (node) => ({ text: node.textContent ?? '', href: node.querySelector('a')?.href }))
  if (foot.href === SOURCE && flat(foot.text).includes('MIT licence')) ok(`the licence and the source are on the page: ${SOURCE}`)
  else fail('the page does not name its licence and source', JSON.stringify(foot))
  if (errors.length) fail(`${errors.length} console errors or failed requests`, errors.slice(0, 5).join('; '))
  else ok('no console error and no failed request')

  // The four searches across, two by two or one above another at every width, never three and one left alone.
  const rows = []
  for (const width of [1280, 1088, 800, 390]) {
    await page.setViewportSize({ width, height: 900 })
    await page.setChecked('#every', false)
    await page.selectOption('#pick', opening.id)
    await page.selectOption('#role', opening.role)
    await page.click('#replay')
    await page.waitForSelector('#trace [data-stage="dense"]')
    const tops = await page.$$eval('#trace [data-stage]', (cards) => cards.filter((card) => ['dense', 'sparse', 'fused', 'reranked']
      .includes(card.dataset.stage)).map((card) => Math.round(card.getBoundingClientRect().top)))
    const counts = Object.values(tops.reduce((by, top) => ({ ...by, [top]: (by[top] ?? 0) + 1 }), {})).join('+')
    rows.push(`${width}: ${counts}`)
    if (!['4', '2+2', '1+1+1+1'].includes(counts)) fail(`at ${width} the four searches sit ${counts} to a row`)
  }
  await page.setViewportSize({ width: 1280, height: 900 })
  if (rows.every((row) => /: (4|2\+2|1\+1\+1\+1)$/.test(row))) ok(`the four searches to a row: ${rows.join(', ')}`)

  // 5. A replay whose file comes late, then one already fetched: the page shows the one picked last.
  const [first, slow] = index.questions
  const late = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' })
  await late.route(`**/traces/${slow.id}.json`, async (route) => {
    await later(800)
    await route.continue()
  })
  await late.goto(replayed.url)
  await late.waitForFunction((opens) => document.getElementById('mode')?.textContent?.startsWith(opens), RECORDING)
  await late.waitForSelector('#compare [data-role]')  // the cards drawn at load, then one role's trace asked for
  await late.setChecked('#every', false)
  const wanted = flat(asked[first.id][first.role].answer.answer)
  await late.selectOption('#role', first.role)
  await late.selectOption('#pick', first.id)
  await late.click('#replay')
  await late.waitForFunction((answer) => (document.getElementById('trace')?.textContent ?? '').replace(/\s+/g, ' ')
    .includes(answer), wanted)
  await late.selectOption('#pick', slow.id)
  await late.click('#replay')
  await late.selectOption('#pick', first.id)
  await late.click('#replay')
  await later(1600)  // past the held file
  if (flat(await late.textContent('#trace')).includes(wanted)) ok(`${slow.id}'s file held back, then ${first.id}: ${first.id} stays drawn`)
  else fail(`${slow.id}'s file, coming after ${first.id} was picked again, drew over it`)
  await late.close()

  // 6. Live, the stand in forgets every session after the page loads, as a restarted server does.
  const signed = await opened(live.url)
  await signed.waitForFunction(() => document.getElementById('mode')?.textContent?.startsWith('Live'), null, { timeout: 10_000 })
  await signed.waitForFunction(() => document.getElementById('feed')?.hidden === false, null, { timeout: 10_000 })
  live.restart()
  const before = { ...live.seen }
  await signed.setChecked('#slow', false)
  await signed.selectOption('#role', index.roles[0])
  await signed.fill('#question', 'How long are call recordings kept?')
  await signed.click('button[value="search"]')
  await signed.waitForFunction((role) => document.getElementById('state')?.textContent === `Searched as ${role}.`, index.roles[0], {
    timeout: 10_000,
  }).catch(() => {})
  const state = await signed.textContent('#state')
  const again = live.seen.signins - before.signins
  if (state === `Searched as ${index.roles[0]}.` && live.seen.refused > before.refused && again >= 1) {
    ok(`live, a session the server forgot: refused once, signed in again, and the search drawn`)
  } else {
    fail('live, the page did not sign in again after its session was forgotten', `${JSON.stringify(state)}; ` +
      `${live.seen.refused - before.refused} refused, ${again} sign ins after`)
  }
  // The feed's line is a status region, read aloud at each change, so a poll that finds nothing new leaves it as it
  // stands (PF-F14): watched through a poll and a half, it changes not once.
  const changes = await signed.evaluate((wait) => new Promise((done) => {
    const line = document.getElementById('feed-origin')
    let seen = 0
    const watch = new MutationObserver((records) => { seen += records.length })
    watch.observe(line, { childList: true, characterData: true, subtree: true })
    setTimeout(() => { watch.disconnect(); done({ role: line.getAttribute('role'), seen }) }, wait)
  }), 7_500)
  if (changes.role === 'status' && changes.seen === 0) ok('live, the feed\'s line a status region, left alone by a poll with nothing new')
  else fail('live, the feed\'s line is no status region, or a poll rewrites it unchanged', JSON.stringify(changes))
  await signed.close()

  // 7. With JavaScript off: the notice says where the figures come from, and no line says a server is being looked for
  // (HS-F24, HS-F27).
  const bare = await browser.newContext({ viewport: { width: 1280, height: 900 }, javaScriptEnabled: false })
  const still = await bare.newPage()
  await still.goto(replayed.url)
  const notice = flat(await still.locator('[data-notice]').innerText().catch(() => ''))
  const looking = await still.locator('#mode').isVisible()
  if (notice.endsWith('recorded in the repository by a command its README names.') && !looking) {
    ok('without JavaScript, the notice says where the figures come from, and no line claims a search')
  } else {
    fail('without JavaScript, the page says what it cannot do', `notice ${JSON.stringify(notice.slice(-80))}; ` +
      `the mode line ${looking ? 'shown' : 'hidden'}`)
  }
  await bare.close()

  // 8. Live, a question the server refuses takes the last trace away with it, and a server that stops is named as
  // stopped, in the state line and in the feed, not in the browser's words (HS-F25, HS-F26). The stand in goes last.
  const lost = await opened(live.url)
  await lost.waitForFunction(() => document.getElementById('feed')?.hidden === false, null, { timeout: 10_000 })
  await lost.setChecked('#slow', false)
  await lost.selectOption('#role', index.roles[0])
  await lost.fill('#question', 'How long are call recordings kept?')
  await lost.click('button[value="search"]')
  await lost.waitForFunction(() => document.querySelectorAll('#trace [data-stage]').length > 0, null, { timeout: 10_000 })
  await lost.fill('#question', '   ')
  await lost.click('button[value="search"]')
  await lost.waitForFunction(() => document.getElementById('state')?.classList.contains('error'), null, { timeout: 10_000 })
    .catch(() => {})
  const refused = { state: await lost.textContent('#state'), stages: await lost.locator('#trace [data-stage]').count() }
  live.garble()
  await lost.fill('#question', 'How long are call recordings kept?')
  await lost.click('button[value="search"]')
  await lost.waitForFunction(() => /^the server answered/.test(document.getElementById('state')?.textContent ?? ''), null, {
    timeout: 10_000,
  }).catch(() => {})
  const proxied = await lost.textContent('#state')
  live.down()
  await lost.fill('#question', 'How long are call recordings kept?')
  await lost.click('button[value="search"]')
  await lost.waitForFunction((stopped) => document.getElementById('state')?.textContent === stopped, STOPPED, {
    timeout: 10_000,
  }).catch(() => {})
  const stoppedSays = await lost.textContent('#state')
  const told = await lost.waitForFunction((stopped) => (document.getElementById('feed-origin')?.textContent ?? '')
    .startsWith(stopped), STOPPED, { timeout: 15_000 }).then(() => true, () => false)
  const plain = proxied === 'the server answered 502 with something other than JSON'
  if (refused.state === 'a question needs at least one word' && refused.stages === 0 && plain && stoppedSays === STOPPED &&
      told) {
    ok('live, a refused question takes the last trace with it; a reply not the server\'s and a stopped server are named so')
  } else {
    fail('live, a refusal, a proxy\'s page or a stopped server leaves the page saying something else',
      `${JSON.stringify(refused)}; ${JSON.stringify(proxied)}; ${JSON.stringify(stoppedSays)}; the feed: ` +
      JSON.stringify(await lost.textContent('#feed-origin')))
  }
  await lost.close()
} finally {
  await browser.close()
  replayed.stop()
  live.stop()
}

if (failed > 0) {
  console.error(`\npage: ${failed} check${failed === 1 ? '' : 's'} failed; a recording can be right and the page that replays it wrong`)
  process.exit(1)
}
console.log('page: replayed where no server answers, every figure drawn is the recording’s, and live it signs in again')
