// The instrument page (T6.2): a role's question through the local server, each stage drawn from the trace the server
// returns. Every unit the trace names passed the role's filter on the server; the page adds nothing to it and writes
// text only as text, never as markup. Slow motion plays the trace it was given one stage at a time, each stage with
// the seconds it took on the server, so every frame shows something that happened.

import './style.css'
import plan from './stages.json'

type Hit = { unit_id: string; rank: number; score: number }
type Unit = { doc_id: string; section: string; labels: string[]; text: string }
type Excerpt = { doc_id: string; section: string; labels: string[]; units: string[]; rank: number; text: string }
type Cited = { doc_id: string; section: string }
type Answer = {
  status: string
  answer: string
  citations: Cited[]
  dropped: unknown[]
  error?: string | null
  model?: string | null
  stop_reason?: string | null
}
type Scored = {
  item: string
  refusal: boolean
  correct: boolean
  grounded: boolean
  partial: number | null
  shown_facts: boolean | null
}
type Trace = {
  role: string
  filter: { form: string; labels: string[]; passes: number; holds_back: number }
  dense: Hit[]
  sparse: Hit[]
  fused: Hit[]
  reranked: Hit[]
  excerpts: Excerpt[]
  units: Record<string, Unit>
  seconds: Record<string, number>
  answer?: Answer
  verdict?: Scored | { error: string } | null
}
type Search = 'dense' | 'sparse' | 'fused' | 'reranked'
type Stage = { key: string; title: string; says: string; timed: string[] }

// The stages in the order the server runs them, and the name of each step it times: one map, which the test suite
// holds to the steps a trace's seconds hold (tests/test_api.py).
const STAGES: Stage[] = plan.stages
const STEPS: Record<string, string> = plan.seconds
const SEARCHES: Search[] = ['dense', 'sparse', 'fused', 'reranked']
const SHOWN = 10
const TURN = 1200 // each stage's turn in slow motion, in milliseconds
const tokens = new Map<string, string>()
const still = window.matchMedia('(prefers-reduced-motion: reduce)')
let timers: number[] = []
let pinned: string | null = null
let summary = ''

const NBSP = String.fromCharCode(160) // a figure and its unit stay on one line
const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const number = (n: number) => n.toLocaleString('en')
const secs = (value: number) => `${value >= 1 ? value.toFixed(2) : value.toFixed(3)}${NBSP}s`
const signed = (value: number, places: number) => value.toFixed(places).replace('-', '−')
const heading = (section: string) => section || '(no heading)'

function element(tag: string, className = '', text = ''): HTMLElement {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text) node.textContent = text
  return node
}

// The roles answer without a session where the server runs; a static host has no such route (N67).
async function served(): Promise<string[] | null> {
  try {
    const response = await fetch('roles')
    return response.ok ? ((await response.json()) as { roles: string[] }).roles : null
  } catch {
    return null
  }
}

async function call<T>(path: string, init: RequestInit = {}, role?: string): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set('Content-Type', 'application/json')
  if (role) headers.set('Authorization', `Bearer ${await token(role)}`)
  const response = await fetch(path, { ...init, headers })
  const body = await response.json()
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))
  return body as T
}

// The role reaches the server once, at sign in, and every query carries the session's token (N64).
async function token(role: string): Promise<string> {
  let held = tokens.get(role)
  if (!held) {
    held = (await call<{ token: string }>('session', { method: 'POST', body: JSON.stringify({ role }) })).token
    tokens.set(role, held)
  }
  return held
}

function filtered(trace: Trace): HTMLElement[] {
  const { labels, passes, holds_back } = trace.filter
  const reads = element('p', 'reads', `${trace.role} reads `)
  labels.forEach((label, index) => reads.append(index ? ', ' : '', element('span', 'label', label)))
  return [reads, element('p', 'counted', `${number(passes)} units pass the filter and ${number(holds_back)} are held back.`)]
}

function hits(trace: Trace, key: Search): HTMLElement[] {
  const all = trace[key]
  if (!all.length) return [element('p', 'minor', 'No candidates.')]
  const list = element('ol', 'hits')
  all.slice(0, SHOWN).forEach((hit, index) => {
    const unit = trace.units[hit.unit_id]
    const row = element('button', 'hit') as HTMLButtonElement
    row.type = 'button'
    row.dataset.unit = hit.unit_id
    row.setAttribute('aria-pressed', 'false')
    row.title = unit ? `${unit.doc_id}, ${heading(unit.section)}\n\n${unit.text}` : hit.unit_id
    row.append(element('span', 'rank', String(hit.rank)), element('span', 'doc', unit?.doc_id ?? '?'),
      element('span', 'score', signed(hit.score, key === 'fused' ? 4 : 3)))
    const item = element('li')
    item.style.setProperty('--i', String(index))
    item.append(row)
    list.append(item)
  })
  const count = all.length > SHOWN ? `${number(all.length)} in all, the first ${SHOWN} shown.` : `${number(all.length)} in all.`
  return [list, element('p', 'minor', count)]
}

function excerpts(trace: Trace): HTMLElement[] {
  if (!trace.excerpts.length) return [element('p', 'minor', 'None: nothing the role may read came near the question.')]
  const cited = new Set((trace.answer?.citations ?? []).map((given) => JSON.stringify([given.doc_id, given.section])))
  return trace.excerpts.map((excerpt, index) => {
    const block = element('article', 'excerpt')
    block.dataset.units = excerpt.units.join(' ')
    block.style.setProperty('--i', String(index))
    const text = element('p', 'text clamped', excerpt.text)
    const more = element('button', 'more', 'Show all of it') as HTMLButtonElement
    more.type = 'button'
    more.setAttribute('aria-expanded', 'false')
    more.addEventListener('click', () => {
      const open = !text.classList.toggle('clamped')
      more.setAttribute('aria-expanded', String(open))
      more.textContent = open ? 'Show less' : 'Show all of it'
    })
    block.append(element('p', 'where', `${index + 1}. ${excerpt.doc_id}, ${heading(excerpt.section)}`),
      element('p', 'labels', excerpt.labels.join(' + ')), text, more)
    if (cited.has(JSON.stringify([excerpt.doc_id, excerpt.section]))) {
      block.classList.add('cited')
      block.append(element('p', 'tag', 'cited by the answer'))
    }
    return block
  })
}

function answered(trace: Trace): HTMLElement[] {
  const given = trace.answer as Answer
  const nodes: HTMLElement[] = []
  if (given.status === 'answered') {
    nodes.push(element('p', 'said', given.answer))
    if (given.citations.length) {
      const list = element('ul', 'cites')
      list.append(...given.citations.map((cite) => element('li', '', `${cite.doc_id}, ${heading(cite.section)}`)))
      nodes.push(element('p', 'minor', 'It cites:'), list)
    } else {
      nodes.push(element('p', 'minor', 'It cites nothing.'))
    }
    if (given.dropped.length) {
      nodes.push(element('p', 'minor', `${given.dropped.length} of its citations named no excerpt shown and count for nothing.`))
    }
  } else if (given.status === 'not_found') {
    nodes.push(element('p', 'said', given.answer),
      element('p', 'minor', 'A refusal, in the same words whether the fact is missing or withheld.'))
  } else {
    nodes.push(element('p', 'error', `The model’s reply could not be read: ${given.error ?? given.status}.`))
  }
  if (given.model) nodes.push(element('p', 'minor', `${given.model}, its stop reason ${given.stop_reason ?? 'not given'}.`))
  return nodes
}

function judged(trace: Trace): HTMLElement[] {
  const given = trace.verdict
  if (!given) return [element('p', 'minor', 'None: the question is not a dev golden question asked as its own role.')]
  if ('error' in given) return [element('p', 'error', given.error)]
  const rows: [string, string][] = [
    ['Item', `${given.item}, a question this role ${given.refusal ? 'must be refused' : 'may be answered'}`],
    ['Correct', given.correct ? 'yes' : 'no'],
  ]
  if (!given.refusal) rows.push(['Grounded', given.grounded ? 'yes: its citations hold every expected fact' : 'no'])
  if (given.partial !== null) rows.push(['Facts stated', `${Math.round(given.partial * 100)}%`])
  if (given.shown_facts !== null) {
    rows.push(['Facts shown', given.shown_facts ? 'every expected fact was in an excerpt' : 'an expected fact was in none'])
  }
  const list = element('dl', 'scored')
  for (const [name, value] of rows) list.append(element('dt', '', name), element('dd', '', value))
  return [list]
}

function card(stage: Stage, index: number, trace: Trace): HTMLElement {
  const node = element('article', 'card')
  node.dataset.stage = stage.key
  const head = element('header', 'head')
  head.append(element('span', 'step', String(index + 1)), element('h2', '', stage.title))
  // Always four parts, an untimed stage's third one empty, so the four searches' rows line up across their columns.
  const timed = stage.timed.filter((step) => step in trace.seconds)
  node.append(head, element('p', 'says', stage.says),
    element('p', 'took', timed.map((step) => `${STEPS[step]} ${secs(trace.seconds[step])}`).join(', ')))
  const body = element('div', 'body')
  const key = stage.key
  body.append(...(key === 'filter' ? filtered(trace) : key === 'excerpts' ? excerpts(trace) : key === 'answer'
    ? answered(trace) : key === 'verdict' ? judged(trace) : hits(trace, key as Search)))
  node.append(body)
  return node
}

// Where the server's seconds went, each step to its scale in the order it ran; the steps no stage owns are this
// view's own work, drawn apart.
function timeline(trace: Trace): HTMLElement[] {
  const total = trace.seconds.total ?? 0
  const owned = new Set(STAGES.flatMap((stage) => stage.timed))
  const bar = element('div', 'strip')
  const told: string[] = []
  let at = 0
  let own = 0
  for (const [step, name] of Object.entries(STEPS)) {
    const value = trace.seconds[step]
    if (value === undefined) continue
    const piece = element('span', owned.has(step) ? 'piece' : 'piece own')
    piece.dataset.step = step
    piece.style.left = `${total ? (at / total) * 100 : 0}%`
    piece.style.width = `${total ? (value / total) * 100 : 0}%`
    piece.title = `${name} ${secs(value)}`
    bar.append(piece)
    told.push(`${name} ${secs(value)}`)
    at += value
    if (!owned.has(step)) own += value
  }
  bar.setAttribute('role', 'img')
  bar.setAttribute('aria-label', `Where the ${secs(total)} went: ${told.join(', ')}.`)
  let said = `${secs(total)} on the server`
  if (own) said += `, ${secs(own)} of it this view’s own counts and texts`
  if (total - at >= 0.0005) said += `, ${secs(total - at)} its other steps`
  return [bar, element('p', 'caption', `${said}.`)]
}

function draw(trace: Trace) {
  const shown = STAGES.filter((stage) => trace.answer || (stage.key !== 'answer' && stage.key !== 'verdict'))
  const cards = new Map(shown.map((stage, index) => [stage.key, card(stage, index, trace)]))
  const searches = element('div', 'searches')
  searches.append(...SEARCHES.map((key) => cards.get(key) as HTMLElement))
  const parts = [...timeline(trace), cards.get('filter') as HTMLElement, searches, cards.get('excerpts') as HTMLElement]
  if (trace.answer) {
    const outcome = element('div', 'outcome')
    outcome.append(cards.get('answer') as HTMLElement, cards.get('verdict') as HTMLElement)
    parts.push(outcome)
  }
  pinned = null
  const root = byId<HTMLElement>('trace')
  root.replaceChildren(...parts)
  // An excerpt short enough to show whole needs no button to show it.
  for (const text of root.querySelectorAll<HTMLElement>('.text.clamped')) {
    if (text.scrollHeight <= text.clientHeight + 1) {
      text.classList.remove('clamped')
      text.nextElementSibling?.remove()
    }
  }
}

// Slow motion: the trace as it came, one stage per turn, its steps filling the timeline as they play, the page
// following the stage that plays. Nothing is made up between stages; with reduced motion set the stages still take
// their turns, and nothing moves or scrolls smoothly.
function play(slow: boolean) {
  for (const timer of timers) window.clearTimeout(timer)
  timers = []
  const root = byId<HTMLElement>('trace')
  const state = byId<HTMLElement>('state')
  const cards = [...root.querySelectorAll<HTMLElement>('.card')]
  const pieces = [...root.querySelectorAll<HTMLElement>('.piece')]
  root.classList.toggle('slow', slow)
  for (const node of [...cards, ...pieces]) node.classList.toggle('waiting', slow)
  for (const node of cards) node.classList.remove('now')
  state.textContent = summary
  if (!slow) return
  cards.forEach((node, index) => timers.push(window.setTimeout(() => {
    for (const other of cards) other.classList.remove('now')
    node.classList.remove('waiting')
    node.classList.add('now')
    node.scrollIntoView({ block: 'nearest', behavior: still.matches ? 'auto' : 'smooth' })
    const stage = STAGES.find((each) => each.key === node.dataset.stage) as Stage
    for (const piece of pieces) if (stage.timed.includes(piece.dataset.step ?? '')) piece.classList.remove('waiting')
    state.textContent = `Stage ${index + 1} of ${cards.length}: ${stage.title}.`
  }, index * TURN)))
  timers.push(window.setTimeout(() => {
    for (const node of cards) node.classList.remove('now')
    for (const piece of pieces) piece.classList.remove('waiting')
    state.textContent = summary
  }, cards.length * TURN))
}

// A unit pointed at, focused or pinned is marked in every stage that holds it and in the excerpt it joined.
function mark(unit: string | null) {
  const root = byId<HTMLElement>('trace')
  for (const node of root.querySelectorAll('.same')) node.classList.remove('same')
  if (!unit) return
  const id = CSS.escape(unit)
  for (const node of root.querySelectorAll(`[data-unit="${id}"], [data-units~="${id}"]`)) node.classList.add('same')
}

function link(root: HTMLElement) {
  const unitOf = (target: EventTarget | null) =>
    target instanceof Element ? (target.closest<HTMLElement>('[data-unit]')?.dataset.unit ?? null) : null
  const enter = (event: Event) => {
    const unit = unitOf(event.target)
    if (unit) mark(unit)
  }
  const leave = (event: FocusEvent | PointerEvent) => {
    if (!unitOf(event.relatedTarget)) mark(pinned)
  }
  root.addEventListener('pointerover', enter)
  root.addEventListener('focusin', enter)
  root.addEventListener('pointerout', leave)
  root.addEventListener('focusout', leave)
  root.addEventListener('click', (event) => {
    const unit = unitOf(event.target)
    if (!unit) return
    pinned = pinned === unit ? null : unit
    for (const row of root.querySelectorAll<HTMLElement>('.hit')) {
      row.setAttribute('aria-pressed', String(row.dataset.unit === pinned))
    }
    mark(pinned ?? unit)
  })
}

async function run(event: SubmitEvent) {
  event.preventDefault()
  const asking = (event.submitter as HTMLButtonElement | null)?.value === 'ask'
  const role = byId<HTMLSelectElement>('role').value
  const question = byId<HTMLInputElement>('question').value
  const buttons = [...byId<HTMLFormElement>('query').querySelectorAll('button')]
  const state = byId<HTMLElement>('state')
  for (const button of buttons) button.disabled = true
  state.classList.remove('error')
  state.textContent = asking ? `Asking as ${role}: the search, then the model, which takes seconds.` : `Searching as ${role}.`
  try {
    const trace = await call<Trace>(asking ? 'ask' : 'search', { method: 'POST', body: JSON.stringify({ question }) }, role)
    summary = `${asking ? 'Asked' : 'Searched'} as ${trace.role}.`
    draw(trace)
    play(byId<HTMLInputElement>('slow').checked)
  } catch (error) {
    state.classList.add('error')
    state.textContent = (error as Error).message
  } finally {
    for (const button of buttons) button.disabled = false
  }
}

// What the change benches and `ntk stats` printed, as bench/page_data.py reads them from the records the README's
// claims are held to: the page replays them, with the command and the time each record names.
type Step = { kind: 'scan' | 'sync'; at: number; took: number; parts: Record<string, number>; detail: string }
type Run = { run: number; change: string; seen: number; events: Step[] }
type Change = {
  bench: string
  command: string
  recorded: string
  doc_id: string
  size: string | null
  reader: string
  catch_up: number
  objective: number
  runs: Run[]
  said: string[]
}
type Figures = {
  documents: number
  bytes: number
  units: number
  layers: Record<string, number>
  vectors: number
  collection: string
  disk: { index: number; manifest: number; vector_cache: number }
  command?: string
  recorded?: string
}

const TITLES: Record<string, (change: Change) => string> = {
  freshness: (c) => `One sentence of ${c.doc_id}, ${c.size}, edited and undone, asked as ${c.reader}`,
  revocation: (c) => `A section of ${c.doc_id} widened to ${c.reader} and narrowed back, asked as ${c.reader}`,
}

async function recorded<T>(name: string): Promise<T | null> {
  try {
    const response = await fetch(`recorded/${name}`)
    return response.ok ? ((await response.json()) as T) : null
  } catch {
    return null
  }
}

function drawFigures(figures: Figures, origin: string) {
  const tiles: [string, string, string][] = [
    ['Documents', number(figures.documents), `${number(figures.bytes)} bytes`],
    ['Units', number(figures.units), Object.entries(figures.layers).map(([layer, n]) => `${layer} ${number(n)}`).join(', ')],
    ['Vectors', number(figures.vectors), `in ${figures.collection}`],
    ['The index on disk', number(figures.disk.index), 'bytes'],
    ['The manifest', number(figures.disk.manifest), 'bytes'],
    ['The vector cache', number(figures.disk.vector_cache), 'bytes'],
  ]
  byId<HTMLElement>('figures').replaceChildren(...tiles.map(([label, value, under]) => {
    const tile = element('div', 'tile')
    tile.append(element('dt', '', label), element('dd', 'value', value), element('dd', 'under', under))
    return tile
  }))
  byId<HTMLElement>('scale-origin').textContent = origin
  byId<HTMLElement>('scale').hidden = false
}

// One lane per run, each scan and sync drawn where it began and as long as it took, the moment a query first saw the
// change marked, and the objective as a hairline. Play runs the lanes at the speed they were recorded.
function drawChange(change: Change): HTMLElement {
  const block = element('article', 'change')
  const title = (TITLES[change.bench] ?? ((c: Change) => c.doc_id))(change)
  block.append(element('h3', '', title), element('p', 'origin', `${change.recorded}: ${change.command}`))
  const legend = element('ul', 'legend')
  for (const [className, text] of [['scan', 'the watcher’s scan'], ['sync', 'its sync into the index'],
    ['seen', 'a query first sees the change'], ['goal', `the ${change.objective} s objective`]]) {
    const item = element('li')
    item.append(element('span', `key ${className}`), document.createTextNode(text))
    legend.append(item)
  }
  const domain = Math.ceil(Math.max(change.objective, ...change.runs.map((run) => run.seen)) + 0.5)
  const at = (seconds: number) => `${(seconds / domain) * 100}%`
  const plot = element('div', 'plot')
  plot.setAttribute('aria-hidden', 'true') // the table below holds every value for a reader who cannot see it
  const axis = element('div', 'lane axis')
  const ticks = element('div', 'track')
  for (let second = 0; second <= domain; second++) {
    const tick = element('span', 'tick', String(second))
    tick.style.left = at(second)
    ticks.append(tick)
  }
  axis.append(element('span', 'name', 'seconds'), ticks)
  plot.append(axis)
  const bars: { node: HTMLElement; event: Step }[] = []
  const dots: { node: HTMLElement; seen: number }[] = []
  for (const run of change.runs) {
    const lane = element('div', 'lane')
    const track = element('div', 'track')
    const goal = element('span', 'goal')
    goal.style.left = at(change.objective)
    track.append(goal)
    for (const event of run.events) {
      const bar = element('span', `bar ${event.kind}`)
      bar.style.left = at(event.at)
      bar.style.width = at(event.took)
      bar.title = `${event.kind} at ${secs(event.at)}, took ${secs(event.took)} (${event.detail})`
      track.append(bar)
      bars.push({ node: bar, event })
    }
    const dot = element('span', 'dot')
    dot.style.left = at(run.seen)
    dot.title = `seen at ${secs(run.seen)}`
    track.append(dot)
    dots.push({ node: dot, seen: run.seen })
    lane.append(element('span', 'name', `run ${run.run}, ${run.change}`), track)
    plot.append(lane)
  }
  const cursor = element('span', 'cursor')
  cursor.hidden = true
  plot.append(cursor)
  const play = element('button', 'second play', 'Play at the recorded speed') as HTMLButtonElement
  play.type = 'button'
  play.hidden = still.matches // with reduced motion nothing moves: the lanes already show where every run ended
  play.addEventListener('click', () => {
    const began = performance.now()
    play.disabled = true
    cursor.hidden = false
    const frame = (now: number) => {
      const t = (now - began) / 1000
      for (const { node, event } of bars) node.style.transform = `scaleX(${Math.min(1, Math.max(0, (t - event.at) / event.took))})`
      for (const { node, seen } of dots) node.classList.toggle('waiting', t < seen)
      cursor.style.left = `calc(var(--name) + (100% - var(--name)) * ${Math.min(t, domain) / domain})`
      if (t < domain) {
        requestAnimationFrame(frame)
      } else {
        cursor.hidden = true
        play.disabled = false
      }
    }
    requestAnimationFrame(frame)
  })
  block.append(legend, plot, play, ...change.said.map((line) => element('p', 'said', line)), table(change))
  return block
}

function table(change: Change): HTMLElement {
  const details = element('details', 'runs')
  details.append(element('summary', '', 'The runs as a table'))
  const grid = element('table')
  const head = element('tr')
  for (const name of ['Run', 'Change', 'Scans and syncs, from the change', 'Seen']) head.append(element('th', '', name))
  grid.append(head)
  for (const run of change.runs) {
    const row = element('tr')
    const steps = run.events.map((e) => `${e.kind} at ${secs(e.at)}, took ${secs(e.took)} (${e.detail})`).join('; ')
    row.append(element('td', '', String(run.run)), element('td', '', run.change), element('td', '', steps || 'none'),
      element('td', '', secs(run.seen)))
    grid.append(row)
  }
  details.append(grid)
  return details
}

async function start() {
  const mode = byId<HTMLParagraphElement>('mode')
  const [roles, changes, figures] = await Promise.all([served(), recorded<{ changes: Change[] }>('changes.json'),
    recorded<Figures>('scale.json')])
  if (changes) {
    byId<HTMLElement>('change-charts').replaceChildren(...changes.changes.map(drawChange))
    byId<HTMLElement>('changes').hidden = false
  }
  if (figures) drawFigures(figures, `${figures.recorded}: ${figures.command}`)
  if (!roles) {
    mode.textContent = 'No local server answers here: start one with ntk serve and open the page it serves.'
    return
  }
  // Live, the build's figures come from the server, the same for every role, through the first role's session.
  call<Figures>('stats', {}, roles[0]).then((live) => drawFigures(live, 'live, from the local server')).catch(() => {})
  const picker = byId<HTMLSelectElement>('role')
  picker.replaceChildren(...roles.map((role) => {
    const option = document.createElement('option')
    option.value = option.textContent = role
    return option
  }))
  mode.textContent = 'Live: the local server, each question through the chosen role’s session.'
  byId<HTMLElement>('instrument').hidden = false
  byId<HTMLFormElement>('query').addEventListener('submit', run)
  byId<HTMLInputElement>('slow').addEventListener('change', (event) => {
    if (byId<HTMLElement>('trace').childElementCount) play((event.target as HTMLInputElement).checked)
  })
  link(byId<HTMLElement>('trace'))
}

void start()
