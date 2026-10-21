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
  counts?: Record<string, number> // a recorded trace keeps each stage's first rows and how many it held
}
// What bench/page_traces.py recorded: each question asked as each role through the app, with when, at which commit
// and with which models, which the page shows on its face when it replays them (house rule 10).
type Recorded = {
  recorded: string
  commit: string
  clean: boolean
  embedder: string
  reranker: string
  models: string[]
  roles: string[]
  questions: { id: string; role: string; lang: string; question: string }[]
  traces: Record<string, Record<string, Trace>>
}
type Feed = { events: { at: string; documents: { doc_id: string; units: number }[] }[] }
type Search = 'dense' | 'sparse' | 'fused' | 'reranked'
type Stage = { key: string; title: string; says: string; timed: string[] }

// The stages in the order the server runs them, and the name of each step it times: one map, which the test suite
// holds to the steps a trace's seconds hold (tests/test_api.py).
const STAGES: Stage[] = plan.stages
const STEPS: Record<string, string> = plan.seconds
const SEARCHES: Search[] = ['dense', 'sparse', 'fused', 'reranked']
const SHOWN = 10
const TURN = 1200 // each stage's turn in slow motion, in milliseconds
const FEED = 5000 // how often the page asks the change feed again, in milliseconds, while it is visible (N66)
const LOOPBACK = ['127.0.0.1', 'localhost']
const tokens = new Map<string, string>()
const still = window.matchMedia('(prefers-reduced-motion: reduce)')
let timers: number[] = []
let pinned: string | null = null
let summary = ''
let recordings: Recorded | null = null
let feedTimer = 0
let feedAfter = ''
let feedRole = ''

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

// The roles answer without a session where the server runs; a static host has no such route (N67). The server answers
// only requests addressed to the loopback's names, so a page served anywhere else asks nothing and replays.
async function served(): Promise<string[] | null> {
  if (!LOOPBACK.includes(location.hostname)) return null
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
  const held = trace.counts?.[key] ?? all.length
  const count = held > SHOWN ? `${number(held)} in all, the first ${SHOWN} shown.` : `${number(held)} in all.`
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

// The same question as every role, a card each: what the role reads, what its search put first, what it was told,
// and the verdict where the question is the role's own; a card opens its role's whole trace below the cards.
function roleCard(role: string, trace: Trace): HTMLElement {
  const card = element('article', 'role-card')
  const { labels, passes } = trace.filter
  card.append(element('h3', '', role), element('p', 'minor', `reads ${labels.join(', ')}; ${number(passes)} units pass`))
  const first = trace.reranked.slice(0, 3).map((hit) => trace.units[hit.unit_id]?.doc_id ?? '?')
  card.append(element('p', 'firsts', first.length ? first.join(', ') : 'nothing found'))
  const told = trace.answer
  if (told) {
    const text = told.status === 'invalid' ? `The reply could not be read: ${told.error ?? ''}` : told.answer
    card.append(element('p', `said clamped${told.status === 'not_found' ? ' refused' : ''}`, text))
  }
  const verdict = trace.verdict
  if (verdict && !('error' in verdict)) {
    const how = verdict.refusal ? 'a refusal' : verdict.grounded ? 'grounded' : 'not grounded'
    card.append(element('p', 'tag', `${verdict.item}, its own role: ${verdict.correct ? 'correct' : 'not correct'}, ${how}`))
  }
  const open = element('button', 'more', 'Its stages') as HTMLButtonElement
  open.type = 'button'
  open.addEventListener('click', () => {
    summary = `${role}’s stages, below the cards.`
    draw(trace)
    play(byId<HTMLInputElement>('slow').checked)
    byId<HTMLElement>('trace').scrollIntoView({ block: 'start', behavior: still.matches ? 'auto' : 'smooth' })
  })
  card.append(open)
  return card
}

// The cards replace a single trace, and a single trace the cards.
function cards(): HTMLElement {
  const grid = byId<HTMLElement>('compare')
  grid.replaceChildren()
  grid.hidden = false
  byId<HTMLElement>('trace').replaceChildren()
  return grid
}

// Replayed, a recorded trace is drawn as a live one is: the same stages, the same slow motion, the same cards.
function replay(event: SubmitEvent) {
  event.preventDefault()
  const by = recordings?.traces[byId<HTMLSelectElement>('pick').value]
  if (!recordings || !by) return
  if (byId<HTMLInputElement>('every').checked) {
    const grid = cards()
    for (const role of recordings.roles) if (by[role]) grid.append(roleCard(role, by[role]))
    summary = 'Replayed as every role, as the app answered each when recorded.'
    byId<HTMLElement>('state').textContent = summary
    return
  }
  byId<HTMLElement>('compare').hidden = true
  const role = byId<HTMLSelectElement>('role').value
  if (!by[role]) return
  summary = `Replayed as ${role}, as the app answered it when recorded.`
  draw(by[role])
  play(byId<HTMLInputElement>('slow').checked)
}

async function run(event: SubmitEvent) {
  event.preventDefault()
  const asking = (event.submitter as HTMLButtonElement | null)?.value === 'ask'
  const every = byId<HTMLInputElement>('every').checked
  const role = byId<HTMLSelectElement>('role').value
  const question = byId<HTMLInputElement>('question').value
  const buttons = [...byId<HTMLFormElement>('query').querySelectorAll('button')]
  const state = byId<HTMLElement>('state')
  const body = { method: 'POST', body: JSON.stringify({ question }) }
  for (const button of buttons) button.disabled = true
  state.classList.remove('error')
  try {
    if (every) {
      const grid = cards()
      const roles = [...byId<HTMLSelectElement>('role').options].map((option) => option.value)
      for (const [index, each] of roles.entries()) {
        state.textContent = `${asking ? 'Asking' : 'Searching'} as ${each}, ${index + 1} of ${roles.length}.`
        grid.append(roleCard(each, await call<Trace>(asking ? 'ask' : 'search', body, each)))
      }
      summary = `${asking ? 'Asked' : 'Searched'} as every role.`
      state.textContent = summary
      return
    }
    byId<HTMLElement>('compare').hidden = true
    state.textContent = asking ? `Asking as ${role}: the search, then the model, which takes seconds.` : `Searching as ${role}.`
    const trace = await call<Trace>(asking ? 'ask' : 'search', body, role)
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
  commit: string
  clean: boolean
  before: number | null
  during: number | null
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
  // When, at which commit and under what load the runs were timed (MA-F20, MA-F22), and the command that timed them.
  const busy = (share: number | null) => (share === null ? 'its load unknown' : `${share} percent busy`)
  const ran = `at ${change.commit}${change.clean ? '' : ', the tree holding changes'}`
  const load = `the processor ${busy(change.before)} before the runs and ${busy(change.during)} over them`
  block.append(element('h3', '', title), element('p', 'origin', `${change.recorded}, ${ran}; ${load}: ${change.command}`))
  const legend = element('ul', 'legend')
  for (const [className, text] of [['scan', 'the watcher’s scan'], ['sync', 'its sync into the index'],
    ['seen', 'a query first sees the change'], ['line', `the ${change.objective} s objective`]]) {
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

// The change feed as the chosen role may see it, asked again every few seconds while the page is visible (N66): each
// finished scan that added units the role may read, newest first. A poll for a role no longer chosen draws nothing.
async function follow(role: string) {
  window.clearTimeout(feedTimer)
  const list = byId<HTMLOListElement>('events')
  if (role !== feedRole) {
    feedRole = role
    feedAfter = ''
    list.replaceChildren()
  }
  try {
    if (!document.hidden) {
      const feed = await call<Feed>(`events?after=${encodeURIComponent(feedAfter)}`, {}, role)
      if (role !== feedRole) return
      if (feed.events.length) list.querySelector('.none')?.remove()
      for (const event of feed.events) {
        // A full build's scan adds to every document: name the first few, and count the rest.
        const where = event.documents.slice(0, 3).map((d) => `${d.doc_id}, ${number(d.units)} unit${d.units === 1 ? '' : 's'}`)
        const rest = event.documents.slice(3)
        if (rest.length) {
          where.push(`and ${number(rest.length)} more documents, ${number(rest.reduce((n, d) => n + d.units, 0))} units`)
        }
        list.prepend(element('li', '', `${event.at.replace('T', ' ')}: ${where.join('; ')}`))
        feedAfter = event.at
      }
      while (list.children.length > SHOWN) list.lastElementChild?.remove()
      if (!list.children.length) list.append(element('li', 'none', 'No scan has added anything this role may read.'))
      byId<HTMLElement>('feed-origin').textContent = `live, as ${role} may see them, asked every ${FEED / 1000} s`
      byId<HTMLElement>('feed').hidden = false
    }
  } catch {
    // the next poll asks again
  }
  if (role === feedRole) feedTimer = window.setTimeout(() => void follow(feedRole), FEED)
}

function fill(select: HTMLSelectElement, entries: [string, string][]) {
  select.replaceChildren(...entries.map(([value, text]) => {
    const option = document.createElement('option')
    option.value = value
    option.textContent = text
    return option
  }))
}

// Where no server answers, the page replays what bench/page_traces.py recorded, with when, at which commit and with
// which models on its face.
function replaying(kept: Recorded) {
  const tree = kept.clean ? '' : ', the tree holding changes'
  byId<HTMLElement>('mode').textContent = `Replayed, as no server answers here: recorded ${kept.recorded} at ${kept.commit}` +
    `${tree}; answers by ${kept.models.join(', ')}, retrieval by ${kept.embedder} and ${kept.reranker}.`
  fill(byId<HTMLSelectElement>('role'), kept.roles.map((role) => [role, role]))
  fill(byId<HTMLSelectElement>('pick'), kept.questions.map((q) => [q.id, `${q.question} (${q.id}, ${q.role})`]))
  byId<HTMLElement>('ask-field').hidden = true
  byId<HTMLInputElement>('question').disabled = true
  byId<HTMLElement>('pick-field').hidden = false
  for (const button of document.querySelectorAll<HTMLElement>('#query .live')) button.hidden = true
  byId<HTMLElement>('replay').hidden = false
  byId<HTMLElement>('note').textContent = 'Each recorded question was asked as every role through the app, and the page ' +
    'replays what it answered each one: pick a role and a question. Slow motion plays the trace one stage at a time; ' +
    'with reduced motion set, nothing moves.'
  byId<HTMLElement>('instrument').hidden = false
  byId<HTMLFormElement>('query').addEventListener('submit', replay)
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
  byId<HTMLInputElement>('slow').addEventListener('change', (event) => {
    if (byId<HTMLElement>('trace').childElementCount) play((event.target as HTMLInputElement).checked)
  })
  link(byId<HTMLElement>('trace'))
  if (!roles) {
    recordings = await recorded<Recorded>('traces.json')
    if (recordings) {
      replaying(recordings)
    } else {
      mode.textContent = 'No local server answers here: start one with ntk serve and open the page it serves.'
    }
    return
  }
  // Live, the build's figures come from the server, the same for every role, through the first role's session.
  call<Figures>('stats', {}, roles[0]).then((live) => drawFigures(live, 'live, from the local server')).catch(() => {})
  const picker = byId<HTMLSelectElement>('role')
  fill(picker, roles.map((role) => [role, role]))
  mode.textContent = 'Live: the local server, each question through the chosen role’s session.'
  byId<HTMLElement>('instrument').hidden = false
  byId<HTMLFormElement>('query').addEventListener('submit', run)
  picker.addEventListener('change', () => void follow(picker.value))
  void follow(picker.value)
}

void start()
