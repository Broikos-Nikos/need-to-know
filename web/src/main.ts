// The instrument page (T6.2): a role's question through the local server, each stage of its search drawn from the
// trace the server returns. Every unit the trace names passed the role's filter on the server; the page adds nothing
// to it and writes text only as text, never as markup.

import './style.css'

type Hit = { unit_id: string; rank: number; score: number }
type Unit = { doc_id: string; section: string; labels: string[]; text: string }
type Excerpt = { doc_id: string; section: string; labels: string[]; units: string[]; rank: number; text: string }
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
}

const STAGES = [
  ['dense', 'Dense', 'cosine'],
  ['sparse', 'Lexical', 'BM25 over the role’s view'],
  ['fused', 'Fused', 'reciprocal rank'],
  ['reranked', 'Reranked', 'cross encoder'],
] as const
const SHOWN = 10
const tokens = new Map<string, string>()

const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const number = (n: number) => n.toLocaleString('en')

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

function drawFilter(trace: Trace) {
  const line = byId<HTMLParagraphElement>('filter')
  const { labels, passes, holds_back } = trace.filter
  line.textContent = `${trace.role} reads ${labels.join(', ')}: ${number(passes)} units pass its filter, ` +
    `${number(holds_back)} are held back, and only the count of those leaves the server.`
  line.hidden = false
}

function drawStages(trace: Trace) {
  const stages = byId<HTMLElement>('stages')
  stages.replaceChildren()
  for (const [key, title, measure] of STAGES) {
    const column = element('div', 'stage')
    const head = element('h2', '', title)
    head.append(element('span', 'measure', measure))
    column.append(head)
    const list = element('ol', 'hits')
    for (const hit of trace[key].slice(0, SHOWN)) {
      const unit = trace.units[hit.unit_id]
      const item = element('li', 'hit')
      item.append(element('span', 'rank', String(hit.rank)), element('span', 'doc', unit?.doc_id ?? '?'),
        element('span', 'score', hit.score.toFixed(key === 'fused' ? 4 : 3)))
      item.title = unit ? unit.text : ''
      list.append(item)
    }
    column.append(list, element('p', 'count', `${trace[key].length} candidates`))
    stages.append(column)
  }
}

function drawExcerpts(trace: Trace) {
  const excerpts = byId<HTMLElement>('excerpts')
  excerpts.replaceChildren(element('h2', '', 'The excerpts a question would be shown'))
  for (const excerpt of trace.excerpts) {
    const block = element('article', 'excerpt')
    block.append(element('p', 'where', `${excerpt.doc_id}, ${excerpt.section || '(no heading)'}`),
      element('p', 'labels', excerpt.labels.join(' + ')), element('p', 'text', excerpt.text))
    excerpts.append(block)
  }
}

async function search(event: SubmitEvent) {
  event.preventDefault()
  const role = byId<HTMLSelectElement>('role').value
  const question = byId<HTMLInputElement>('question').value
  const go = byId<HTMLButtonElement>('go')
  go.disabled = true
  try {
    const trace = await call<Trace>('search', { method: 'POST', body: JSON.stringify({ question }) }, role)
    drawFilter(trace)
    drawStages(trace)
    drawExcerpts(trace)
  } catch (error) {
    byId<HTMLElement>('stages').replaceChildren(element('p', 'error', String((error as Error).message)))
  } finally {
    go.disabled = false
  }
}

async function start() {
  const mode = byId<HTMLParagraphElement>('mode')
  const roles = await served()
  if (!roles) {
    mode.textContent = 'No local server answers here: start one with ntk serve and open the page it serves.'
    return
  }
  const picker = byId<HTMLSelectElement>('role')
  picker.replaceChildren(...roles.map((role) => {
    const option = document.createElement('option')
    option.value = option.textContent = role
    return option
  }))
  mode.textContent = 'Live: the local server, each search through the chosen role’s session.'
  byId<HTMLElement>('instrument').hidden = false
  byId<HTMLFormElement>('ask').addEventListener('submit', search)
}

void start()
