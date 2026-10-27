// Op-Art Notebook: an essay with eight hand-built figures, all drawn from the network's real outputs.
import './notebook.css'
import { data } from '../../data'
import { hero } from './hero'
import { sheets } from './sheets'
import { quilt } from './quilt'
import { mirror } from './mirror'
import { convFig, eloFig, lossFig, reachFig } from './figs'

const games = data.elo?.games ?? []
const facts: Record<string, string> = {
  params: `${(data.parameters / 1e6).toFixed(1)} million`,
  elo: String(data.elo?.rating ?? 935),
  margin: String(data.elo?.margin ?? 99),
  games: String(games.length || 160),
  wins: String(games.filter((g) => g.score === 1).length),
  draws: String(games.filter((g) => g.score === 0.5).length),
  batch: data.learning.training.batch.toLocaleString('en-GB'),
  epochs: String(data.checkpoint.epoch),
}
document.querySelectorAll<HTMLElement>('[data-fact]').forEach(el => { el.textContent = facts[el.dataset.fact ?? ''] ?? '' })

const byId = (id: string): HTMLElement => document.getElementById(id)!
hero(byId('fig-hero'))
lossFig(byId('fig-loss'))
sheets(byId('fig-sheets'))
convFig(byId('fig-conv'))
reachFig(byId('fig-reach'))
quilt(byId('fig-quilt'))
eloFig(byId('fig-elo'))
mirror(byId('fig-mirror'), byId('mirror-table'))

// Figures are numbered in the order they appear, whatever number their module was written with.
document.querySelectorAll<HTMLElement>('.fig-no').forEach((b, i) => { b.textContent = `Figure ${i + 1}. ` })

// Margin notes sit beside the paragraph before them on wide screens; on narrow ones they stay inline.
const wideQuery = matchMedia('(min-width: 1180px)')
function placeNotes(): void {
  document.querySelectorAll<HTMLElement>('aside.side').forEach((note) => {
    const anchor = note.previousElementSibling as HTMLElement | null
    const wide = wideQuery.matches && anchor
    note.style.top = wide ? `${anchor.offsetTop}px` : ''
    note.style.left = wide ? `${anchor.offsetLeft + anchor.offsetWidth + 36}px` : ''
  })
}
addEventListener('resize', placeNotes)
document.fonts?.ready.then(placeNotes)
placeNotes()
