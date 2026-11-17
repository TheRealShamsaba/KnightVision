// One plain menu for every page: four tabs that say what they are, always visible, the current one marked.
// A bar at the top on wide screens, a tab bar along the bottom on phones. Pages restyle it through the
// .kv-nav variables in their own CSS.
import './nav.css'

const TABS = [
  { href: '/', label: 'Story', title: 'The whole story, told through the mail' },
  { href: '/flight.html', label: '3D Desk', title: 'A working model of the network, in 3D' },
  { href: '/prototypes/notebook.html', label: 'Notebook', title: 'How it works, in eight figures you can play with' },
  { href: '/#play', label: 'Play', title: 'Play a game against the network' },
]

export function mountNav(opts: { current: string; className?: string }): HTMLElement {
  const nav = document.createElement('nav')
  nav.className = `kv-nav ${opts.className ?? ''}`
  nav.setAttribute('aria-label', 'KnightVision')
  nav.innerHTML = `<a class="kv-brand" href="/">KnightVision</a><ul>${TABS.map((t) =>
    `<li><a href="${t.href}" title="${t.title}"${t.href === opts.current ? ' aria-current="page"' : ''}${t.label === 'Play' ? ' class="kv-play"' : ''}>${t.label}</a></li>`).join('')}</ul>`
  document.body.append(nav)
  // On the story page, Play is just further down, so scroll instead of reloading.
  if (opts.current === '/') {
    nav.querySelector('.kv-play')!.addEventListener('click', (e) => {
      e.preventDefault()
      document.getElementById('play')?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })
    })
  }
  return nav
}
