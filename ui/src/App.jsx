import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Globe from './components/Globe'
import RiskCard from './components/RiskCard'
import TradePanel from './components/TradePanel'
import SignalFeed from './components/SignalFeed'
import CountrySelector from './components/CountrySelector'
import ShowcaseCard from './components/ShowcaseCard'
import { api } from './lib/api'
import { useIsMobile } from './hooks/useIsMobile'
import { NIGHT_SKY_URL } from './lib/sky'

const SHOWCASE_COUNTRIES = ['TW', 'RU', 'IR', 'KE', 'FR', 'NG', 'BR', 'US']

// Intro: the logo fades in centred, holds, then glides to the top-left; the
// rest of the page fades in after it lands. Timings in ms.
const LOGO_FADE_MS = 600
const LOGO_HOLD_MS = 400
const LOGO_MOVE_MS = 900
const FONT_WAIT_MS = 2500
// Screen widths the star field scrolls per full turn of the globe.
const SKY_SCREENS_PER_TURN = 2
const LOGO_MOVE = { layout: { duration: LOGO_MOVE_MS / 1000, ease: [0.65, 0, 0.35, 1] } }

function prefersReducedMotion() {
  return typeof window !== 'undefined'
    && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

// Resolves once the logo fonts and the star field are usable (or after
// FONT_WAIT_MS), so the centred logo never fades in as a fallback font and
// reflows mid-animation, and the sky doesn't pop in behind it.
// fonts.load() resolves empty until the Google Fonts stylesheet has declared
// the face, so keep asking until it actually returns one.
function logoFontsReady() {
  const sky = new Promise(r => {
    const img = new Image()
    img.onload = img.onerror = r
    img.src = NIGHT_SKY_URL
  })
  if (!document.fonts?.load) return Promise.race([sky, new Promise(r => setTimeout(r, FONT_WAIT_MS))])
  const deadline = Date.now() + FONT_WAIT_MS
  const want = async (font) => {
    while ((await document.fonts.load(font)).length === 0 && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 100))
    }
  }
  return Promise.race([
    Promise.all([sky, want('700 56px Orbitron'), want('400 12px "Chakra Petch"')]),
    new Promise(r => setTimeout(r, FONT_WAIT_MS)),
  ]).catch(() => {})
}

// Wordmark + tagline. Each part carries a layoutId so that when the intro
// swaps the centred copy for the header copy, framer-motion glides it across.
// w-fit keeps each box the same size in both places, so the move is a pure
// translate with no scale distortion.
function Logo({ centered }) {
  return (
    <div className={`flex flex-col ${centered ? 'items-center' : 'items-start'}`}>
      <motion.h1
        layoutId="logo-word"
        transition={LOGO_MOVE}
        className="font-logo font-bold leading-none tracking-tight w-fit"
        style={{ fontSize: 'clamp(2.2rem, 4.5vw, 3.5rem)', color: 'white', letterSpacing: '-0.01em' }}
      >
        exo
      </motion.h1>
      <motion.p
        layoutId="logo-tagline"
        transition={LOGO_MOVE}
        className="font-display tracking-[0.3em] uppercase mt-2 w-fit"
        style={{ fontSize: '0.72rem', color: '#00E676', letterSpacing: '0.28em'}}
      >
        Geopolitical Intelligence
      </motion.p>
    </div>
  )
}

export default function App() {
  const [countries,       setCountries]       = useState([])
  const [riskData,        setRiskData]        = useState({})
  const [tradeData,       setTradeData]       = useState({})
  const [selected,        setSelected]        = useState(null)
  const [signals,         setSignals]         = useState([])
  const [signalsLoading,  setSignalsLoading]  = useState(false)
  const [riskLoading,     setRiskLoading]     = useState(false)
  const [showcaseIso2,    setShowcaseIso2]    = useState(null)
  const [showcaseType,    setShowcaseType]    = useState(null)
  const [menuOpen,        setMenuOpen]        = useState(false)
  const [hoveredShowcase, setHoveredShowcase] = useState(null)
  const menuOpenRef = useRef(false)
  const globeApiRef = useRef(null)
  const skyRef      = useRef(null)
  const [error,           setError]           = useState(null)
  const isMobile = useIsMobile()
  // 'loading' (waiting on fonts) → 'center' → 'moving' → 'done'
  const [intro, setIntro] = useState(() => prefersReducedMotion() ? 'done' : 'loading')
  const introDone = intro === 'done'
  // On phones the detail cards fill the screen, and the header/selector (z-30)
  // would otherwise paint on top of them. Give the cards the screen instead.
  const hideChrome = isMobile && !!selected

  useEffect(() => {
    if (prefersReducedMotion()) return
    let cancelled = false
    const timers = []
    const at = (ms, phase) => timers.push(setTimeout(() => setIntro(phase), ms))
    logoFontsReady().then(() => {
      if (cancelled) return
      setIntro('center')
      at(LOGO_FADE_MS + LOGO_HOLD_MS, 'moving')
      at(LOGO_FADE_MS + LOGO_HOLD_MS + LOGO_MOVE_MS, 'done')
    })
    return () => { cancelled = true; timers.forEach(clearTimeout) }
  }, [])

  // Pan the star field with the camera so it drifts as the globe turns (the
  // sky is distant, so it moves against the globe's surface). One full turn
  // scrolls SKY_SCREENS_PER_TURN image widths; the strip holds two copies of the panorama, whose
  // edges meet seamlessly, so the wrap is invisible. Transform-only, so each
  // frame is a cheap composite rather than a full-screen repaint.
  useEffect(() => {
    let rafId, lastX = null
    const tick = () => {
      const lng = globeApiRef.current?.getViewLng()
      const el  = skyRef.current
      if (lng != null && el) {
        const w = window.innerWidth
        const x = Math.round(((((lng / 360) * SKY_SCREENS_PER_TURN * w) % w) + w) % w * 10) / 10
        if (x !== lastX) {
          el.style.transform = `translate3d(${x}px, 0, 0)`
          lastX = x
        }
      }
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [])

  // Countries list
  useEffect(() => {
    api.countries().then(setCountries).catch(e => setError(e.message))
  }, [])

  // Pre-fetch showcase data so cards appear immediately
  useEffect(() => {
    if (!countries.length) return
    SHOWCASE_COUNTRIES.forEach(iso2 => {
      api.risk(iso2).then(snap => setRiskData(d => ({ ...d, [iso2]: snap }))).catch(() => {})
      api.trade(iso2).then(data => setTradeData(d => ({ ...d, [iso2]: data }))).catch(() => {})
    })
  }, [countries])

  // Guard: ref is set synchronously so the poll callback sees it immediately,
  // even if the React state update hasn't flushed yet.
  const handleMenuOpenChange = useCallback((val) => {
    menuOpenRef.current = val
    setMenuOpen(val)
  }, [])

  const handleShowcaseChange = useCallback((iso2, type) => {
    if (menuOpenRef.current) return   // pin showcase while menu is open
    setShowcaseIso2(iso2)
    setShowcaseType(type)
  }, [])

  const handleShowcaseHover = useCallback((iso2, hovering) => {
    setHoveredShowcase(h => hovering ? iso2 : (h === iso2 ? null : h))
  }, [])

  const handleSelect = useCallback(async (iso2) => {
    if (iso2 === selected) return
    setSelected(iso2)
    setHoveredShowcase(null)
    setSignals([])

    if (!riskData[iso2]) {
      setRiskLoading(true)
      api.risk(iso2)
        .then(snap => setRiskData(d => ({ ...d, [iso2]: snap })))
        .catch(() => {})
        .finally(() => setRiskLoading(false))
    }
    if (!tradeData[iso2]) {
      // No trade entry (404) still ends loading: the panel renders nothing for an empty list.
      api.trade(iso2)
        .then(data => setTradeData(d => ({ ...d, [iso2]: data })))
        .catch(() => setTradeData(d => ({ ...d, [iso2]: { country: iso2, partners: [] } })))
    }

    setSignalsLoading(true)
    api.signals(iso2).then(setSignals).catch(() => {}).finally(() => setSignalsLoading(false))
  }, [selected, riskData, tradeData])

  const handleDeselect = useCallback(() => setSelected(null), [])

  const selectedCountry = countries.find(c => c.iso2 === selected)
  const snapshot        = selected ? riskData[selected] : null

  return (
    <div className="relative w-screen h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>

      {/* Star field — shows through the transparent globe canvas, so the intro
          and the main page share one sky. */}
      <div className="absolute inset-0 overflow-hidden">
        <div
          ref={skyRef}
          className="absolute top-0 h-full"
          style={{
            left: '-100%',
            width: '200%',
            backgroundImage: `url(${NIGHT_SKY_URL})`,
            backgroundSize: '50% 100%',
            backgroundRepeat: 'repeat-x',
            willChange: 'transform',
          }}
        />
      </div>

      {/* Globe — full screen. Mounted (and loading) behind the intro, shown once it ends. */}
      <motion.div
        className="absolute inset-0"
        initial={false}
        animate={{ opacity: introDone ? 1 : 0 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        style={{ pointerEvents: introDone ? 'auto' : 'none' }}
      >
        <Globe
          ref={globeApiRef}
          riskData={riskData}
          tradeData={tradeData}
          countries={countries}
          selectedIso2={selected}
          onCountryClick={handleSelect}
          onGlobeClick={handleDeselect}
          onShowcaseChange={handleShowcaseChange}
          paused={!!selected || menuOpen || (hoveredShowcase !== null && hoveredShowcase === showcaseIso2)}
        />
      </motion.div>

      {/* Showcase card — appears during auto-rotation (desktop/tablet only; fixed
          384px card + per-frame connector-line math don't fit phone widths) */}
      <motion.div
        initial={false}
        animate={{ opacity: introDone ? 1 : 0 }}
        transition={{ duration: 0.6, delay: introDone ? 0.3 : 0 }}
      >
      <AnimatePresence>
        {introDone && !selected && !isMobile && (
          <ShowcaseCard
            showcaseIso2={showcaseIso2}
            countries={countries}
            riskData={riskData}
            tradeData={tradeData}
            globeApiRef={globeApiRef}
            onSelect={handleSelect}
            onHoverChange={handleShowcaseHover}
          />
        )}
      </AnimatePresence>
      </motion.div>

      {/* ── Intro: logo centred on the bare page ── */}
      {(intro === 'loading' || intro === 'center') && (
        <motion.div
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
          style={{ zIndex: 40 }}
          initial={{ opacity: 0 }}
          animate={{ opacity: intro === 'center' ? 1 : 0 }}
          transition={{ duration: LOGO_FADE_MS / 1000, ease: 'easeOut' }}
        >
          <Logo centered />
        </motion.div>
      )}

      {/* ── Header ── */}
      <motion.div
        initial={false}
        animate={{ opacity: hideChrome ? 0 : 1 }}
        transition={{ duration: 0.3 }}
        className="absolute top-0 left-0 right-0 flex items-start justify-between pointer-events-none py-6 md:py-8"
        style={{ zIndex: 30, paddingInline: 'var(--edge-x)' }}
      >
        {/* Logo — lands here from the centre during the intro */}
        {(intro === 'moving' || introDone) && <Logo />}

      </motion.div>

      {/* ── Country selector — bottom-left ── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={introDone ? { opacity: hideChrome ? 0 : 1, y: 0 } : { opacity: 0, y: 16 }}
        transition={{ duration: 0.6, delay: introDone ? 0.15 : 0, ease: [0.16, 1, 0.3, 1] }}
        className="absolute"
        style={{
          bottom: 'clamp(1.25rem, 4vh, 2.5rem)',
          left: 'var(--edge-x)',
          width: 'min(22rem, calc(100vw - 2 * var(--edge-x)))',
          zIndex: 30,
          pointerEvents: hideChrome || !introDone ? 'none' : 'auto',
        }}
      >
        <CountrySelector
          countries={countries}
          selected={selected}
          onSelect={handleSelect}
          onOpenChange={handleMenuOpenChange}
        />
      </motion.div>

      {/* ── Selected country overlay ── */}
      <AnimatePresence>
        {selected && (
          <motion.div
            key="overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 flex items-center justify-center"
            style={{ zIndex: 20 }}
          >
            {/* Fullscreen click-catcher — clicking outside the cards dismisses */}
            <div className="absolute inset-0" onClick={handleDeselect} style={{ cursor: 'default' }} />
            <motion.div
              initial={{ opacity: 0, y: 30, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.97 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="grid gap-5 w-full pointer-events-auto relative grid-cols-1 md:grid-cols-2 px-4 md:px-10 py-6 md:py-10"
              style={{
                maxWidth: '1120px',
                maxHeight: '100dvh',
                overflowY: 'auto',
                zIndex: 1,
              }}
              onClick={e => e.stopPropagation()}
            >
              {/* Back to globe — explicit dismiss, since tapping outside the
                  cards is unreliable on phones where they fill the screen */}
              <div style={{ gridColumn: '1 / -1' }} className="flex justify-end mb-1 md:mb-2">
                <button
                  onClick={handleDeselect}
                  className="font-display text-[11px] tracking-[0.2em] uppercase rounded px-3 py-2 border-white flex items-center gap-2"
                  style={{
                    background: 'rgba(10,10,10,0.75)',
                    border: '1.5px solid #ffffff',
                    color: '#00E676',
                  }}
                >
                  <span aria-hidden="true">←</span> Back to globe
                </button>
              </div>

              {/* Risk card — spans full width */}
              <div style={{ gridColumn: '1 / -1' }}>
                {riskLoading || !snapshot ? (
                  <div
                    className="rounded-xl p-6 border-white"
                    style={{
                      background: 'rgba(10,10,10,0.75)',
                      border: '1px solid #ffffff',
                    }}
                  >
                    <div className="space-y-3">
                      {[80, 65, 55, 45, 38, 30].map((w, i) => (
                        <div key={i} className="h-3.5 rounded bg-white/5 animate-pulse" style={{ width: `${w}%` }} />
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{
                    background: 'rgba(10,10,10,0.75)',
                    borderRadius: '0.75rem',
                    border: '2.5px solid #ffffff',
                    overflow: 'hidden',
                  }}>
                    <RiskCard snapshot={snapshot} country={selectedCountry} />
                  </div>
                )}
              </div>

              {/* Trade panel — left half */}
              <div
                className="rounded-xl border-white"
                style={{
                  background: 'rgba(10,10,10,0.75)',
                  border: '2.5px solid #ffffff',
                  padding: '1.0rem',
                }}
              >
                <TradePanel tradeData={tradeData[selected]} loading={!tradeData[selected]} />
              </div>

              {/* Signal feed — right half */}
              <div
                className="rounded-xl border-white"
                style={{
                  background: 'rgba(10,10,10,0.75)',
                  border: '2.5px solid #ffffff',
                  padding: '1.0rem',
                }}
              >
                <SignalFeed signals={signals} loading={signalsLoading} />
              </div>
            </motion.div>

          </motion.div>
        )}
      </AnimatePresence>

      {/* Error toast */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="absolute top-24 left-1/2 -translate-x-1/2 font-mono text-xs px-4 py-2 rounded max-w-[calc(100vw-2rem)] text-center"
            style={{ background: 'rgba(255,59,92,0.12)', border: '1px solid rgba(255,59,92,0.3)', color: '#ff3b5c' }}
          >
            API unavailable — {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
