import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Globe from './components/Globe'
import RiskCard from './components/RiskCard'
import TradePanel from './components/TradePanel'
import SignalFeed from './components/SignalFeed'
import CountrySelector from './components/CountrySelector'
import ShowcaseCard from './components/ShowcaseCard'
import { api } from './lib/api'

const SHOWCASE_COUNTRIES = ['TW', 'RU', 'IR', 'KE', 'FR', 'NG', 'BR', 'US']

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
  const menuOpenRef = useRef(false)
  const globeApiRef = useRef(null)
  const [error,           setError]           = useState(null)

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

  const handleSelect = useCallback(async (iso2) => {
    if (iso2 === selected) return
    setSelected(iso2)
    setSignals([])

    if (!riskData[iso2]) {
      setRiskLoading(true)
      api.risk(iso2)
        .then(snap => setRiskData(d => ({ ...d, [iso2]: snap })))
        .catch(() => {})
        .finally(() => setRiskLoading(false))
    }
    if (!tradeData[iso2]) {
      api.trade(iso2).then(data => setTradeData(d => ({ ...d, [iso2]: data }))).catch(() => {})
    }

    setSignalsLoading(true)
    api.signals(iso2).then(setSignals).catch(() => {}).finally(() => setSignalsLoading(false))
  }, [selected, riskData, tradeData])

  const handleDeselect = useCallback(() => setSelected(null), [])

  const selectedCountry = countries.find(c => c.iso2 === selected)
  const snapshot        = selected ? riskData[selected] : null

  return (
    <div className="relative w-screen h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>

      {/* Globe — full screen */}
      <div className="absolute inset-0">
        <Globe
          ref={globeApiRef}
          riskData={riskData}
          tradeData={tradeData}
          countries={countries}
          selectedIso2={selected}
          onCountryClick={handleSelect}
          onGlobeClick={handleDeselect}
          onShowcaseChange={handleShowcaseChange}
          paused={!!selected || menuOpen}
        />
      </div>

      {/* Showcase card — appears during auto-rotation */}
      <AnimatePresence>
        {!selected && (
          <ShowcaseCard
            showcaseIso2={showcaseIso2}
            countries={countries}
            riskData={riskData}
            tradeData={tradeData}
            globeApiRef={globeApiRef}
          />
        )}
      </AnimatePresence>

      {/* ── Header ── */}
      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="absolute top-0 left-0 right-0 flex items-start justify-between pointer-events-none"
        style={{ padding: '2rem 3rem', zIndex: 30 }}
      >
        {/* Logo */}
        <div>
          <h1
            className="font-logo font-bold leading-none tracking-tight"
            style={{ fontSize: 'clamp(2.2rem, 4.5vw, 3.5rem)', color: 'white', letterSpacing: '-0.01em' }}
          >
            exo
          </h1>
          <p
            className="font-display tracking-[0.3em] uppercase mt-2"
            style={{ fontSize: '0.72rem', color: 'rgba(0,230,118,0.55)', letterSpacing: '0.28em' }}
          >
            Geopolitical Intelligence
          </p>
        </div>

      </motion.div>

      {/* ── Country selector — bottom-left ── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
        className="absolute"
        style={{ bottom: '2.5rem', left: '3rem', width: '22rem', zIndex: 30 }}
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
              className="grid gap-5 w-full pointer-events-auto relative"
              style={{
                gridTemplateColumns: '1fr 1fr',
                maxWidth: '1120px',
                padding: '0 2.5rem',
                zIndex: 1,
              }}
              onClick={e => e.stopPropagation()}
            >
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
                    border: '1px solid #ffffff',
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
                  border: '1px solid #ffffff',
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
                  border: '1px solid #ffffff',
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
            className="absolute top-24 left-1/2 -translate-x-1/2 font-mono text-xs px-4 py-2 rounded"
            style={{ background: 'rgba(255,59,92,0.12)', border: '1px solid rgba(255,59,92,0.3)', color: '#ff3b5c' }}
          >
            API unavailable — {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
