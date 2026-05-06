import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { riskColor, riskLabel } from '../lib/riskColor'
import { SHOWCASE } from '../data/showcaseData'

const CARD_WIDTH_PX = 384  // 24rem at 16px base
const SIDE_OFFSET_PX = 40  // 2.5rem at 16px base

// Updates SVG path and circle imperatively on every animation frame —
// no React state so the line tracks the globe dot at 60fps with zero re-renders.
function useGlobeLine(showcaseIso2, entry, country, globeApiRef, pathRef, circleRef) {
  useEffect(() => {
    if (!showcaseIso2 || !entry || !country) {
      if (pathRef.current)   pathRef.current.style.opacity   = '0'
      if (circleRef.current) circleRef.current.style.opacity = '0'
      return
    }

    let rafId
    const tick = () => {
      const screenPos = globeApiRef?.current?.getCountryScreenPos(country.lat, country.lon)
      if (screenPos && pathRef.current && circleRef.current) {
        const W = window.innerWidth
        const H = window.innerHeight
        const cardCenterY = H * entry.position.topPct
        const cardX = entry.position.side === 'right'
          ? W - SIDE_OFFSET_PX - CARD_WIDTH_PX
          : SIDE_OFFSET_PX + CARD_WIDTH_PX
        const { x: cX, y: cY } = screenPos
        const mx = (cardX + cX) / 2
        pathRef.current.setAttribute('d',
        // `M ${cardX} ${cardCenterY} L ${mx} ${cardCenterY} L ${mx} ${cY} L ${cX} ${cY}`)                          
        `M ${cardX} ${cardCenterY} L ${mx} ${cY} L ${cX} ${cY}`)
        pathRef.current.style.opacity   = '1'
        circleRef.current.setAttribute('cx', cX)
        circleRef.current.setAttribute('cy', cY)
        circleRef.current.style.opacity = '0.8'
      } else {
        if (pathRef.current)   pathRef.current.style.opacity   = '0'
        if (circleRef.current) circleRef.current.style.opacity = '0'
      }
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [showcaseIso2, entry, country, globeApiRef, pathRef, circleRef])
}

// Transform live API snapshot → showcase risk shape
function liveRisk(snap) {
  if (!snap) return null
  const dims = [...snap.dimensions]
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map(d => ({ name: d.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()), score: d.score }))
  return {
    composite:  snap.composite_score,
    structural: snap.structural_score,
    shortTerm:  snap.short_term_score,
    acute:      snap.acute_score,
    dims,
  }
}

// Transform live API trade data → showcase trade shape
function liveTrade(data) {
  if (!data?.partners?.length) return null
  return {
    partners: data.partners.map(p => ({ name: p.name, share: p.share_pct, goods: p.goods ?? [] })),
  }
}

export default function ShowcaseCard({ showcaseIso2, countries, riskData, tradeData, globeApiRef }) {
  const entry   = SHOWCASE.find(s => s.iso2 === showcaseIso2)
  const country = countries?.find(c => c.iso2 === showcaseIso2)
  const type    = entry?.type

  // Refs for imperative SVG line updates (no React state = no re-renders per frame)
  const pathRef   = useRef(null)
  const circleRef = useRef(null)
  useGlobeLine(showcaseIso2, entry, country, globeApiRef, pathRef, circleRef)

  // Prefer live API data; fall back to static while pre-fetch is in flight
  const risk  = liveRisk(riskData?.[showcaseIso2])  ?? entry?.risk
  const trade = liveTrade(tradeData?.[showcaseIso2]) ?? entry?.trade

  const pos    = entry?.position
  const slideX = pos?.side === 'right' ? 24 : -24

  const wrapStyle = pos ? {
    [pos.side]: '2.5rem',
    top: `${pos.topPct * 100}vh`,
    width: '24rem',
    zIndex: 10,
  } : {}

  return (
    <>
      {/* SVG connecting line — always mounted, updated imperatively each frame */}
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 9 }}>
        <path
          ref={pathRef}
          fill="none"
          stroke="rgba(0,230,118,0.38)"
          strokeWidth="3"
          style={{ opacity: 0, transition: 'opacity 0.3s' }}
        />
        <circle
          ref={circleRef}
          r="4"
          fill="#00E676"
          style={{ opacity: 0, transition: 'opacity 0.4s 0.2s' }}
        />
      </svg>

      {/* Card */}
      <AnimatePresence>
        {showcaseIso2 && country && entry && (
          <motion.div
            key={showcaseIso2}
            initial={{ opacity: 0, x: slideX }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: slideX }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            transformTemplate={(_, generated) => `translateY(-50%) ${generated}`}
            className="absolute pointer-events-none"
            style={wrapStyle}
          >
            <div
              className="rounded-lg overflow-hidden border-white"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '2.5px solid #ffffff',
              }}
            >
              <div style={{ padding: '1.0rem' }}>
                {/* Country header */}
                <div className="flex items-start justify-between mb-5">
                  <div className="flex items-center gap-3">
                    <img
                      src={`https://flagcdn.com/w80/${country.iso2.toLowerCase()}.png`}
                      alt={country.iso2}
                      style={{ width: 32, height: 'auto', borderRadius: 3, display: 'block', flexShrink: 0 }}
                    />
                    <div>
                      <div className="font-display text-[10px] tracking-[0.2em] text-white mb-1 uppercase">
                        {country.region}
                      </div>
                      <div className="font-display text-lg font-semibold text-white leading-tight">
                        {country.name}
                      </div>
                    </div>
                  </div>

                  {/* Risk: composite score + label in top-right */}
                  {type === 'risk' && risk && (
                    <div className="text-right shrink-0 ml-3">
                      <div className="font-mono text-xl font-medium leading-none" style={{ color: riskColor(risk.composite) }}>
                        {risk.composite.toFixed(2)}
                      </div>
                      <div className="font-display text-[9px] tracking-widest mt-1" style={{ color: riskColor(risk.composite) }}>
                        {riskLabel(risk.composite)}
                      </div>
                    </div>
                  )}
                </div>

                {/* Risk card */}
                {type === 'risk' && risk && (
                  <>
                    <div className="font-display text-xs tracking-[0.2em] text-white mt-6 mb-4">
                      RISK SCORE
                    </div>

                    {/* Tier mini-scores */}
                    <div className="grid grid-cols-3 gap-2 mt-4 mb-1">
                      {[
                        { label: 'STRUCTURAL', score: risk.structural },
                        { label: 'SHORT-TERM', score: risk.shortTerm },
                        { label: 'ACUTE',      score: risk.acute },
                      ].map(({ label, score }) => (
                        <div
                          key={label}
                          className="rounded p-2 text-center"
                          style={{}}
                        >
                          <div className="font-display text-[9px] tracking-wider text-white mb-0.5">{label}</div>
                          <div className="font-mono text-sm" style={{ color: riskColor(score) }}>
                            {score.toFixed(2)}
                          </div>
                        </div>
                      ))}
                    </div>

                  </>
                )}

                {/* Trade card */}
                {type === 'trade' && trade && (
                  <>
                    <div className="font-display text-xs tracking-[0.2em] text-white mt-6 mb-4">
                      TOP TRADE PARTNERS
                    </div>
                    <div className="space-y-3">
                      {trade.partners.map((p, i) => (
                        <motion.div
                          key={p.name}
                          initial={{ opacity: 0, x: 8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.1 + i * 0.07 }}
                          className="flex items-center justify-between gap-2"
                        >
                          <span className="font-mono text-sm text-white truncate flex-1">{p.name}</span>
                          <div className="flex items-center gap-2 shrink-0">
                            <span className="font-mono text-xs text-white">{p.share}%</span>
                            <div className="w-14 h-[3px] rounded-full overflow-hidden bg-white/5">
                              <motion.div
                                className="h-full rounded-full"
                                style={{ background: 'rgba(0,230,118,0.5)' }}
                                initial={{ width: 0 }}
                                animate={{ width: `${(p.share / trade.partners[0].share) * 100}%` }}
                                transition={{ duration: 0.6, delay: 0.15 + i * 0.07 }}
                              />
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Connecting dot on card edge facing globe */}
            <div
              className="absolute top-1/2 w-1.5 h-1.5 rounded-full"
              style={{
                ...(pos.side === 'right'
                  ? { left: 0, transform: 'translate(-50%, -50%)' }
                  : { right: 0, transform: 'translate(50%, -50%)' }),
                background: '#00E676',
                boxShadow: '0 0 6px #00E676',
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
