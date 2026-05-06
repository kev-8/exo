import { motion, AnimatePresence } from 'framer-motion'

function timeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime()
  const m = Math.floor(diff / 60000)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

// Per-signal display config: human label + how to format the raw value
const SIGNAL_CONFIG = {
  // GDELT
  news_sentiment:       { label: 'News Sentiment',      fmt: v => v.toFixed(2) + ' / 1.0' },
  // World Bank
  debt_to_gdp:          { label: 'Debt / GDP',           fmt: v => v.toFixed(1) + '%' },
  inflation_rate:       { label: 'Inflation Rate',       fmt: v => v.toFixed(1) + '%' },
  unemployment_rate:    { label: 'Unemployment',         fmt: v => v.toFixed(1) + '%' },
  gdp_growth:           { label: 'GDP Growth',           fmt: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%' },
  energy_imports_pct:   { label: 'Energy Imports',       fmt: v => v.toFixed(1) + '% of use' },
  gross_savings:        { label: 'Gross Savings',        fmt: v => v.toFixed(1) + '% of GNI' },
  trade_openness:       { label: 'Trade Openness',       fmt: v => v.toFixed(1) + '% of GDP' },
  current_account:      { label: 'Current Account',      fmt: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '% of GDP' },
  // FRED
  fred_unrate:          { label: 'Unemployment Rate',    fmt: v => v.toFixed(1) + '%' },
  fred_cpiaucsl:        { label: 'CPI Inflation',        fmt: v => v.toFixed(1) },
  fred_fedfunds:        { label: 'Fed Funds Rate',       fmt: v => v.toFixed(2) + '%' },
  fred_t10y2y:          { label: 'Yield Curve Spread',   fmt: v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%' },
  fred_umcsent:         { label: 'Consumer Sentiment',   fmt: v => v.toFixed(1) },
  economic_indicator:   { label: 'FRED Composite',       fmt: v => v.toFixed(2) + ' / 1.0' },
}

function formatValue(sig) {
  const cfg = SIGNAL_CONFIG[sig.signal_type]
  if (cfg) return cfg.fmt(sig.value)
  // Fallback: if value looks like a 0-1 score, show as score; otherwise 2dp
  return sig.value >= 0 && sig.value <= 1
    ? sig.value.toFixed(2)
    : sig.value.toFixed(2)
}

function labelFor(sig) {
  return SIGNAL_CONFIG[sig.signal_type]?.label
    ?? sig.signal_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// Source pill styles
const SOURCE_STYLE = {
  gdelt:      { label: 'GDELT', color: 'rgba(251,146,60,0.15)',  border: 'rgba(251,146,60,0.3)',  text: 'rgb(251,146,60)' },
  world_bank: { label: 'WB',    color: 'rgba(99,179,237,0.12)',  border: 'rgba(99,179,237,0.3)',  text: 'rgb(99,179,237)' },
  fred:       { label: 'FRED',  color: 'rgba(167,243,208,0.12)', border: 'rgba(167,243,208,0.3)', text: 'rgb(134,239,172)' },
}

function SourcePill({ source }) {
  const s = SOURCE_STYLE[source] ?? {
    label: source.toUpperCase().slice(0, 6),
    color: 'rgba(255,255,255,0.05)',
    border: 'rgba(255,255,255,0.1)',
    text: 'rgba(255,255,255,0.4)',
  }
  return (
    <span
      className="font-mono text-[9px] tracking-wider px-1.5 py-0.5 rounded shrink-0"
      style={{ background: s.color, border: `1.5px solid ${s.border}`, color: s.text }}
    >
      {s.label}
    </span>
  )
}

export default function SignalFeed({ signals, loading }) {
  return (
    <div className="space-y-3.5">
      <div className="font-display text-xs tracking-[0.2em] text-slate-200 mb-5"> SIGNAL FEED</div>

      {loading && (
        <div className="space-y-2.5">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="h-7 rounded animate-pulse"
              style={{ background: 'rgba(0,230,118,0.06)', opacity: 1 - i * 0.12 }}
            />
          ))}
        </div>
      )}

      {!loading && !signals?.length && (
        <div className="font-mono text-xs py-6 text-center" style={{ color: 'rgba(0,230,118,0.25)' }}>
          No signals yet
          <div className="text-[10px] mt-1 text-slate-200">Pipeline initialising</div>
        </div>
      )}

      {!loading && !!signals?.length && (
        <AnimatePresence>
          {signals.map((sig, i) => (
            <motion.div
              key={`${sig.source}-${sig.signal_type}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
              className="flex items-center gap-3"
            >
              { <SourcePill source={sig.source} /> }
              <span className="font-display text-sm text-slate-200 flex-1 truncate">
                {labelFor(sig)}
              </span>
              <span className="font-mono text-sm text-white shrink-0 tabular-nums">
                {formatValue(sig)}
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
      )}
    </div>
  )
}
