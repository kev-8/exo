import { motion, AnimatePresence } from 'framer-motion'

function timeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime()
  const m = Math.floor(diff / 60000)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

// Per-signal display config: human label + how to format the value. `fmt`
// gets the stored value and the record's metadata, which holds the raw
// figure for signals stored as 0–1 scores; each falls back to the score.
const pct = (x, dp = 0) => (x * 100).toFixed(dp) + '%'
const score = v => v.toFixed(2) + ' / 1.0'
const signed = (x, dp = 1) => (x >= 0 ? '+' : '') + x.toFixed(dp)
const count = (unit) => (v, m) => m?.raw_count != null ? `${m.raw_count} ${unit}` : score(v)
const wgi = (v, m) => m?.estimate != null ? signed(m.estimate, 2) + ' est.' : score(v)

const SIGNAL_CONFIG = {
  // GDELT
  news_sentiment:       { label: 'News Sentiment',      fmt: score },
  news_magnitude:       { label: 'News Volume',         fmt: (v, m) => m?.ratio != null ? m.ratio.toFixed(1) + '× baseline' : score(v) },
  // World Bank
  debt_to_gdp:          { label: 'Debt / GDP',           fmt: v => v.toFixed(1) + '%' },
  inflation_rate:       { label: 'Inflation Rate',       fmt: v => v.toFixed(1) + '%' },
  unemployment_rate:    { label: 'Unemployment',         fmt: v => v.toFixed(1) + '%' },
  gdp_growth:           { label: 'GDP Growth',           fmt: v => signed(v) + '%' },
  // Stored as (energy imports % + 100) / 200; negative = net energy exporter.
  energy_imports_pct:   { label: 'Energy Imports',       fmt: v => (v * 200 - 100).toFixed(1) + '% of use' },
  gross_savings:        { label: 'Gross Savings',        fmt: v => v.toFixed(1) + '% of GNI' },
  // Stored as (exports + imports) / 200; the raw percentages are in metadata.
  trade_openness:       { label: 'Trade Openness',       fmt: (v, m) => m?.exports_pct_gdp != null ? (m.exports_pct_gdp + m.imports_pct_gdp).toFixed(0) + '% of GDP' : score(v) },
  exports_pct_gdp:      { label: 'Exports',              fmt: v => v.toFixed(1) + '% of GDP' },
  imports_pct_gdp:      { label: 'Imports',              fmt: v => v.toFixed(1) + '% of GDP' },
  current_account:      { label: 'Current Account',      fmt: v => signed(v) + '% of GDP' },
  electricity_access:   { label: 'Electricity Access',   fmt: v => v.toFixed(1) + '% of pop.' },
  // FRED
  fred_unrate:          { label: 'Unemployment Rate',    fmt: v => v.toFixed(1) + '%' },
  fred_cpiaucsl:        { label: 'CPI Inflation',        fmt: v => v.toFixed(1) },
  fred_fedfunds:        { label: 'Fed Funds Rate',       fmt: v => v.toFixed(2) + '%' },
  fred_t10y2y:          { label: 'Yield Curve Spread',   fmt: v => signed(v, 2) + '%' },
  fred_umcsent:         { label: 'Consumer Sentiment',   fmt: v => v.toFixed(1) },
  economic_indicator:   { label: 'FRED Composite',       fmt: score },
  // WGI — estimates run about −2.5 (worst) to +2.5 (best)
  political_stability:  { label: 'Political Stability',  fmt: wgi },
  voice_accountability: { label: 'Voice & Accountability', fmt: wgi },
  rule_of_law:          { label: 'Rule of Law',          fmt: wgi },
  control_of_corruption:{ label: 'Control of Corruption', fmt: wgi },
  // UNGA
  ideal_point_variance: { label: 'UN Vote Variance',     fmt: score },
  // OFAC
  sdn_entity_count:     { label: 'Sanctioned Entities',  fmt: count('on SDN list') },
  // WITS
  trade_concentration:  { label: 'Trade with US + EU',   fmt: (v, m) => m?.us_pct != null ? pct(m.us_pct + m.eu_pct) + ' of trade' : score(v) },
  secondary_exposure:   { label: 'Trade with Sanctioned', fmt: (v, m) => m?.sanctioned_pct != null ? pct(m.sanctioned_pct, 1) + ' of trade' : score(v) },
  // UCDP
  ucdp_ged_events:      { label: 'Conflict Events',      fmt: (v, m) => m?.raw_event_count != null ? `${Math.round(m.raw_event_count)} since 2010` : score(v) },
  ucdp_ged_fatalities:  { label: 'Conflict Deaths',      fmt: (v, m) => m?.raw_fatalities != null ? `${Math.round(m.raw_fatalities).toLocaleString()} since 2010` : score(v) },
  ucdp_candidate_events:{ label: 'Recent Conflict Events', fmt: (v, m) => m?.raw_event_count != null ? `${Math.round(m.raw_event_count)} this year` : score(v) },
  // Comtrade (AI exposure)
  hw_contested_share:   { label: 'Chip & IT Imports',    fmt: (v, m) => m?.raw_share != null ? pct(m.raw_share) + ' from CN/HK/TW' : score(v) },
  supplier_chokepoint:  { label: 'Supplier Chokepoint',  fmt: (v, m) => m?.top_share ? `${pct(m.top_share)} of HS ${m.top_hs_code}` : 'none' },
  cn_mineral_dependence:{ label: 'China Mineral Reliance', fmt: score },
  // Export-control lists (AI exposure)
  entity_list_count:    { label: 'US Entity List',       fmt: count('entities') },
  jp_eul_count:         { label: 'Japan End User List',  fmt: count('entities') },
  cn_ec_count:          { label: 'China Control Lists',  fmt: count('entities') },
  // Digital infrastructure (AI exposure)
  cables_in_service:    { label: 'Submarine Cables',     fmt: count('in service') },
  ixp_count:            { label: 'Internet Exchanges',   fmt: count('IXPs') },
  facility_count:       { label: 'Data Centres',         fmt: count('facilities') },
}

function formatValue(sig) {
  const cfg = SIGNAL_CONFIG[sig.signal_type]
  if (cfg) return cfg.fmt(sig.value, sig.metadata)
  return sig.value.toFixed(2)
}

function labelFor(sig) {
  return SIGNAL_CONFIG[sig.signal_type]?.label
    ?? sig.signal_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// Source pill styles
const pill = (label, rgb) => ({
  label, color: `rgba(${rgb},0.12)`, border: `rgba(${rgb},0.3)`, text: `rgb(${rgb})`,
})
const SOURCE_STYLE = {
  gdelt:           pill('GDELT',    '251,146,60'),
  world_bank:      pill('WB',       '99,179,237'),
  fred:            pill('FRED',     '134,239,172'),
  wgi:             pill('WGI',      '129,140,248'),
  unga_votes:      pill('UNGA',     '196,181,253'),
  ofac:            pill('OFAC',     '248,113,113'),
  wits:            pill('WITS',     '45,212,191'),
  ucdp_ged:        pill('UCDP',     '251,191,36'),
  ucdp_candidate:  pill('UCDP',     '251,191,36'),
  comtrade:        pill('COMTRADE', '244,114,182'),
  export_controls: pill('EXPORT',   '232,121,249'),
  digital_infra:   pill('INFRA',    '148,163,184'),
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
      className="font-mono text-[9px] tracking-wider px-1.5 py-0.5 rounded shrink-0 w-[4rem] text-center"
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
