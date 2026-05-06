import { motion, AnimatePresence } from 'framer-motion'
import { riskColor, riskLabel, riskClass } from '../lib/riskColor'
import ScoreBar from './ScoreBar'


const DIMENSION_LABELS = {
  political_stability:    'Political Stability',
  conflict_intensity:     'Conflict Intensity',
  policy_predictability:  'Policy Predictability',
  sanctions_risk:         'Sanctions Risk',
  economic_stress:        'Economic Stress',
}

const TIER_LABELS = {
  structural: 'STRUCTURAL',
  short_term: 'SHORT-TERM',
  acute:      'ACUTE',
}

export default function RiskCard({ snapshot, country, onClose }) {
  if (!snapshot) return null

  const { composite_score, structural_score, short_term_score, acute_score, dimensions } = snapshot
  const compositeColor = riskColor(composite_score)

  return (
    <AnimatePresence>
      <motion.div
        key="risk-card"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 40 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="relative w-full rounded-lg overflow-hidden border-white"
        style={{
          background: 'rgba(10,10,10,0.75)'
        }}
      >
        <div style={{ padding: '1.0rem' }}>
          <div className="grid gap-10" style={{ gridTemplateColumns: '2fr 3fr' }}>
            {/* ── Left column: identity + composite + tiers ── */}
            <div className="flex flex-col justify-between">
              {/* Header */}
              <div className="flex items-start justify-between mb-8">
                <div className="flex items-start gap-4">
                  {country?.iso2 && (
                    <img
                      src={`https://flagcdn.com/w80/${country.iso2.toLowerCase()}.png`}
                      alt={country.iso2}
                      style={{ width: 42, height: 'auto', borderRadius: 3, display: 'block' }}
                    />
                  )}
                  <div>
                    <div className="font-display text-xs tracking-[0.2em] text-slate-200 uppercase mb-1">{country?.region}</div>
                    <h2 className="font-display text-2xl font-semibold tracking-wide text-white">
                      {country?.name}
                    </h2>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-4xl font-medium" style={{ color: compositeColor }}>
                    {composite_score.toFixed(2)}
                  </div>
                  <div className="font-display text-sm tracking-[0.15em] mt-1" style={{ color: compositeColor }}>
                    {riskLabel(composite_score)}
                  </div>
                </div>
              </div>

              {/* Three-tier scores */}
              <div className="grid grid-cols-3 gap-3 mt-6">
                {[
                  { label: 'STRUCTURAL', score: structural_score },
                  { label: 'SHORT-TERM', score: short_term_score },
                  { label: 'ACUTE',      score: acute_score },
                ].map(({ label, score }, i) => (
                  <motion.div
                    key={label}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 + i * 0.07 }}
                    className="rounded p-4 text-center"
                    style={{}}
                  >
                    <div className="font-display text-[10px] tracking-[0.2em] text-slate-200 mb-1.5">{label}</div>
                    <div className="font-mono text-xl font-medium" style={{ color: riskColor(score) }}>
                      {score.toFixed(2)}
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Timestamp */}
              <div className="mt-auto pt-10 font-mono text-[10px] text-slate-500">
                as of {new Date(snapshot.as_of_ts).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}
              </div>
            </div>

            {/* ── Right column: dimension bars ── */}
            <div className="flex flex-col justify-center">
              <div className="font-display text-xs tracking-[0.2em] text-slate-200 mb-6">RISK DIMENSIONS</div>
              <div className="space-y-5">
                {dimensions.map((dim, i) => (
                  <ScoreBar
                    key={dim.name}
                    label={DIMENSION_LABELS[dim.name] || dim.name}
                    score={dim.score}
                    delay={0.25 + i * 0.06}
                    large
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
