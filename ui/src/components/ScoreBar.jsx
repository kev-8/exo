import { motion } from 'framer-motion'
import { riskColor } from '../lib/riskColor'

export default function ScoreBar({ label, score, delay = 0, showValue = true, large = false }) {
  const color = riskColor(score)
  const pct = Math.round((score ?? 0) * 100)

  return (
    <div className="flex items-center gap-3 w-full">
      <span
        className={`font-display text-slate-400 shrink-0 ${large ? 'text-sm' : 'text-xs'}`}
        style={{ minWidth: '10rem' }}
      >{label}</span>
      <div className="h-[4px] bg-white/5 rounded-full overflow-hidden flex-1">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, delay, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      {showValue && (
        <span className={`font-mono w-10 text-right shrink-0 ${large ? 'text-sm' : 'text-xs'}`} style={{ color }}>
          {(score ?? 0).toFixed(2)}
        </span>
      )}
    </div>
  )
}
