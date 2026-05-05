import { motion } from 'framer-motion'

export default function TradePanel({ tradeData, loading }) {
  if (loading) {
    return (
      <div className="space-y-2">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-8 rounded bg-white/3 animate-pulse" style={{ opacity: 1 - i * 0.15 }} />
        ))}
      </div>
    )
  }

  if (!tradeData?.partners?.length) return null

  const max = tradeData.partners[0].trade_usd_b

  return (
    <div>
      <div className="font-display text-xs tracking-[0.2em] text-slate-200 mb-6">TOP EXPORT PARTNERS</div>
      <div className="space-y-5">
        {tradeData.partners.map((p, i) => (
          <motion.div
            key={p.iso2}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }}
            className="group"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-sm text-slate-200">{p.name}</span>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-slate-200">{p.share_pct}%</span>
                <span className="font-mono text-sm text-slate-200">${p.trade_usd_b}B</span>
              </div>
            </div>
            {/* Bar */}
            <div className="h-[2px] bg-white/5 rounded-full overflow-hidden mb-2.5">
              <motion.div
                className="h-full rounded-full"
                style={{ background: 'linear-gradient(90deg, rgba(0,230,118,0.6), rgba(0,230,118,0.2))' }}
                initial={{ width: 0 }}
                animate={{ width: `${(p.trade_usd_b / max) * 100}%` }}
                transition={{ duration: 0.7, delay: 0.1 + i * 0.06, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
            {/* Goods tags */}
            <div className="flex gap-1.5 flex-wrap">
              {p.goods.map(g => (
                <span
                  key={g}
                  className="font-mono text-[10px] px-2 py-0.5 rounded"
                  style={{ background: 'rgba(0,230,118,0.06)', color: 'rgba(0,230,118,0.5)', border: '1px solid rgba(0,230,118,0.1)' }}
                >
                  {g}
                </span>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
