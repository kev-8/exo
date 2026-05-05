/** Map a 0–1 risk score to a hex colour string. */
export function riskColor(score) {
  if (score == null) return '#64748b'
  if (score < 0.35) return '#10b981'   // green   — LOW
  if (score < 0.55) return '#f59e0b'   // amber   — MODERATE
  if (score < 0.70) return '#f43f5e'   // rose    — ELEVATED (distinct jump from amber)
  return '#ff1744'                      // red     — HIGH
}

/** Tailwind-compatible class string. */
export function riskClass(score) {
  if (score == null) return 'text-slate-500'
  if (score < 0.35) return 'risk-low'
  if (score < 0.55) return 'risk-medium'
  return 'risk-high'
}

/** Human-readable label. */
export function riskLabel(score) {
  if (score == null) return 'NO DATA'
  if (score < 0.35) return 'LOW'
  if (score < 0.55) return 'MODERATE'
  if (score < 0.70) return 'ELEVATED'
  return 'HIGH'
}
