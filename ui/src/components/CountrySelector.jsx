import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

function FlagImg({ iso2, size = 24 }) {
  return (
    <img
      src={`https://flagcdn.com/w80/${iso2.toLowerCase()}.png`}
      alt={iso2}
      style={{ width: size, height: 'auto', borderRadius: 2, display: 'block' }}
    />
  )
}

export default function CountrySelector({ countries, selected, onSelect, onOpenChange }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  const setOpenWithCallback = (val) => {
    const next = typeof val === 'function' ? val(open) : val
    setOpen(next)
    onOpenChange?.(next)
  }

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpenWithCallback(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const selectedCountry = countries?.find(c => c.iso2 === selected)

  return (
    <div ref={ref} className="relative w-full">
      <button
        onClick={() => setOpenWithCallback(o => !o)}
        className="w-full flex items-center justify-between rounded font-display tracking-wide transition-all duration-200"
        style={{
          padding: '1rem 1.25rem',
          fontSize: '1rem',
          background: open ? 'rgba(0,230,118,0.08)' : 'rgba(255,255,255,0.03)',
          border: `1.5px solid ${open ? 'rgba(0,230,118,0.3)' : 'white'}`,
          color: selectedCountry ? 'white' : 'rgba(255,255,255,0.3)',
        }}
      >
        <span className="flex items-center gap-3">
          {selectedCountry ? (
            <>
              <FlagImg iso2={selectedCountry.iso2} size={26} />
              <span>{selectedCountry.name}</span>
            </>
          ) : (
            <span className="tracking-[0.15em] text-xs" style={{ color: '#00E676' }}>SELECT A COUNTRY</span>
          )}
        </span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          style={{ fontSize: '0.95rem', display: 'inline-block', lineHeight: 1, color: '#00E676' }}
        >
          ∧
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-full left-0 right-0 mb-1 rounded z-50 overflow-hidden"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1.5px solid white',
              boxShadow: '0 16px 40px rgba(0,0,0,0.6)',
              padding: '0.5rem',
            }}
          >
            {countries?.map((c) => (
              <button
                key={c.iso2}
                onClick={() => { onSelect(c.iso2); setOpenWithCallback(false) }}
                className="w-full flex items-center gap-3 text-left transition-all duration-150 font-mono rounded"
                style={{
                  padding: '0.75rem 1rem',
                  fontSize: '0.95rem',
                  background: c.iso2 === selected ? 'rgba(0,230,118,0.08)' : 'transparent',
                  color: c.iso2 === selected ? '#00E676' : 'white',
                  borderLeft: c.iso2 === selected ? '2px solid #00E676' : '2px solid transparent',
                }}
                onMouseEnter={e => {
                  if (c.iso2 !== selected) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'
                }}
                onMouseLeave={e => {
                  if (c.iso2 !== selected) e.currentTarget.style.background = 'transparent'
                }}
              >
                <FlagImg iso2={c.iso2} size={26} />
                <span>{c.name}</span>
                <span className="ml-auto text-xs text-white/50">{c.iso2}</span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
