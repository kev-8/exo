/**
 * Static showcase data — pre-baked so cards render instantly
 * without waiting for the API.
 *
 * Positions:
 *   side     'left' | 'right'
 *   topPct   0–1 (fraction of screen height for card vertical center)
 */

export const SHOWCASE = [
  {
    iso2: 'TW', label: 'Taiwan', lat: 23.7, lon: 121, type: 'risk',
    position: { side: 'right', topPct: 0.44 },
    risk: {
      composite: 0.4612,
      structural: 0.42, shortTerm: 0.4064, acute: 0.5783,
      dims: [
        { name: 'Conflict Intensity',  score: 0.51 },
        { name: 'Political Stability', score: 0.50 },
        { name: 'Policy Predictability', score: 0.50 },
      ],
    },
  },
  {
    iso2: 'RU', label: 'Russia', lat: 61.5, lon: 105, type: 'trade',
    position: { side: 'right', topPct: 0.26 },
    trade: {
      partners: [
        { name: 'China',      share: 33.9, goods: ['Oil & Gas', 'Coal', 'Timber'] },
        { name: 'India',      share: 13.7, goods: ['Crude Oil', 'Fertilizers'] },
        { name: 'Turkey',     share:  7.8, goods: ['Natural Gas', 'Grain'] },
        { name: 'Belarus',    share:  5.1, goods: ['Oil & Gas', 'Machinery'] },
      ],
    },
  },
  {
    iso2: 'IN', label: 'India', lat: 20.6, lon: 79, type: 'trade',
    position: { side: 'left', topPct: 0.55 },
    trade: {
      partners: [
        { name: 'United States', share: 17.7, goods: ['Pharmaceuticals', 'IT Services'] },
        { name: 'UAE',           share:  7.1, goods: ['Petroleum Products', 'Gems'] },
        { name: 'China',         share:  3.7, goods: ['Chemicals', 'Electronics'] },
        { name: 'Netherlands',   share:  2.9, goods: ['Petroleum Products', 'Chemicals'] },
      ],
    },
  },
  {
    iso2: 'IR', label: 'Iran', lat: 32.4, lon: 54, type: 'risk',
    position: { side: 'left', topPct: 0.30 },
    risk: {
      composite: 0.5484,
      structural: 0.5009, shortTerm: 0.4071, acute: 0.5937,
      dims: [
        { name: 'Political Stability', score: 0.8148 },
        { name: 'Policy Predictability', score: 0.5101 },
        { name: 'Conflict Intensity',  score: 0.51 },
      ],
    },
  },
  {
    iso2: 'IL', label: 'Israel', lat: 31.0, lon: 35, type: 'risk',
    position: { side: 'right', topPct: 0.62 },
    risk: {
      composite: 0.4691,
      structural: 0.4083, shortTerm: 0.4061, acute: 0.593,
      dims: [
        { name: 'Political Stability', score: 0.5821 },
        { name: 'Conflict Intensity',  score: 0.51 },
        { name: 'Policy Predictability', score: 0.4381 },
      ],
    },
  },
  {
    iso2: 'UA', label: 'Ukraine', lat: 48.4, lon: 31, type: 'trade',
    position: { side: 'left', topPct: 0.38 },
    trade: {
      partners: [
        { name: 'Poland',  share: 18.6, goods: ['Grain', 'Steel', 'Machinery'] },
        { name: 'China',   share: 14.0, goods: ['Sunflower Oil', 'Iron Ore'] },
        { name: 'Germany', share:  9.7, goods: ['Steel', 'Chemicals'] },
        { name: 'Turkey',  share:  8.7, goods: ['Grain', 'Steel'] },
      ],
    },
  },
  {
    iso2: 'US', label: 'United States', lat: 37.1, lon: -95, type: 'risk',
    position: { side: 'right', topPct: 0.34 },
    risk: {
      composite: 0.37,
      structural: 0.3613, shortTerm: 0.4066, acute: 0.4108,
      dims: [
        { name: 'Policy Predictability', score: 0.441 },
        { name: 'Political Stability',   score: 0.4093 },
        { name: 'Economic Stress',       score: 0.3992 },
      ],
    },
  },
]
