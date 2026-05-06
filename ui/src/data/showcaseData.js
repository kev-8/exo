/**
 * Static showcase data — pre-baked so cards render instantly
 * without waiting for the API.
 *
 * Positions:
 *   side     'left' | 'right'
 *   topPct   0–1 (fraction of screen height for card vertical center)
 *
 * Ordered west-to-east by longitude so cards appear in geographic
 * sequence as the globe auto-rotates eastward.
 */

export const SHOWCASE = [
  {
    iso2: 'US', label: 'United States', lat: 37.09, lon: -95.71, type: 'risk',
    position: { side: 'right', topPct: 0.38 },
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
  {
    iso2: 'BR', label: 'Brazil', lat: -14.24, lon: -51.93, type: 'risk',
    position: { side: 'left', topPct: 0.55 },
    risk: {
      composite: 0.42,
      structural: 0.39, shortTerm: 0.44, acute: 0.46,
      dims: [
        { name: 'Economic Stress',       score: 0.52 },
        { name: 'Political Stability',   score: 0.45 },
        { name: 'Policy Predictability', score: 0.41 },
      ],
    },
  },
  {
    iso2: 'FR', label: 'France', lat: 46.23, lon: 2.21, type: 'risk',
    position: { side: 'left', topPct: 0.35 },
    risk: {
      composite: 0.28,
      structural: 0.22, shortTerm: 0.28, acute: 0.34,
      dims: [
        { name: 'Policy Predictability', score: 0.31 },
        { name: 'Economic Stress',       score: 0.29 },
        { name: 'Political Stability',   score: 0.27 },
      ],
    },
  },
  {
    iso2: 'NG', label: 'Nigeria', lat: 9.08, lon: 8.68, type: 'trade',
    position: { side: 'right', topPct: 0.52 },
    trade: {
      partners: [
        { name: 'India',  share: 14.2, goods: ['Crude Oil', 'LNG'] },
        { name: 'Spain',  share: 11.8, goods: ['Crude Oil', 'Petroleum'] },
        { name: 'China',  share:  9.6, goods: ['Crude Oil', 'Refined Products'] },
        { name: 'France', share:  8.1, goods: ['Crude Oil', 'LNG'] },
      ],
    },
  },
  {
    iso2: 'KE', label: 'Kenya', lat: -0.02, lon: 37.91, type: 'trade',
    position: { side: 'left', topPct: 0.48 },
    trade: {
      partners: [
        { name: 'Uganda',   share: 11.3, goods: ['Tea', 'Horticulture', 'Cement'] },
        { name: 'USA',      share:  9.7, goods: ['Tea', 'Apparel', 'Coffee'] },
        { name: 'Pakistan', share:  7.4, goods: ['Tea', 'Flowers'] },
        { name: 'China',    share:  6.9, goods: ['Tea', 'Sesame'] },
      ],
    },
  },
  {
    iso2: 'IR', label: 'Iran', lat: 32.43, lon: 53.69, type: 'risk',
    position: { side: 'left', topPct: 0.30 },
    risk: {
      composite: 0.5484,
      structural: 0.5009, shortTerm: 0.4071, acute: 0.5937,
      dims: [
        { name: 'Political Stability',   score: 0.8148 },
        { name: 'Policy Predictability', score: 0.5101 },
        { name: 'Conflict Intensity',    score: 0.51 },
      ],
    },
  },
  {
    iso2: 'RU', label: 'Russia', lat: 61.52, lon: 105.32, type: 'trade',
    position: { side: 'right', topPct: 0.26 },
    trade: {
      partners: [
        { name: 'China',   share: 33.9, goods: ['Oil & Gas', 'Coal', 'Timber'] },
        { name: 'India',   share: 13.7, goods: ['Crude Oil', 'Fertilizers'] },
        { name: 'Turkey',  share:  7.8, goods: ['Natural Gas', 'Grain'] },
        { name: 'Belarus', share:  5.1, goods: ['Oil & Gas', 'Machinery'] },
      ],
    },
  },
  {
    iso2: 'TW', label: 'Taiwan', lat: 23.70, lon: 120.96, type: 'risk',
    position: { side: 'right', topPct: 0.44 },
    risk: {
      composite: 0.4612,
      structural: 0.42, shortTerm: 0.4064, acute: 0.5783,
      dims: [
        { name: 'Conflict Intensity',    score: 0.51 },
        { name: 'Political Stability',   score: 0.50 },
        { name: 'Policy Predictability', score: 0.50 },
      ],
    },
  },
]
