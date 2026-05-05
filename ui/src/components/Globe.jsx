import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react'
import * as THREE from 'three'
import { SHOWCASE } from '../data/showcaseData'
import { riskColor } from '../lib/riskColor'

const PIN_ALTITUDE = 0.015

// Lat/lon for trade partner countries not in the tracked 10-country list
const WORLD_COORDS = {
  TR: [38.9, 35.2], BY: [53.7, 28.0], DE: [51.2, 10.5], PL: [52.0, 20.0],
  JP: [36.2, 138.3], KR: [36.5, 127.8], SA: [23.9, 45.1], AE: [24.0, 54.4],
  GB: [55.4, -3.4],  FR: [46.2, 2.2],  NL: [52.4, 4.9],  SG: [1.3, 103.8],
  MY: [3.1, 101.7],  BD: [23.7, 90.4], BR: [-14.2, -51.9], MX: [23.6, -102.6],
  IT: [42.8, 12.8],  ES: [40.5, -3.7], CH: [46.8, 8.2],  BE: [50.8, 4.5],
  ID: [-0.8, 113.9], TH: [15.9, 100.9], VN: [14.1, 108.3], EG: [26.8, 30.8],
  ZA: [-30.6, 22.9], NG: [9.1, 8.7],   AU: [-25.3, 133.8], CA: [56.1, -106.3],
  AR: [-38.4, -63.6], AT: [47.5, 14.6], CZ: [49.8, 15.5], HU: [47.2, 19.5],
  RO: [45.9, 24.9],  KZ: [48.0, 66.9], UZ: [41.4, 64.6], AZ: [40.1, 47.6],
  GE: [42.3, 43.4],  AM: [40.1, 45.0], FI: [64.0, 26.0], SE: [60.1, 18.6],
  NO: [60.5, 8.5],   DK: [56.3, 9.5],  HK: [22.4, 114.1], TZ: [-6.4, 34.9],
  ET: [9.1, 40.5],   GH: [7.9, -1.0],  CI: [7.5, -5.5],  CM: [3.9, 11.5],
}

function getDestCoords(iso2, countries) {
  const tracked = countries?.find(c => c.iso2 === iso2)
  if (tracked) return [tracked.lat, tracked.lon]
  return WORLD_COORDS[iso2] ?? null
}

function angDist(a, b) {
  return Math.abs(((a - b + 540) % 360) - 180)
}

function makePinEl(color, highlighted, onClickRef, iso2) {
  const r     = highlighted ? 7  : 4
  const spike = highlighted ? 10 : 6
  const ring  = highlighted ? 11 : 0
  const svgW  = (ring || r) * 2 + 4
  const svgH  = (ring || r) * 2 + spike + 4
  const cx    = svgW / 2
  const cy    = (ring || r) + 2

  const el = document.createElement('div')
  el.style.cssText = 'cursor:pointer;display:block;user-select:none;'
  el.innerHTML = `
    <svg width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}" style="overflow:visible;display:block">
      ${ring ? `<circle cx="${cx}" cy="${cy}" r="${ring}" fill="none" stroke="${color}" stroke-width="1" opacity="0.35"/>` : ''}
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="${color}" opacity="${highlighted ? 1 : 0.82}"/>
      <line x1="${cx}" y1="${cy + r}" x2="${cx}" y2="${cy + r + spike}"
            stroke="${color}" stroke-width="${highlighted ? 2 : 1.5}" opacity="${highlighted ? 0.85 : 0.55}" stroke-linecap="round"/>
    </svg>`
  el.addEventListener('click', e => { e.stopPropagation(); onClickRef.current?.(iso2) })
  return el
}

const Globe = forwardRef(function Globe({
  riskData, tradeData, countries,
  selectedIso2, onCountryClick, onGlobeClick, onShowcaseChange, paused,
}, ref) {
  const mountRef          = useRef(null)
  const globeRef          = useRef(null)
  const pollRef           = useRef(null)
  const activeRef         = useRef(null)
  const onCountryClickRef = useRef(onCountryClick)
  const [GlobeGL, setGlobeGL] = useState(null)

  useEffect(() => { onCountryClickRef.current = onCountryClick }, [onCountryClick])
  useEffect(() => { import('globe.gl').then(m => setGlobeGL(() => m.default)) }, [])

  const buildArcs = useCallback((iso2) => {
    if (!iso2 || !tradeData?.[iso2]) return []
    const origin = countries?.find(c => c.iso2 === iso2)
    if (!origin) return []

    // Top 3 partners with resolvable coordinates
    const partners = (tradeData[iso2].partners || [])
      .slice(0, 6)                              // try first 6, take top 3 with coords
      .map(p => {
        const coords = getDestCoords(p.iso2, countries)
        if (!coords) return null
        return { ...p, destLat: coords[0], destLng: coords[1] }
      })
      .filter(Boolean)
      .slice(0, 3)

    return partners.map(p => ({
      startLat: origin.lat, startLng: origin.lon,
      endLat:   p.destLat,  endLng:   p.destLng,
      color:    ['rgba(0,230,118,0.08)', 'rgba(0,230,118,0.85)'],
      stroke:   Math.max(0.5, (p.share_pct / 50) * 2),
      label:    `${p.name} — ${p.goods.join(', ')} — $${p.trade_usd_b}B`,
    }))
  }, [tradeData, countries])

  const buildPins = useCallback((highlightIso2) => {
    if (!countries) return []
    return countries.map(c => ({
      lat: c.lat, lng: c.lon, iso2: c.iso2,
      highlighted: c.iso2 === highlightIso2,
      color: c.iso2 === highlightIso2 ? '#00E676' : riskColor(riskData?.[c.iso2]?.composite_score),
    }))
  }, [countries, riskData])

  // globe.gl caches HTML elements and won't re-call htmlElement() when data
  // is replaced with same-length arrays. Clear first to force recreation.
  const refreshPins = useCallback((highlightIso2) => {
    const globe = globeRef.current
    if (!globe) return
    globe.htmlElementsData([])
    globe.htmlElementsData(buildPins(highlightIso2))
  }, [buildPins])

  useEffect(() => {
    if (!GlobeGL || !mountRef.current) return
    const el    = mountRef.current
    const globe = GlobeGL()(el)
    globeRef.current = globe

    // Amber cone material — shared across all arrowhead objects
    const coneMat = new THREE.MeshLambertMaterial({ color: 0x00E676, opacity: 0.88, transparent: true })

    globe
      .width(el.clientWidth).height(el.clientHeight)
      .backgroundColor('rgba(0,0,0,0)')
      .backgroundImageUrl('https://unpkg.com/three-globe/example/img/night-sky.png')
      .atmosphereAltitude(0)
      .globeImageUrl(null)
      .globeMaterial(new THREE.MeshPhongMaterial({
        color: 0x001a2e,
        emissive: 0x001a2e,
        wireframe: true,
        wireframeLinewidth: 1,
      }))
      // Country border polygons
      .polygonsData([])
      .polygonGeoJsonGeometry(d => d.geometry)
      .polygonCapColor(() => 'rgba(251,191,36,0.12)')
      .polygonSideColor(() => 'rgba(251,191,36,0.06)')
      .polygonStrokeColor(() => 'rgba(251,191,36,0.55)')
      .polygonAltitude(0.001)
      // Pins
      .htmlElementsData([])
      .htmlLat('lat').htmlLng('lng')
      .htmlAltitude(PIN_ALTITUDE)
      .htmlElement(d => makePinEl(d.color, d.highlighted, onCountryClickRef, d.iso2))
      // Trade arcs
      .arcsData([])
      .arcStartLat('startLat').arcStartLng('startLng')
      .arcEndLat('endLat').arcEndLng('endLng')
      .arcColor('color').arcStroke('stroke')
      .arcDashLength(0.45).arcDashGap(0.15)
      .arcDashAnimateTime(1600).arcAltitudeAutoScale(0.35)
      // Arrowhead cones at arc endpoints
      .customLayerData([])
      .customThreeObject(() =>
        new THREE.Mesh(new THREE.ConeGeometry(1.6, 5.5, 6), coneMat.clone())
      )
      .customThreeObjectUpdate((obj, d) => {
        // Position slightly above the surface so cones clear country pins (PIN_ALTITUDE = 0.015)
        const c1 = globe.getCoords(d.endLat, d.endLng, 0.03)
        obj.position.set(c1.x, c1.y, c1.z)

        // Orient cone along the arc's arrival tangent at the endpoint
        // Tangent of arrival at P1 = normalize(dot(P0_hat, P1_hat)*P1_hat - P0_hat)
        const c0  = globe.getCoords(d.startLat, d.startLng, 0)
        const v0  = new THREE.Vector3(c0.x, c0.y, c0.z).normalize()
        const v1  = new THREE.Vector3(c1.x, c1.y, c1.z).normalize()
        const tan = v1.clone().multiplyScalar(v0.dot(v1)).sub(v0).normalize()

        // ConeGeometry tip is along +Y; rotate so +Y aligns with arrival tangent
        obj.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tan)
      })
      // Labels
      .labelsData([])
      .labelLat('lat').labelLng('lng').labelText('name')
      .labelSize(0.55)
      .labelColor(() => 'rgba(0,230,118,0.9)')
      .labelDotRadius(0.25).labelResolution(2)
      .enablePointerInteraction(true)
      .onGlobeClick(() => onGlobeClick?.())

    // Load country borders GeoJSON
    fetch('https://raw.githubusercontent.com/vasturiano/globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson')
      .then(r => r.json())
      .then(data => { if (globeRef.current) globeRef.current.polygonsData(data.features) })
      .catch(() => {})

    // Load country borders GeoJSON
    fetch('https://raw.githubusercontent.com/vasturiano/globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson')
      .then(r => r.json())
      .then(data => { if (globeRef.current) globeRef.current.polygonsData(data.features) })
      .catch(() => {})

    globe.renderer().setPixelRatio(window.devicePixelRatio)
    globe.controls().autoRotate      = true
    globe.controls().autoRotateSpeed = 0.35
    globe.controls().enableZoom      = false
    globe.controls().enablePan       = false
    globe.controls().minPolarAngle   = Math.PI * 0.2
    globe.controls().maxPolarAngle   = Math.PI * 0.8

    const resize = () => globe.width(el.clientWidth).height(el.clientHeight)
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      globe._destructor?.()
      globeRef.current = null
    }
  }, [GlobeGL]) // eslint-disable-line react-hooks/exhaustive-deps

  // Poll for showcase country
  useEffect(() => {
    if (paused) { clearInterval(pollRef.current); return }
    pollRef.current = setInterval(() => {
      const globe = globeRef.current
      if (!globe) return
      let pov
      try { pov = globe.pointOfView() } catch { return }
      const lng = pov?.lng ?? 0

      let closest = null, closestDist = Infinity
      for (const s of SHOWCASE) {
        const d = angDist(lng, s.lon)
        if (d < closestDist) { closestDist = d; closest = s }
      }

      const threshold = activeRef.current ? 28 : 18
      const next = closestDist < threshold ? closest.iso2 : null

      if (next !== activeRef.current) {
        activeRef.current = next
        const entry = next ? SHOWCASE.find(s => s.iso2 === next) : null
        onShowcaseChange?.(next, entry?.type ?? null)
        refreshPins(next)
        if (next && entry?.type === 'trade') {
          const arcs = buildArcs(next)
          globe.arcsData(arcs)
          globe.customLayerData(arcs)
          const label = countries?.find(c => c.iso2 === next)
          globe.labelsData(label ? [label] : [])
        } else {
          globe.arcsData([])
          globe.customLayerData([])
          const label = next ? countries?.find(c => c.iso2 === next) : null
          globe.labelsData(label ? [label] : [])
        }
      }
    }, 200)
    return () => clearInterval(pollRef.current)
  }, [paused, refreshPins, buildArcs, countries, onShowcaseChange])

  // Fly to selected country
  useEffect(() => {
    if (!globeRef.current || !selectedIso2 || !countries) return
    const globe = globeRef.current
    const c = countries.find(co => co.iso2 === selectedIso2)
    if (!c) return
    clearInterval(pollRef.current)
    globe.controls().autoRotate = false
    globe.pointOfView({ lat: c.lat, lng: c.lon, altitude: 1.7 }, 1200)
    setTimeout(() => {
      const arcs = buildArcs(selectedIso2)
      refreshPins(selectedIso2)
      globe.arcsData(arcs)
      globe.customLayerData(arcs)
      globe.labelsData([c])
    }, 1300)
  }, [selectedIso2, countries, buildPins, buildArcs])

  useEffect(() => {
    if (!globeRef.current) return
    if (paused) {
      globeRef.current.controls().autoRotate = false
    } else {
      globeRef.current.controls().autoRotate      = true
      globeRef.current.controls().autoRotateSpeed = 0.35
    }
  }, [paused])

  useEffect(() => {
    if (!globeRef.current || paused) return
    refreshPins(activeRef.current)
  }, [riskData, paused, refreshPins])

  useImperativeHandle(ref, () => ({
    getCountryScreenPos(lat, lng) {
      const globe = globeRef.current
      if (!globe) return null
      try {
        const coords = globe.getCoords(lat, lng, PIN_ALTITUDE)
        if (!coords) return null
        const camera = globe.camera()
        const el = mountRef.current
        if (!camera || !el) return null
        const camDir   = camera.position.clone().normalize()
        const pointDir = new THREE.Vector3(coords.x, coords.y, coords.z).normalize()
        if (pointDir.dot(camDir) < 0.15) return null
        const v = new THREE.Vector3(coords.x, coords.y, coords.z)
        v.project(camera)
        return {
          x: (v.x + 1) / 2 * el.clientWidth,
          y: -(v.y - 1) / 2 * el.clientHeight,
        }
      } catch { return null }
    },
  }), [])

  return <div ref={mountRef} className="w-full h-full" style={{ cursor: 'grab' }} />
})

export default Globe
