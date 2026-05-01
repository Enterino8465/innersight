export const MAX_DISPLAY = 6
export const SVG_W = 480
export const SVG_H = 300
export const R = 12

// Evenly spaces layers horizontally across the SVG width
export function getColumnX(numLayers, idx) {
  return ((idx + 1) * SVG_W) / (numLayers + 1)
}

// Returns an array of cy values for up to MAX_DISPLAY circles, vertically centred
export function getCircleYs(layerSize) {
  const count = Math.min(layerSize, MAX_DISPLAY)
  const top = 20
  const bottom = SVG_H - 55  // reserve space for '...' and label
  const mid = (top + bottom) / 2

  if (count === 1) return [mid]

  const spacing = Math.min((bottom - top) / (count - 1), 40)
  const span = (count - 1) * spacing
  const startY = mid - span / 2
  return Array.from({ length: count }, (_, i) => startY + i * spacing)
}
