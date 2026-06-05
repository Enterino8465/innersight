import React from 'react'
import {
  SWrap,
  SGrid,
  SCorner,
  SColHeader,
  SRowLabel,
  SCell,
  SLegend,
  SLegendBar,
  SEmpty,
} from './styled'

interface Props {
  /** z-scores indexed as matrix[day][feature]. */
  matrix: number[][]
  featureNames: string[]
  dayLabels: string[]
}

const lerp = (a: number, b: number, x: number): number => Math.round(a + (b - a) * x)

/** Blue (≤ −3σ) → white (0) → red (≥ +3σ). */
const cellColor = (z: number): string => {
  const t = Math.max(-1, Math.min(1, z / 3))
  if (t <= 0) {
    const x = t + 1 // 0 (blue) … 1 (white)
    return `rgb(${lerp(37, 255, x)},${lerp(99, 255, x)},${lerp(235, 255, x)})`
  }
  const x = 1 - t // 1 (white) … 0 (red)
  return `rgb(255,${lerp(38, 255, x)},${lerp(38, 255, x)})`
}

const DeviationHeatmap: React.FC<Props> = ({ matrix, featureNames, dayLabels }) => {
  const nDays = matrix.length
  if (nDays === 0 || featureNames.length === 0) {
    return <SEmpty>No deviation data available.</SEmpty>
  }

  return (
    <SWrap>
      {/* Rows = features, columns = days. */}
      <SGrid $days={nDays}>
        <SCorner />
        {dayLabels.map((label, di) => (
          <SColHeader key={di} title={label}>
            {di % 3 === 0 ? di : ''}
          </SColHeader>
        ))}

        {featureNames.map((feature, fi) => (
          <React.Fragment key={feature}>
            <SRowLabel title={feature}>{feature}</SRowLabel>
            {matrix.map((dayRow, di) => {
              const z = dayRow[fi] ?? 0
              return (
                <SCell
                  key={di}
                  $bg={cellColor(z)}
                  title={`${feature} · ${dayLabels[di] ?? `day ${di}`}\nz = ${z.toFixed(2)}`}
                />
              )
            })}
          </React.Fragment>
        ))}
      </SGrid>

      <SLegend>
        <span>−3σ</span>
        <SLegendBar />
        <span>+3σ</span>
      </SLegend>
    </SWrap>
  )
}

export default DeviationHeatmap
