import React from 'react'
import { useNetworkDiagram } from './hooks'
import { getColumnX, getCircleYs, SVG_W, SVG_H, R, MAX_DISPLAY } from './helpers'

const NetworkDiagram = () => {
  const { layerSizes, loading, error } = useNetworkDiagram()

  if (loading) return <p>Loading diagram...</p>
  if (error) return <p className="error">Failed to load config</p>

  const columns = layerSizes.map((size, i) => ({
    x: getColumnX(layerSizes.length, i),
    ys: getCircleYs(size),
    size,
    truncated: size > MAX_DISPLAY,
  }))

  return (
    <svg width={SVG_W} height={SVG_H}>
      {/* Connection lines: every circle in layer i → every circle in layer i+1 */}
      {columns.slice(0, -1).map((col, i) => {
        const next = columns[i + 1]
        return col.ys.flatMap((y1, j) =>
          next.ys.map((y2, k) => (
            <line
              key={`l-${i}-${j}-${k}`}
              x1={col.x} y1={y1}
              x2={next.x} y2={y2}
              stroke="#9FE1CB"
              strokeWidth={0.5}
              opacity={0.6}
            />
          ))
        )
      })}

      {/* Circles, truncation marker, and size labels */}
      {columns.map((col, i) => (
        <g key={`col-${i}`}>
          {col.ys.map((y, j) => (
            <circle
              key={j}
              cx={col.x} cy={y}
              r={R}
              fill="white"
              stroke="#3C3489"
              strokeWidth={1.5}
            />
          ))}

          {col.truncated && (
            <text
              x={col.x}
              y={col.ys[col.ys.length - 1] + R + 16}
              textAnchor="middle"
              fontSize={13}
              fill="#5F5E5A"
            >
              ...
            </text>
          )}

          <text
            x={col.x}
            y={SVG_H - 8}
            textAnchor="middle"
            fontSize={12}
            fill="#5F5E5A"
          >
            {col.size}
          </text>
        </g>
      ))}
    </svg>
  )
}

export default NetworkDiagram
