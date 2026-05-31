import React from 'react'
import {
  SWrap,
  SMatrixBlock,
  SAxisLabel,
  SGrid,
  SCell,
  SCellKind,
  SCellCount,
  SCellPct,
  SLegend,
  SLegendItem,
} from './styled'

interface Props {
  matrix: number[][] | null
}

const CELLS = [
  { kind: 'tn', row: 0, col: 0, label: 'True Neg'  },
  { kind: 'fp', row: 0, col: 1, label: 'False Pos' },
  { kind: 'fn', row: 1, col: 0, label: 'False Neg' },
  { kind: 'tp', row: 1, col: 1, label: 'True Pos'  },
] as const

const LEGEND = [
  { kind: 'tp', text: 'True Positive — correctly flagged malicious' },
  { kind: 'tn', text: 'True Negative — correctly cleared normal' },
  { kind: 'fp', text: 'False Positive — false alarm' },
  { kind: 'fn', text: 'False Negative — missed threat (dangerous)' },
] as const

const ConfusionMatrix: React.FC<Props> = ({ matrix }) => {
  if (!matrix || matrix.length < 2) return null

  const total = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
  const pct = (n: number) => total > 0 ? `${((n / total) * 100).toFixed(1)}%` : '—'

  return (
    <SWrap>
      <SMatrixBlock>
        <SAxisLabel>Predicted → / Actual ↓</SAxisLabel>
        <SGrid>
          {CELLS.map(({ kind, row, col, label }) => {
            const count = matrix[row][col]
            return (
              <SCell key={kind} $kind={kind}>
                <SCellKind>{label}</SCellKind>
                <SCellCount>{count.toLocaleString()}</SCellCount>
                <SCellPct>{pct(count)}</SCellPct>
              </SCell>
            )
          })}
        </SGrid>
      </SMatrixBlock>

      <SLegend>
        {LEGEND.map(({ kind, text }) => (
          <SLegendItem key={kind} $kind={kind}>{text}</SLegendItem>
        ))}
      </SLegend>
    </SWrap>
  )
}

export default ConfusionMatrix
