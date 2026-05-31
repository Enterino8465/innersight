import styled from 'styled-components'

const CELL_BG: Record<string, string> = {
  tp:      '#d1fae5',
  tn:      '#f5f4f0',
  fp:      '#fef3c7',
  fn:      '#fee2e2',
}

const CELL_BORDER: Record<string, string> = {
  tp:      '#0F6E56',
  tn:      '#d1d0ca',
  fp:      '#D97706',
  fn:      '#DC2626',
}

export const SWrap = styled.div`
  display: flex;
  gap: 32px;
  align-items: flex-start;
  flex-wrap: wrap;
`

export const SMatrixBlock = styled.div``

export const SAxisLabel = styled.p`
  margin: 0 0 6px;
  font-size: 12px;
  font-weight: 600;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`

export const SGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  max-width: 340px;
`

export const SCell = styled.div<{ $kind: string }>`
  padding: 18px 12px;
  border-radius: 8px;
  text-align: center;
  border: 2px solid ${({ $kind }) => CELL_BORDER[$kind] ?? '#d1d0ca'};
  background: ${({ $kind }) => CELL_BG[$kind] ?? '#f5f4f0'};
`

export const SCellKind = styled.div`
  font-size: 11px;
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 6px;
`

export const SCellCount = styled.div`
  font-size: 26px;
  font-weight: 800;
  color: #1a1a2e;
  line-height: 1;
`

export const SCellPct = styled.div`
  font-size: 12px;
  color: #888780;
  margin-top: 4px;
`

export const SLegend = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 28px;
`

export const SLegendItem = styled.div<{ $kind: string }>`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #5f5e5a;

  &::before {
    content: '';
    display: block;
    width: 12px;
    height: 12px;
    border-radius: 3px;
    background: ${({ $kind }) => CELL_BG[$kind] ?? '#f5f4f0'};
    border: 2px solid ${({ $kind }) => CELL_BORDER[$kind] ?? '#d1d0ca'};
    flex-shrink: 0;
  }
`
