import styled from 'styled-components'

export const SWrap = styled.div`
  overflow-x: auto;
  padding: 4px 0;
`

export const SGrid = styled.div<{ $days: number }>`
  display: grid;
  grid-template-columns: 130px repeat(${({ $days }) => $days}, minmax(14px, 1fr));
  gap: 1px;
  min-width: max-content;
`

export const SCorner = styled.div`
  background: transparent;
`

export const SColHeader = styled.div`
  font-size: 9px;
  color: #888780;
  text-align: center;
  padding-bottom: 2px;
  white-space: nowrap;
`

export const SRowLabel = styled.div`
  font-size: 11px;
  color: #1a1a2e;
  display: flex;
  align-items: center;
  padding-right: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`

export const SCell = styled.div<{ $bg: string }>`
  height: 18px;
  border-radius: 2px;
  background: ${({ $bg }) => $bg};
  cursor: default;
`

export const SLegend = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  font-size: 11px;
  color: #888780;
`

export const SLegendBar = styled.div`
  width: 140px;
  height: 10px;
  border-radius: 5px;
  background: linear-gradient(to right, rgb(37, 99, 235), white, rgb(220, 38, 38));
`

export const SEmpty = styled.p`
  color: #888780;
  font-size: 13px;
  padding: 24px 0;
  text-align: center;
`
