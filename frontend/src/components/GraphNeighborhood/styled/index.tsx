import styled from 'styled-components'

export const SWrap = styled.div`
  width: 100%;
  background: #faf9f6;
  border: 1px solid #ededea;
  border-radius: 8px;
  overflow: hidden;
`

export const SSvg = styled.svg`
  display: block;
  width: 100%;
  height: 320px;
`

export const SLegend = styled.div`
  display: flex;
  gap: 16px;
  padding: 8px 12px;
  font-size: 11px;
  color: #888780;
  flex-wrap: wrap;
`

export const SDot = styled.span<{ $color: string }>`
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: ${({ $color }) => $color};
  margin-right: 5px;
  vertical-align: middle;
`

export const SEmpty = styled.p`
  color: #888780;
  font-size: 13px;
  padding: 32px;
  text-align: center;
`
