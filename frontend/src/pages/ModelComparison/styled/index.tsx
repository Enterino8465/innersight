import styled from 'styled-components'

export const SPage = styled.main`
  padding: 0 24px 40px;
  max-width: 1100px;
  margin: 0 auto;
`

export const SHeader = styled.div`
  padding: 28px 0 4px;
`

export const SHeaderTitle = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SSubtitle = styled.p`
  margin: 6px 0 18px;
  font-size: 14px;
  color: #888780;
`

export const STable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  margin-bottom: 28px;
`

export const STh = styled.th`
  text-align: left;
  padding: 10px 12px;
  font-size: 12px;
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 2px solid #e8e7e2;
`

export const STr = styled.tr<{ $highlight?: boolean }>`
  background: ${({ $highlight }) => ($highlight ? '#f3f1fb' : 'transparent')};
`

export const STd = styled.td<{ $bold?: boolean }>`
  padding: 10px 12px;
  border-bottom: 1px solid #ededea;
  color: #1a1a2e;
  font-weight: ${({ $bold }) => ($bold ? 700 : 400)};
`

export const SSectionTitle = styled.h2`
  margin: 12px 0 12px;
  font-size: 13px;
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.08em;
`

export const SChartWrap = styled.div`
  width: 100%;
  height: 300px;
  margin-bottom: 28px;
`

export const SError = styled.div`
  margin-top: 16px;
  padding: 14px 16px;
  border-radius: 8px;
  background: #fff4f4;
  border: 1px solid #f3c9c9;
  color: #b91c1c;
  font-size: 14px;
`

export const SEmpty = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 56px 24px;
  gap: 10px;
  text-align: center;
`

export const SEmptyText = styled.p`
  margin: 0;
  font-size: 15px;
  color: #888780;
`

export const SEmptyHint = styled.p`
  margin: 0;
  font-size: 13px;
  color: #c0bfba;
`
