import styled from 'styled-components'

export const SPage = styled.main`
  padding: 0 24px 48px;
  max-width: 1100px;
  margin: 0 auto;
`

export const SNotificationBanner = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  margin: 16px 0;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 8px;
  font-size: 14px;
  color: #713f12;
`

export const SDismissBtn = styled.button`
  all: unset;
  cursor: pointer;
  font-size: 20px;
  color: #92400e;
  line-height: 1;
  flex-shrink: 0;

  &:hover {
    color: #713f12;
  }
`

export const SHeaderCard = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 28px;
  margin: 24px 0;
  background: white;
  border: 1px solid #e8e7e2;
  border-radius: 12px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
`

export const SUserInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`

export const SUserName = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SMeta = styled.p`
  margin: 0;
  font-size: 14px;
  color: #888780;
`

export const SRiskScoreBlock = styled.div`
  text-align: right;
`

export const SRiskLabel = styled.p`
  margin: 0 0 4px;
  font-size: 11px;
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.07em;
`

export const SRiskValue = styled.p<{ $score: number }>`
  margin: 0;
  font-size: 40px;
  font-weight: 800;
  line-height: 1;
  color: ${({ $score }) =>
    $score < 0.4 ? '#0F6E56' : $score < 0.7 ? '#D97706' : '#DC2626'};
`

export const SSectionTitle = styled.h2`
  margin: 32px 0 12px;
  font-size: 12px;
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.08em;
`

export const SEmpty = styled.p`
  color: #888780;
  font-size: 14px;
  padding: 8px 0;
`

export const SPlaceholderCard = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 200px;
  background: white;
  border: 1px dashed #d1d0ca;
  border-radius: 12px;
  color: #c0bfba;
  font-size: 14px;
  text-align: center;
`

export const SPlaceholderIcon = styled.div`
  font-size: 28px;
  opacity: 0.4;
  line-height: 1;
`

export const SError = styled.p`
  padding-top: 32px;
  color: #DC2626;
  font-size: 15px;
`
