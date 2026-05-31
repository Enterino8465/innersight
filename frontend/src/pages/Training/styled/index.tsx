import styled from 'styled-components'

export const SPage = styled.main`
  padding: 0 24px 48px;
  max-width: 960px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
`

export const SPageHeader = styled.div`
  padding: 28px 0 4px;
`

export const SPageTitle = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SCard = styled.section`
  background: white;
  border: 1px solid #e8e7e2;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
`

export const SCardTitle = styled.h2`
  margin: 0 0 20px;
  font-size: 13px;
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.07em;
`

export const SMetricsCard = styled.section`
  background: white;
  border: 2px solid #3c3489;
  border-radius: 14px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(60, 52, 137, 0.1);
`

export const SMetricsTitle = styled.h2`
  margin: 0 0 20px;
  font-size: 15px;
  font-weight: 700;
  color: #3c3489;
`

export const SImbalanceBanner = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 8px;
  font-size: 14px;
  color: #713f12;
`

export const SBannerIcon = styled.span`
  font-size: 16px;
  flex-shrink: 0;
`
