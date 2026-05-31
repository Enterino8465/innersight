import styled from 'styled-components'
import { Link } from 'react-router-dom'

export const SWrap = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 56px);
  gap: 12px;
  text-align: center;
  padding: 32px 24px;
`

export const SCode = styled.div`
  font-size: 72px;
  font-weight: 800;
  color: #e8e7e2;
  line-height: 1;
`

export const SHeading = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SMessage = styled.p`
  margin: 0;
  font-size: 15px;
  color: #888780;
`

export const SHomeLink = styled(Link)`
  margin-top: 8px;
  padding: 10px 24px;
  background: #3c3489;
  color: white;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.15s;

  &:hover {
    background: #2e2870;
  }
`
