import styled from 'styled-components'

export const SWrap = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  gap: 16px;
  font-family: inherit;
  color: #1a1a2e;
`

export const SHeading = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
`

export const SSubtext = styled.p`
  margin: 0;
  font-size: 14px;
  color: #888780;
`

export const SReloadBtn = styled.button`
  margin-top: 8px;
  padding: 10px 24px;
  background: #3c3489;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;

  &:hover {
    background: #2e2870;
  }
`
