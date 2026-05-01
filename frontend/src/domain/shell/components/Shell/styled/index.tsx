import styled, { createGlobalStyle } from 'styled-components'

export const SGlobalReset = createGlobalStyle`
  * { box-sizing: border-box; }
  body { margin: 0; font-family: Arial, sans-serif; background: #f5f5f0; }
`

export const SAppShell = styled.div`
  display: flex;
  flex-direction: row;
  gap: 16px;
  padding: 24px;
  min-height: 100vh;
`

export const SPanel = styled.div`
  flex: 1;
  background: white;
  border: 1px solid #d3d1c7;
  border-radius: 8px;
  padding: 20px;
`

export const SPanelTitle = styled.h2`
  font-size: 16px;
  font-weight: 500;
  margin: 0 0 12px;
  color: #3C3489;
`

