import styled, { keyframes } from 'styled-components'

const spin = keyframes`
  to { transform: rotate(360deg); }
`

export const SSpin = styled.div`
  width: 32px;
  height: 32px;
  border: 3px solid #e8e7e2;
  border-top-color: #3c3489;
  border-radius: 50%;
  animation: ${spin} 0.7s linear infinite;
  margin: 64px auto;
`
