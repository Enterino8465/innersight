import styled, { keyframes } from 'styled-components'

const spin = keyframes`to { transform: rotate(360deg); }`

export const SBtnSpinner = styled.span`
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: white;
  border-radius: 50%;
  animation: ${spin} 0.65s linear infinite;
  vertical-align: middle;
  margin-right: 6px;
`

export const SRow = styled.div`
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
`

export const SActionBtn = styled.button<{ $color: string }>`
  padding: 10px 22px;
  border: none;
  border-radius: 8px;
  background: ${({ $color }) => $color};
  color: white;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: filter 0.15s;
  min-width: 148px;

  &:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
`
