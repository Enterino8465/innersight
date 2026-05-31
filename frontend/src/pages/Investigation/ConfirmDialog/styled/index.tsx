import styled from 'styled-components'

export const SOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`

export const SDialog = styled.div`
  background: white;
  border-radius: 14px;
  padding: 28px 32px 24px;
  max-width: 440px;
  width: 90%;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.25);
`

export const STitle = styled.h2`
  margin: 0 0 12px;
  font-size: 18px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SMessage = styled.p`
  margin: 0 0 24px;
  font-size: 14px;
  color: #5f5e5a;
  line-height: 1.65;
`

export const SFooter = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 10px;
`

export const SCancelBtn = styled.button`
  padding: 8px 20px;
  border: 1px solid #d1d0ca;
  border-radius: 6px;
  background: white;
  color: #5f5e5a;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.12s;

  &:hover:not(:disabled) {
    background: #f5f4f0;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

export const SConfirmBtn = styled.button<{ $color: string }>`
  padding: 8px 20px;
  border: none;
  border-radius: 6px;
  background: ${({ $color }) => $color};
  color: white;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: filter 0.12s;

  &:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`
