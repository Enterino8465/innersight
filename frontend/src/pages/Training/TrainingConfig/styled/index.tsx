import styled from 'styled-components'

export const SGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 16px;
`

export const SField = styled.label`
  display: flex;
  flex-direction: column;
  gap: 6px;
`

export const SLabel = styled.span`
  font-size: 12px;
  font-weight: 600;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`

export const SInput = styled.input<{ $hasError?: boolean }>`
  height: 36px;
  padding: 0 10px;
  border: 1px solid ${({ $hasError }) => $hasError ? '#DC2626' : '#d1d0ca'};
  border-radius: 6px;
  font-size: 14px;
  color: #1a1a2e;
  background: ${({ $hasError }) => $hasError ? '#fff5f5' : 'white'};
  width: 100%;
  box-sizing: border-box;
  transition: border-color 0.15s, background 0.15s;

  &:focus {
    outline: none;
    border-color: ${({ $hasError }) => $hasError ? '#DC2626' : '#3c3489'};
  }

  &:disabled {
    background: #f5f4f0;
    color: #c0bfba;
    cursor: not-allowed;
    border-color: #d1d0ca;
  }
`

export const SHint = styled.span`
  font-size: 11px;
  color: #c0bfba;
`

export const SError = styled.span`
  font-size: 11px;
  color: #DC2626;
  line-height: 1.3;
`
