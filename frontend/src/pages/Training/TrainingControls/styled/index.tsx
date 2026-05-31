import styled from 'styled-components'

export const SRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
`

export const SStartBtn = styled.button`
  padding: 9px 24px;
  border: none;
  border-radius: 8px;
  background: #0F6E56;
  color: white;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: filter 0.15s;
  min-width: 80px;

  &:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
`

export const SCancelBtn = styled.button`
  padding: 9px 24px;
  border: 1px solid #DC2626;
  border-radius: 8px;
  background: white;
  color: #DC2626;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  min-width: 80px;

  &:hover:not(:disabled) {
    background: #DC2626;
    color: white;
  }

  &:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
`

const STATUS_BG: Record<string, string> = {
  idle:    '#e8e7e2',
  running: '#ede9fc',
  done:    '#d1fae5',
  failed:  '#fee2e2',
}

const STATUS_COLOR: Record<string, string> = {
  idle:    '#888780',
  running: '#3c3489',
  done:    '#0F6E56',
  failed:  '#DC2626',
}

export const SStatusPill = styled.span<{ $status: string }>`
  padding: 4px 14px;
  border-radius: 9999px;
  font-size: 13px;
  font-weight: 600;
  text-transform: capitalize;
  background: ${({ $status }) => STATUS_BG[$status] ?? STATUS_BG.idle};
  color: ${({ $status }) => STATUS_COLOR[$status] ?? STATUS_COLOR.idle};
`

export const SProgress = styled.span`
  font-size: 14px;
  color: #5f5e5a;
  font-family: monospace;
`
