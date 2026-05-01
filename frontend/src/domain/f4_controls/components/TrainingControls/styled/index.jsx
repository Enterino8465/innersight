import styled from 'styled-components'

const STATUS_COLORS = {
  idle: '#888780',
  running: '#3C3489',
  done: '#0F6E56',
}

export const SStatusLabel = styled.p`
  margin: 8px 0;
  font-size: 14px;
  font-weight: 500;
  color: ${({ $status }) => STATUS_COLORS[$status] ?? STATUS_COLORS.idle};
  text-transform: capitalize;
`

export const SStartButton = styled.button`
  padding: 8px 20px;
  background: #3C3489;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;

  &:disabled {
    background: #c0bfba;
    cursor: not-allowed;
  }
`

export const SEpochCounter = styled.p`
  margin: 8px 0 0;
  font-size: 13px;
  color: #5F5E5A;
`
