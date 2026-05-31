import styled from 'styled-components'

const TYPE_BG: Record<string, string> = {
  logon: '#3c3489',
  file:  '#D97706',
  email: '#0F6E56',
  usb:   '#DC2626',
  http:  '#6B7280',
}

export const SFilterRow = styled.div`
  display: flex;
  gap: 6px;
  margin-bottom: 14px;
  flex-wrap: wrap;
`

export const SChip = styled.button<{ $active: boolean }>`
  padding: 4px 14px;
  border-radius: 9999px;
  border: 1px solid ${({ $active }) => ($active ? '#3c3489' : '#d1d0ca')};
  background: ${({ $active }) => ($active ? '#3c3489' : 'white')};
  color: ${({ $active }) => ($active ? 'white' : '#888780')};
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;

  &:hover {
    border-color: #3c3489;
    color: ${({ $active }) => ($active ? 'white' : '#3c3489')};
  }
`

export const SList = styled.ul`
  list-style: none;
  margin: 0;
  padding: 0;
  background: white;
  border: 1px solid #e8e7e2;
  border-radius: 12px;
  overflow: hidden;
`

export const SItem = styled.li<{ $suspicious: boolean }>`
  display: grid;
  grid-template-columns: 168px 58px 1fr;
  gap: 14px;
  align-items: center;
  padding: 11px 16px;
  border-bottom: 1px solid #e8e7e2;
  border-left: 3px solid ${({ $suspicious }) => ($suspicious ? '#D97706' : 'transparent')};

  &:last-child {
    border-bottom: none;
  }
`

export const STimestamp = styled.span`
  font-size: 12px;
  color: #888780;
  font-family: monospace;
  white-space: nowrap;
`

export const STypeTag = styled.span<{ $type: string }>`
  display: inline-block;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: white;
  background: ${({ $type }) => TYPE_BG[$type] ?? '#6B7280'};
  white-space: nowrap;
`

export const SDescription = styled.span`
  font-size: 13px;
  color: #1a1a2e;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`

export const SEmpty = styled.p`
  color: #888780;
  font-size: 14px;
  padding: 8px 0;
`
