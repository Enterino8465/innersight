import styled from 'styled-components'

const STATUS_BG: Record<string, string> = {
  open:    '#DC2626',
  muted:   '#D97706',
  learned: '#0F6E56',
  blocked: '#6B7280',
}

export const SPage = styled.main`
  padding: 0 24px 40px;
  max-width: 1200px;
  margin: 0 auto;
`

export const SHeader = styled.div`
  padding: 28px 0 20px;
`

export const SHeaderTitle = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SEmpty = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 64px 24px;
  gap: 10px;
  color: #888780;
  text-align: center;
`

export const SEmptyIcon = styled.div`
  font-size: 36px;
  opacity: 0.3;
  line-height: 1;
`

export const SEmptyText = styled.p`
  margin: 0;
  font-size: 15px;
  color: #888780;
`

export const SError = styled.p`
  padding-top: 32px;
  color: #DC2626;
  font-size: 15px;
`

export const STable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
`

export const STh = styled.th`
  text-align: left;
  padding: 10px 12px;
  font-size: 12px;
  font-weight: 600;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 2px solid #e8e7e2;
  white-space: nowrap;
`

export const SSortBtn = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  color: ${({ $active }) => ($active ? '#3c3489' : '#888780')};
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
  user-select: none;

  &:hover {
    color: #3c3489;
  }
`

export const STr = styled.tr`
  cursor: pointer;
  border-bottom: 1px solid #e8e7e2;

  td {
    padding: 14px 12px;
    color: #1a1a2e;
    vertical-align: middle;
  }

  &:hover {
    background: #f5f4f0;
  }

  &:last-child {
    border-bottom: none;
  }
`

export const SScoreBarWrap = styled.div`
  width: 80px;
  height: 8px;
  border-radius: 4px;
  background: #e8e7e2;
  overflow: hidden;
`

export const SScoreBarFill = styled.div<{ $score: number }>`
  height: 100%;
  border-radius: 4px;
  width: ${({ $score }) => Math.round($score * 100)}%;
  background: ${({ $score }) =>
    $score < 0.4 ? '#0F6E56' : $score < 0.7 ? '#D97706' : '#DC2626'};
`

export const SStatusBadge = styled.span<{ $status: string }>`
  display: inline-block;
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 600;
  color: white;
  background: ${({ $status }) => STATUS_BG[$status] ?? '#6B7280'};
  text-transform: capitalize;
`

export const SChevron = styled.span`
  display: block;
  font-size: 20px;
  color: #c0bfba;
  text-align: center;
  line-height: 1;
`
