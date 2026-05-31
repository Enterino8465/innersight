import styled, { keyframes, css } from 'styled-components'

const pulse = keyframes`
  0%, 100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.5); }
  50%       { box-shadow: 0 0 0 5px rgba(255, 255, 255, 0);  }
`

export const SPage = styled.main`
  padding: 0 24px 40px;
  max-width: 1400px;
  margin: 0 auto;
`

export const SHeader = styled.div`
  padding: 28px 0 16px;
`

export const SHeaderTitle = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SFilterBar = styled.div`
  display: flex;
  gap: 10px;
  padding-bottom: 28px;
  flex-wrap: wrap;
`

export const SSearchInput = styled.input`
  height: 36px;
  padding: 0 12px;
  border: 1px solid #d1d0ca;
  border-radius: 6px;
  font-size: 14px;
  color: #1a1a2e;
  background: white;
  min-width: 200px;

  &::placeholder {
    color: #c0bfba;
  }

  &:focus {
    outline: none;
    border-color: #3c3489;
  }
`

export const SSelect = styled.select`
  height: 36px;
  padding: 0 28px 0 12px;
  border: 1px solid #d1d0ca;
  border-radius: 6px;
  font-size: 14px;
  color: #1a1a2e;
  background: white;
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%23888780' d='M6 8L0 0h12z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;

  &:focus {
    outline: none;
    border-color: #3c3489;
  }
`

export const SDeptSection = styled.section`
  margin-bottom: 32px;
`

export const SDeptLabel = styled.h2`
  margin: 0 0 12px;
  font-size: 13px;
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.08em;
`

export const SGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, 80px);
  gap: 8px;
`

export const SCell = styled.div<{ $score: number; $hasAlert: boolean }>`
  position: relative;
  width: 80px;
  height: 80px;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: flex-end;
  padding: 6px;
  overflow: visible;
  background: ${({ $score }) =>
    $score < 0.4 ? '#0F6E56' : $score < 0.7 ? '#D97706' : '#DC2626'};
  transition: filter 0.15s;

  ${({ $hasAlert }) =>
    $hasAlert &&
    css`
      border: 2px solid rgba(255, 255, 255, 0.55);
      animation: ${pulse} 2s ease-in-out infinite;
    `}

  &:hover {
    filter: brightness(1.12);
    z-index: 10;
  }

  &::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(26, 26, 46, 0.95);
    color: white;
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-line;
    padding: 8px 12px;
    border-radius: 6px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 200;
    min-width: 160px;
    text-align: left;
  }

  &:hover::after {
    opacity: 1;
  }
`

export const SCellLabel = styled.span`
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.92);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  width: 100%;
  display: block;
`

export const SEmpty = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 64px 24px;
  gap: 10px;
  text-align: center;
`

export const SEmptyText = styled.p`
  margin: 0;
  font-size: 15px;
  color: #888780;
`

export const SEmptyHint = styled.p`
  margin: 0;
  font-size: 13px;
  color: #c0bfba;
`

export const SClearBtn = styled.button`
  all: unset;
  cursor: pointer;
  margin-top: 4px;
  font-size: 13px;
  font-weight: 600;
  color: #3c3489;
  text-decoration: underline;
  text-underline-offset: 2px;

  &:hover {
    color: #2e2870;
  }
`

export const SError = styled.p`
  padding-top: 32px;
  color: #DC2626;
  font-size: 15px;
`
