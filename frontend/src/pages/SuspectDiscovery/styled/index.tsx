import styled from 'styled-components'

export const SPage = styled.main`
  padding: 0 24px 40px;
  max-width: 1100px;
  margin: 0 auto;
`

export const SHeader = styled.div`
  padding: 28px 0 4px;
`

export const SHeaderTitle = styled.h1`
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1a1a2e;
`

export const SSubtitle = styled.p`
  margin: 6px 0 0;
  font-size: 14px;
  color: #888780;
`

export const SControls = styled.div`
  display: flex;
  gap: 14px;
  align-items: center;
  padding: 22px 0 24px;
  flex-wrap: wrap;
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
  min-width: 220px;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%23888780' d='M6 8L0 0h12z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;

  &:focus {
    outline: none;
    border-color: #3c3489;
  }
`

export const SToggle = styled.label`
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #1a1a2e;
  cursor: pointer;

  input {
    cursor: pointer;
  }
`

export const SButton = styled.button`
  height: 36px;
  padding: 0 18px;
  border: none;
  border-radius: 6px;
  background: #3c3489;
  color: white;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;

  &:hover {
    background: #2e2870;
  }

  &:disabled {
    background: #c0bfba;
    cursor: not-allowed;
  }
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
  font-weight: 700;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 2px solid #e8e7e2;
`

export const STr = styled.tr`
  &:nth-child(even) {
    background: #faf9f6;
  }
`

export const STd = styled.td`
  padding: 10px 12px;
  border-bottom: 1px solid #ededea;
  color: #1a1a2e;
  vertical-align: middle;
`

export const SSimBar = styled.div<{ $value: number }>`
  margin-top: 4px;
  height: 6px;
  border-radius: 3px;
  background: #9fe1cb;
  width: ${({ $value }) => Math.max(0, Math.min(1, $value)) * 100}%;
`

export const SError = styled.div`
  margin-top: 16px;
  padding: 14px 16px;
  border-radius: 8px;
  background: #fff4f4;
  border: 1px solid #f3c9c9;
  color: #b91c1c;
  font-size: 14px;
`

export const SEmpty = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 56px 24px;
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
