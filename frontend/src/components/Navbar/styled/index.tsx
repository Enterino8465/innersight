import styled from 'styled-components'
import { NavLink } from 'react-router-dom'

export const SNav = styled.nav`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 56px;
  background: #3c3489;
  color: white;
`

export const SBrand = styled.span`
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.04em;
`

export const SLinks = styled.div`
  display: flex;
  gap: 24px;
`

export const SNavLink = styled(NavLink)`
  color: rgba(255, 255, 255, 0.65);
  text-decoration: none;
  font-size: 14px;
  font-weight: 400;
  padding-bottom: 2px;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;

  &:hover {
    color: white;
  }

  &.active {
    color: white;
    font-weight: 700;
    border-bottom-color: #9fe1cb;
  }
`
