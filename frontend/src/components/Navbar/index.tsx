import React from 'react'
import { SNav, SBrand, SLinks, SNavLink } from './styled'

const Navbar: React.FC = () => (
  <SNav>
    <SBrand>UEBA Console</SBrand>
    <SLinks>
      <SNavLink to="/alerts">Alerts</SNavLink>
      <SNavLink to="/employees">Employees</SNavLink>
      <SNavLink to="/suspects">Suspects</SNavLink>
      <SNavLink to="/models">Models</SNavLink>
      <SNavLink to="/training">Training</SNavLink>
    </SLinks>
  </SNav>
)

export default Navbar
