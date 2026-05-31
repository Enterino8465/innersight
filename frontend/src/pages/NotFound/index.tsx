import React from 'react'
import { SWrap, SCode, SHeading, SMessage, SHomeLink } from './styled'

const NotFound: React.FC = () => (
  <SWrap>
    <SCode>404</SCode>
    <SHeading>Page not found</SHeading>
    <SMessage>The page you're looking for doesn't exist or has been moved.</SMessage>
    <SHomeLink to="/alerts">Back to Alerts</SHomeLink>
  </SWrap>
)

export default NotFound
