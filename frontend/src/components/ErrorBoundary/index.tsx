import React from 'react'
import { SWrap, SHeading, SSubtext, SReloadBtn } from './styled'

interface Props {
  children: React.ReactNode
}

interface State {
  hasError: boolean
}

class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <SWrap>
          <SHeading>Something went wrong</SHeading>
          <SSubtext>An unexpected error occurred in the application.</SSubtext>
          <SReloadBtn onClick={() => window.location.reload()}>Reload page</SReloadBtn>
        </SWrap>
      )
    }
    return this.props.children
  }
}

export default ErrorBoundary
