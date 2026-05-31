import React from 'react'
import { createPortal } from 'react-dom'
import { SOverlay, SDialog, STitle, SMessage, SFooter, SCancelBtn, SConfirmBtn } from './styled'

interface Props {
  isOpen: boolean
  title: string
  message: string
  confirmLabel: string
  confirmColor: string
  isLoading: boolean
  onConfirm: () => void
  onCancel: () => void
}

const ConfirmDialog: React.FC<Props> = ({
  isOpen,
  title,
  message,
  confirmLabel,
  confirmColor,
  isLoading,
  onConfirm,
  onCancel,
}) => {
  if (!isOpen) return null

  return createPortal(
    <SOverlay onClick={onCancel}>
      <SDialog onClick={e => e.stopPropagation()}>
        <STitle>{title}</STitle>
        <SMessage>{message}</SMessage>
        <SFooter>
          <SCancelBtn onClick={onCancel} disabled={isLoading}>
            Cancel
          </SCancelBtn>
          <SConfirmBtn $color={confirmColor} onClick={onConfirm} disabled={isLoading}>
            {isLoading ? 'Processing…' : confirmLabel}
          </SConfirmBtn>
        </SFooter>
      </SDialog>
    </SOverlay>,
    document.body,
  )
}

export default ConfirmDialog
