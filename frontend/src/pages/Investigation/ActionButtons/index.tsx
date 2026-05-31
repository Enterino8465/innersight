import React, { useState, useEffect, useRef } from 'react'
import ConfirmDialog from '../ConfirmDialog'
import { SRow, SActionBtn, SBtnSpinner } from './styled'

type ActionKey = 'learn' | 'mute' | 'block'

interface Props {
  alertId: string | null
  actionInProgress: ActionKey | null
  onLearn: () => void
  onMute:  () => void
  onBlock: () => void
}

const CONFIG: Record<ActionKey, { title: string; message: string; confirmLabel: string; color: string; label: string }> = {
  learn: {
    title:        'Mark as Normal',
    message:      'This will fine-tune the model to treat this behaviour as normal. The alert will be marked as learned.',
    confirmLabel: 'Mark as Normal',
    color:        '#0F6E56',
    label:        'Mark as Normal',
  },
  mute: {
    title:        'Mute Incident',
    message:      'This alert will be muted and removed from the active list. You can re-open it at any time.',
    confirmLabel: 'Mute',
    color:        '#D97706',
    label:        'Mute Incident',
  },
  block: {
    title:        'Block Employee',
    message:      'This will initiate an access revocation request for this employee and log a block notification. This action cannot be undone automatically.',
    confirmLabel: 'Block Employee',
    color:        '#DC2626',
    label:        'Block Employee',
  },
}

const ActionButtons: React.FC<Props> = ({ alertId, actionInProgress, onLearn, onMute, onBlock }) => {
  const [pending,   setPending]   = useState<ActionKey | null>(null)
  const [succeeded, setSucceeded] = useState<ActionKey | null>(null)
  const prevInProgress = useRef<ActionKey | null>(null)

  useEffect(() => {
    const prev = prevInProgress.current
    prevInProgress.current = actionInProgress

    if (prev !== null && actionInProgress === null) {
      setSucceeded(prev)
      const id = setTimeout(() => setSucceeded(null), 1000)
      return () => clearTimeout(id)
    }
  }, [actionInProgress])

  const anyBusy = !alertId || actionInProgress !== null

  const btnLabel = (key: ActionKey) => {
    if (actionInProgress === key) return <><SBtnSpinner />Submitting…</>
    if (succeeded === key)        return '✓'
    return CONFIG[key].label
  }

  const handleConfirm = () => {
    if (pending === 'learn') onLearn()
    if (pending === 'mute')  onMute()
    if (pending === 'block') onBlock()
    setPending(null)
  }

  return (
    <>
      <SRow>
        <SActionBtn $color="#0F6E56" disabled={anyBusy} onClick={() => setPending('learn')}>
          {btnLabel('learn')}
        </SActionBtn>
        <SActionBtn $color="#D97706" disabled={anyBusy} onClick={() => setPending('mute')}>
          {btnLabel('mute')}
        </SActionBtn>
        <SActionBtn $color="#DC2626" disabled={anyBusy} onClick={() => setPending('block')}>
          {btnLabel('block')}
        </SActionBtn>
      </SRow>

      {pending && (
        <ConfirmDialog
          isOpen
          title={CONFIG[pending].title}
          message={CONFIG[pending].message}
          confirmLabel={CONFIG[pending].confirmLabel}
          confirmColor={CONFIG[pending].color}
          isLoading={actionInProgress !== null}
          onConfirm={handleConfirm}
          onCancel={() => setPending(null)}
        />
      )}
    </>
  )
}

export default ActionButtons
