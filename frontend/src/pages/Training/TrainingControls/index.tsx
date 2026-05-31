import React from 'react'
import { SRow, SStartBtn, SCancelBtn, SStatusPill, SProgress } from './styled'

interface Props {
  status: 'idle' | 'running' | 'done' | 'failed'
  epoch: number
  totalEpochs: number
  canStart: boolean
  onStart: () => void
  onCancel: () => void
}

const TrainingControls: React.FC<Props> = ({ status, epoch, totalEpochs, canStart, onStart, onCancel }) => (
  <SRow>
    <SStartBtn onClick={onStart} disabled={status === 'running' || !canStart}>
      {status === 'running' ? 'Training…' : 'Start'}
    </SStartBtn>
    <SCancelBtn onClick={onCancel} disabled={status !== 'running'}>
      Cancel
    </SCancelBtn>
    <SStatusPill $status={status}>{status}</SStatusPill>
    {(status === 'running' || status === 'done') && (
      <SProgress>Epoch {epoch} / {totalEpochs}</SProgress>
    )}
  </SRow>
)

export default TrainingControls
