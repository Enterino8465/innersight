import React from 'react'
import { useTrainingControls } from './hooks'
import { SStartButton, SStatusLabel, SEpochCounter } from './styled'

const TrainingControls = ({ onTrainingStart, onNewEvent, onTrainingDone }) => {
  const { status, epoch, total, start } = useTrainingControls({ onTrainingStart, onNewEvent, onTrainingDone })

  return (
    <div>
      <SStartButton onClick={start} disabled={status === 'running'}>
        Start
      </SStartButton>
      <SStatusLabel $status={status}>{status}</SStatusLabel>
      <SEpochCounter>Epoch {epoch} / {total}</SEpochCounter>
    </div>
  )
}

export default TrainingControls
