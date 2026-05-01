import React, { useState, useCallback } from 'react'
import DataPanel from '../../../f2_data_panel/components/DataPanel'
import NetworkDiagram from '../../../f3_diagram/components/NetworkDiagram'
import TrainingControls from '../../../f4_controls/components/TrainingControls'
import LossChart from '../../../f5_chart/components/LossChart'
import PredictionPanel from '../../../f6_prediction/components/PredictionPanel'
import { SGlobalReset, SAppShell, SPanel, SPanelTitle } from './styled'

const Shell: React.FC = () => {
  const [lossEvents, setLossEvents] = useState<{ epoch: number; loss: number }[]>([])
  const [totalEpochs, setTotalEpochs] = useState(3)
  const [trainingDone, setTrainingDone] = useState(false)

  const handleTrainingStart = useCallback(() => {
    setLossEvents([])
    setTrainingDone(false)
  }, [])

  const handleNewEvent = useCallback((data: { epoch: number; total: number; loss: number }) => {
    setTotalEpochs(data.total)
    setLossEvents(prev => [...prev, { epoch: data.epoch, loss: data.loss }])
  }, [])

  const handleTrainingDone = useCallback(() => setTrainingDone(true), [])

  return (
    <>
      <SGlobalReset />
      <SAppShell>
        <SPanel id="data-panel">
          <DataPanel />
        </SPanel>
        <SPanel id="network-panel">
          <SPanelTitle>Network</SPanelTitle>
          <NetworkDiagram />
        </SPanel>
        <SPanel id="training-panel">
          <SPanelTitle>Training</SPanelTitle>
          <TrainingControls
            onTrainingStart={handleTrainingStart}
            onNewEvent={handleNewEvent}
            onTrainingDone={handleTrainingDone}
          />
          <LossChart events={lossEvents} totalEpochs={totalEpochs} />
          <PredictionPanel trainingDone={trainingDone} />
        </SPanel>
      </SAppShell>
    </>
  )
}

export default Shell
