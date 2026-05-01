import React from 'react'
import { usePredictionPanel } from './hooks'
import { SColumns, SColumn, SColumnHeader, SValue, STrueLabel } from './styled'

const PredictionPanel = ({ trainingDone }) => {
  const { prediction, loading } = usePredictionPanel(trainingDone)

  if (!trainingDone) return null
  if (loading) return <p>Computing prediction...</p>
  if (!prediction) return null

  return (
    <>
      <SColumns>
        <SColumn>
          <SColumnHeader>Input</SColumnHeader>
          {prediction.input.map((v, i) => (
            <SValue key={i}>F{i}: {v.toFixed(4)}</SValue>
          ))}
        </SColumn>
        <SColumn>
          <SColumnHeader>Output</SColumnHeader>
          {prediction.output.map((v, i) => (
            <SValue key={i}>O{i}: {v.toFixed(4)}</SValue>
          ))}
        </SColumn>
      </SColumns>
      <STrueLabel>True label: {prediction.label.toFixed(4)}</STrueLabel>
    </>
  )
}

export default PredictionPanel
