import React from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'
import { SPlaceholder } from './styled'

const LossChart = ({ events = [], totalEpochs = 3 }) => {
  if (events.length === 0) {
    return <SPlaceholder>Run training to see loss</SPlaceholder>
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={events}>
        <CartesianGrid strokeDasharray="3 3" stroke="#d3d1c7" />
        <XAxis
          dataKey="epoch"
          type="number"
          domain={[1, totalEpochs]}
          allowDataOverflow
          tickCount={totalEpochs}
        />
        <YAxis domain={[0, 2]} />
        <Line
          dataKey="loss"
          stroke="#3C3489"
          dot={{ r: 4 }}
          type="linear"
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default LossChart
