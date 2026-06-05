import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ResponsiveContainer,
} from 'recharts'
import { SChartWrap, SEmpty } from './styled'

interface Props {
  /** One attention weight per day. */
  attention: number[]
  /** Optional [startDay, endDay] of the known attack window, shaded in red. */
  attackWindow?: [number, number]
}

const AttentionTimeline: React.FC<Props> = ({ attention, attackWindow }) => {
  if (attention.length === 0) {
    return <SEmpty>No attention data available.</SEmpty>
  }

  const data = attention.map((weight, day) => ({ day, weight }))

  return (
    <SChartWrap>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8e7e2" />
          <XAxis
            dataKey="day"
            tick={{ fontSize: 11, fill: '#888780' }}
            label={{ value: 'Day', position: 'insideBottom', offset: -2, fontSize: 11, fill: '#888780' }}
          />
          <YAxis tick={{ fontSize: 11, fill: '#888780' }} />
          <Tooltip
            formatter={val => [Number(val).toFixed(4), 'Attention']}
            labelFormatter={label => `Day ${label}`}
          />
          {attackWindow && (
            <ReferenceArea
              x1={attackWindow[0]}
              x2={attackWindow[1]}
              fill="#DC2626"
              fillOpacity={0.12}
              label={{ value: 'attack', position: 'insideTop', fontSize: 10, fill: '#DC2626' }}
            />
          )}
          <Bar dataKey="weight" fill="#3c3489" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </SChartWrap>
  )
}

export default AttentionTimeline
