import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { ScorePoint } from '../../../store/slices/investigationSlice'
import { SChartWrap, SNoData } from './styled'

interface Props {
  data: ScorePoint[]
}

const ScoreHistoryChart: React.FC<Props> = ({ data }) => {
  if (data.length === 0) {
    return (
      <SChartWrap>
        <SNoData>No score history available — train the model first.</SNoData>
      </SChartWrap>
    )
  }

  return (
    <SChartWrap>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 16, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8e7e2" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: '#888780' }}
            tickFormatter={(val, idx) => (idx % 6 === 0 ? String(val).slice(5) : '')}
            interval={0}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: '#888780' }}
            tickCount={6}
          />
          <Tooltip
            formatter={(val) => [Number(val).toFixed(3), 'Risk score']}
            labelFormatter={(label) => `Date: ${label}`}
          />
          <ReferenceLine y={0.7} stroke="#DC2626" strokeDasharray="4 4" strokeOpacity={0.6} />
          <ReferenceLine y={0.4} stroke="#D97706" strokeDasharray="4 4" strokeOpacity={0.6} />
          <Line
            dataKey="score"
            stroke="#3c3489"
            strokeWidth={2}
            dot={{ r: 2, fill: '#3c3489' }}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </SChartWrap>
  )
}

export default ScoreHistoryChart
