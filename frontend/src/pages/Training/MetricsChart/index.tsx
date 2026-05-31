import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { HistoryEntry } from '../../../store/slices/trainingSlice'
import { SChartWrap, SPlaceholder } from './styled'

interface Props {
  history: HistoryEntry[]
}

const LINES = [
  { key: 'train_loss',    label: 'Train Loss',    color: '#3c3489', dash: ''      },
  { key: 'val_loss',      label: 'Val Loss',      color: '#3c3489', dash: '5 5'   },
  { key: 'val_f1',        label: 'Val F1',        color: '#0F6E56', dash: ''      },
  { key: 'val_recall',    label: 'Val Recall',    color: '#D97706', dash: ''      },
  { key: 'val_precision', label: 'Val Precision', color: '#2563EB', dash: ''      },
] as const

const MetricsChart: React.FC<Props> = ({ history }) => {
  if (history.length === 0) {
    return (
      <SChartWrap>
        <SPlaceholder>Metrics will appear as training progresses.</SPlaceholder>
      </SChartWrap>
    )
  }

  return (
    <SChartWrap>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={history} margin={{ top: 4, right: 16, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8e7e2" />
          <XAxis
            dataKey="epoch"
            tick={{ fontSize: 11, fill: '#888780' }}
            label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fontSize: 11, fill: '#888780' }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: '#888780' }}
            tickCount={6}
          />
          <Tooltip
            formatter={(val, name) => [Number(val).toFixed(4), name]}
            labelFormatter={(label) => `Epoch ${label}`}
          />
          <Legend verticalAlign="top" height={36} iconSize={12} />
          {LINES.map(({ key, label, color, dash }) => (
            <Line
              key={key}
              dataKey={key}
              name={label}
              stroke={color}
              strokeWidth={2}
              strokeDasharray={dash || undefined}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </SChartWrap>
  )
}

export default MetricsChart
