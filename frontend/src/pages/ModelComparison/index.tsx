import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import Spinner from '../../components/Spinner'
import { useModelComparison, type LadderRow } from './hooks'
import {
  SPage,
  SHeader,
  SHeaderTitle,
  SSubtitle,
  STable,
  STh,
  STr,
  STd,
  SSectionTitle,
  SChartWrap,
  SError,
  SEmpty,
  SEmptyText,
  SEmptyHint,
} from './styled'

const num = (v: number | null, digits = 3): string => (v != null ? v.toFixed(digits) : '—')

const auprcCell = (row: LadderRow): string =>
  row.auprc != null
    ? `${row.auprc.toFixed(3)}${row.auprc_std != null ? ` ± ${row.auprc_std.toFixed(3)}` : ''}`
    : '—'

const latencyCell = (v: number | null): string => (v != null ? `${v.toFixed(1)}d` : '—')

const ModelComparison: React.FC = () => {
  const { data, status, error } = useModelComparison()
  const ladder = data?.ladder ?? []
  const chartData = ladder
    .filter(r => r.auprc != null)
    .map(r => ({ model: r.model, auprc: r.auprc as number }))

  // Best (last completed) model that has per-scenario metrics, if any.
  const scenarioModel = Object.keys(data?.per_scenario ?? {}).slice(-1)[0]
  const scenarios = scenarioModel ? data!.per_scenario[scenarioModel] : undefined

  return (
    <SPage>
      <SHeader>
        <SHeaderTitle>Model Comparison</SHeaderTitle>
        <SSubtitle>
          The progressive ladder: each rung adds capability (handcrafted features → temporal CNN → graph context → static fusion).
        </SSubtitle>
      </SHeader>

      {status === 'loading' && <Spinner />}

      {status === 'failed' && <SError>Failed to load model comparison{error ? `: ${error}` : '.'}</SError>}

      {status === 'succeeded' && ladder.length === 0 && (
        <SEmpty>
          <SEmptyText>No model results yet</SEmptyText>
          <SEmptyHint>Run training (e.g. <code>make train-quick</code>) to populate the comparison.</SEmptyHint>
        </SEmpty>
      )}

      {status === 'succeeded' && ladder.length > 0 && (
        <>
          <STable>
            <thead>
              <tr>
                <STh>Model</STh>
                <STh>AUPRC ± std</STh>
                <STh>P@10</STh>
                <STh>P@20</STh>
                <STh>F1</STh>
                <STh>Detection Latency</STh>
              </tr>
            </thead>
            <tbody>
              {ladder.map((row, i) => (
                <STr key={row.model} $highlight={i === ladder.length - 1}>
                  <STd $bold>{row.model}</STd>
                  <STd>{auprcCell(row)}</STd>
                  <STd>{num(row.p_at_10, 2)}</STd>
                  <STd>{num(row.p_at_20, 2)}</STd>
                  <STd>{num(row.f1)}</STd>
                  <STd>{latencyCell(row.median_days)}</STd>
                </STr>
              ))}
            </tbody>
          </STable>

          {chartData.length > 0 && (
            <>
              <SSectionTitle>AUPRC by model</SSectionTitle>
              <SChartWrap>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e7e2" />
                    <XAxis dataKey="model" tick={{ fontSize: 11, fill: '#888780' }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#888780' }} tickCount={6} />
                    <Tooltip formatter={val => [Number(val).toFixed(4), 'AUPRC']} />
                    <Bar dataKey="auprc" fill="#3c3489" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </SChartWrap>
            </>
          )}

          {scenarios && scenarioModel && (
            <>
              <SSectionTitle>Per-scenario AUPRC — {scenarioModel}</SSectionTitle>
              <STable>
                <thead>
                  <tr>
                    <STh>Scenario</STh>
                    <STh>AUPRC</STh>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(scenarios).map(([scenario, metrics]) => (
                    <STr key={scenario}>
                      <STd $bold>Scenario {scenario}</STd>
                      <STd>{num(metrics.auprc)}</STd>
                    </STr>
                  ))}
                </tbody>
              </STable>
            </>
          )}
        </>
      )}
    </SPage>
  )
}

export default ModelComparison
