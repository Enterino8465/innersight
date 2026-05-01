import React from 'react'
import { useDataPanel } from './hooks'

const DataPanel = () => {
  const { data, loading, error } = useDataPanel()

  if (loading) return <p>Loading data...</p>
  if (error) return <p className="error">Failed to load data</p>

  const [samples, features] = data.shape

  return (
    <>
      <h2>Data</h2>
      <p className="shape">{samples} samples, {features} features</p>
      <table>
        <thead>
          <tr>
            {Array.from({ length: features }, (_, i) => <th key={i}>F{i}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, rowIdx) => (
            <tr key={rowIdx}>
              {row.map((cell, colIdx) => <td key={colIdx}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

export default DataPanel
