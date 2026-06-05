import React, { useMemo } from 'react'
import {
  forceSimulation,
  forceManyBody,
  forceLink,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from 'd3-force'
import { SWrap, SSvg, SLegend, SDot, SEmpty } from './styled'

export interface GraphNode {
  id: string
  type: string
  label: string
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
  color: string
}

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

type SimNode = GraphNode & SimulationNodeDatum
type SimLink = SimulationLinkDatum<SimNode> & { weight: number; color: string }

const W = 460
const H = 320

const NODE_COLOR: Record<string, string> = {
  user: '#3c3489',
  pc: '#2563eb',
  url: '#D97706',
  file: '#0F6E56',
}
const nodeColor = (type: string): string => NODE_COLOR[type] ?? '#888780'

const clampX = (v: number | undefined): number => Math.max(28, Math.min(W - 28, v ?? W / 2))
const clampY = (v: number | undefined): number => Math.max(28, Math.min(H - 28, v ?? H / 2))

const GraphNeighborhood: React.FC<Props> = ({ nodes, edges }) => {
  // Run the force layout to completion once (static, no animation cleanup needed).
  const layout = useMemo(() => {
    if (nodes.length === 0) {
      return { simNodes: [] as SimNode[], simLinks: [] as SimLink[] }
    }
    const simNodes: SimNode[] = nodes.map(n => ({ ...n }))
    const ids = new Set(simNodes.map(n => n.id))
    const simLinks: SimLink[] = edges
      .filter(e => ids.has(e.source) && ids.has(e.target))
      .map(e => ({ source: e.source, target: e.target, weight: e.weight, color: e.color }))

    const sim = forceSimulation<SimNode>(simNodes)
      .force('charge', forceManyBody().strength(-240))
      .force('link', forceLink<SimNode, SimLink>(simLinks).id(d => d.id).distance(80))
      .force('center', forceCenter(W / 2, H / 2))
      .force('collide', forceCollide(26))
      .stop()
    for (let i = 0; i < 300; i++) sim.tick()
    return { simNodes, simLinks }
  }, [nodes, edges])

  if (nodes.length === 0) {
    return <SEmpty>No connections to display.</SEmpty>
  }

  const centerId = nodes[0]?.id

  return (
    <SWrap>
      <SSvg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {layout.simLinks.map((link, i) => {
          const s = link.source as SimNode
          const t = link.target as SimNode
          return (
            <line
              key={i}
              x1={clampX(s.x)}
              y1={clampY(s.y)}
              x2={clampX(t.x)}
              y2={clampY(t.y)}
              stroke={link.color}
              strokeWidth={Math.min(4, 1 + link.weight * 0.25)}
              strokeOpacity={0.7}
            />
          )
        })}
        {layout.simNodes.map(node => {
          const isCenter = node.id === centerId
          const r = isCenter ? 16 : 9
          return (
            <g key={node.id} transform={`translate(${clampX(node.x)},${clampY(node.y)})`}>
              <circle
                r={r}
                fill={nodeColor(node.type)}
                stroke={isCenter ? '#1a1a2e' : 'white'}
                strokeWidth={isCenter ? 2 : 1.5}
              />
              <text x={0} y={r + 11} textAnchor="middle" fontSize={isCenter ? 11 : 9} fill="#1a1a2e">
                {node.label}
              </text>
            </g>
          )
        })}
      </SSvg>
      <SLegend>
        <span><SDot $color={nodeColor('user')} />User</span>
        <span><SDot $color={nodeColor('pc')} />PC</span>
        <span><SDot $color={nodeColor('url')} />URL</span>
        <span><SDot $color={nodeColor('file')} />File</span>
      </SLegend>
    </SWrap>
  )
}

export default GraphNeighborhood
