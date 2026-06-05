import { useEffect, useMemo, useState } from 'react'
import { useAppDispatch, useAppSelector } from '../../../store/hooks'
import type { GraphNode, GraphEdge } from '../../../components/GraphNeighborhood'
import {
  fetchInvestigation,
  resetInvestigation,
  submitLearn,
  submitMute,
  submitBlock,
  clearNotification,
} from '../../../store/slices/investigationSlice'

export const useInvestigation = (userId: string) => {
  const dispatch = useAppDispatch()
  const state    = useAppSelector(s => s.investigation)

  useEffect(() => {
    dispatch(resetInvestigation())
    dispatch(fetchInvestigation(userId))
  }, [dispatch, userId])

  const handleLearn   = () => { if (state.currentAlert) dispatch(submitLearn(state.currentAlert.id)) }
  const handleMute    = () => { if (state.currentAlert) dispatch(submitMute(state.currentAlert.id)) }
  const handleBlock   = () => { if (state.currentAlert) dispatch(submitBlock(state.currentAlert.id)) }
  const handleDismiss = () => dispatch(clearNotification())

  return { ...state, handleLearn, handleMute, handleBlock, handleDismiss }
}

// ── Per-user visualisation data (deviations / attention / graph) ────────────

export interface DeviationData {
  matrix: number[][]
  featureNames: string[]
  dayLabels: string[]
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface TopFeature {
  name: string
  z: number
}

export const useUserVisualizations = (userId: string) => {
  const [deviations, setDeviations] = useState<DeviationData | null>(null)
  const [attention, setAttention] = useState<number[] | null>(null)
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!userId) return
    let active = true
    setLoading(true)
    setDeviations(null)
    setAttention(null)
    setGraph(null)

    const json = (url: string) =>
      fetch(url).then(r => (r.ok ? r.json() : null)).catch(() => null)

    const base = `/api/users/${encodeURIComponent(userId)}`
    Promise.all([json(`${base}/deviations`), json(`${base}/attention`), json(`${base}/graph`)])
      .then(([dev, att, gph]) => {
        if (!active) return
        if (dev?.matrix?.length) {
          setDeviations({ matrix: dev.matrix, featureNames: dev.feature_names, dayLabels: dev.day_labels })
        }
        if (att?.attention?.length) setAttention(att.attention)
        if (gph?.nodes?.length) setGraph({ nodes: gph.nodes, edges: gph.edges })
        setLoading(false)
      })

    return () => { active = false }
  }, [userId])

  // Top-5 most anomalous features by peak |z| (signed peak kept for direction).
  const topFeatures = useMemo<TopFeature[]>(() => {
    if (!deviations || deviations.matrix.length === 0) return []
    const peaks = deviations.featureNames.map((name, fi) => {
      let z = 0
      for (const day of deviations.matrix) {
        const v = day[fi] ?? 0
        if (Math.abs(v) > Math.abs(z)) z = v
      }
      return { name, z }
    })
    return peaks.sort((a, b) => Math.abs(b.z) - Math.abs(a.z)).slice(0, 5)
  }, [deviations])

  const hasData = deviations !== null && deviations.matrix.length > 0

  return { deviations, attention, graph, topFeatures, loading, hasData }
}
