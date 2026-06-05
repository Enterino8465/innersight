import React from 'react'
import { useParams } from 'react-router-dom'
import Spinner from '../../components/Spinner'
import ScoreHistoryChart from './ScoreHistoryChart'
import ActivityTimeline from './ActivityTimeline'
import ActionButtons from './ActionButtons'
import DeviationHeatmap from '../../components/DeviationHeatmap'
import AttentionTimeline from '../../components/AttentionTimeline'
import GraphNeighborhood from '../../components/GraphNeighborhood'
import { useInvestigation, useUserVisualizations } from './hooks'
import {
  SPage,
  SNotificationBanner,
  SDismissBtn,
  SHeaderCard,
  SUserInfo,
  SUserName,
  SMeta,
  SRiskScoreBlock,
  SRiskLabel,
  SRiskValue,
  SSectionTitle,
  SEmpty,
  SPlaceholderCard,
  SPlaceholderIcon,
  SError,
  SCard,
  SSplit,
  SColumn,
  SFeatureList,
  SFeatureRow,
  SFeatureName,
  SFeatureTrack,
  SFeatureBar,
  SFeatureValue,
} from './styled'

const featureColor = (z: number): string =>
  Math.abs(z) < 2 ? '#0F6E56' : Math.abs(z) < 3 ? '#D97706' : '#DC2626'

const InvestigationPage: React.FC = () => {
  const { userId = '' } = useParams<{ userId: string }>()
  const {
    activity,
    scoreHistory,
    employeeMeta,
    currentAlert,
    status,
    actionInProgress,
    notification,
    handleLearn,
    handleMute,
    handleBlock,
    handleDismiss,
  } = useInvestigation(userId)

  const {
    deviations,
    attention,
    graph,
    topFeatures,
    loading: vizLoading,
    hasData,
  } = useUserVisualizations(userId)

  if (status === 'idle' || status === 'loading') {
    return (
      <SPage>
        <Spinner />
      </SPage>
    )
  }

  if (status === 'failed') {
    return (
      <SPage>
        <SError>Failed to load investigation data.</SError>
      </SPage>
    )
  }

  const score = employeeMeta?.latest_score ?? 0
  const alertCount = employeeMeta?.alert_count ?? 0

  return (
    <SPage>
      {notification && (
        <SNotificationBanner>
          <span>{notification}</span>
          <SDismissBtn onClick={handleDismiss} aria-label="Dismiss">×</SDismissBtn>
        </SNotificationBanner>
      )}

      <SHeaderCard>
        <SUserInfo>
          <SUserName>{employeeMeta?.name ?? userId}</SUserName>
          <SMeta>
            {employeeMeta?.department ?? 'Unknown'}
            {' · '}
            {employeeMeta?.role ?? 'N/A'}
            {' · '}
            {alertCount} alert{alertCount !== 1 ? 's' : ''}
          </SMeta>
        </SUserInfo>
        <SRiskScoreBlock>
          <SRiskLabel>Current Risk</SRiskLabel>
          <SRiskValue $score={score}>{score.toFixed(2)}</SRiskValue>
        </SRiskScoreBlock>
      </SHeaderCard>

      <SSectionTitle>30-Day Risk Score</SSectionTitle>
      {scoreHistory !== null ? (
        <ScoreHistoryChart data={scoreHistory} />
      ) : (
        <SPlaceholderCard>
          <SPlaceholderIcon>📈</SPlaceholderIcon>
          Score history not yet available
        </SPlaceholderCard>
      )}

      <SSectionTitle>Behavioural Deviations (z-score)</SSectionTitle>
      {vizLoading ? (
        <Spinner />
      ) : hasData && deviations ? (
        <SCard>
          <DeviationHeatmap
            matrix={deviations.matrix}
            featureNames={deviations.featureNames}
            dayLabels={deviations.dayLabels}
          />
        </SCard>
      ) : (
        <SPlaceholderCard>
          <SPlaceholderIcon>🔥</SPlaceholderIcon>
          No model trained yet — run training first
        </SPlaceholderCard>
      )}

      <SSectionTitle>Per-Feature Breakdown</SSectionTitle>
      {vizLoading ? (
        <Spinner />
      ) : topFeatures.length > 0 ? (
        <SCard>
          <SFeatureList>
            {topFeatures.map(f => (
              <SFeatureRow key={f.name}>
                <SFeatureName>{f.name.replace(/_/g, ' ')}</SFeatureName>
                <SFeatureTrack>
                  <SFeatureBar $ratio={Math.abs(f.z) / 8} $color={featureColor(f.z)} />
                </SFeatureTrack>
                <SFeatureValue>
                  {Math.abs(f.z).toFixed(1)}σ {f.z >= 0 ? 'above' : 'below'} normal
                </SFeatureValue>
              </SFeatureRow>
            ))}
          </SFeatureList>
        </SCard>
      ) : (
        <SEmpty>No anomalous features to highlight.</SEmpty>
      )}

      <SSectionTitle>Model Attention</SSectionTitle>
      {vizLoading ? (
        <Spinner />
      ) : attention && attention.length > 0 ? (
        <SCard>
          <AttentionTimeline attention={attention} />
        </SCard>
      ) : (
        <SPlaceholderCard>
          <SPlaceholderIcon>📊</SPlaceholderIcon>
          No model trained yet — run training first
        </SPlaceholderCard>
      )}

      <SSectionTitle>Actions</SSectionTitle>
      <ActionButtons
        alertId={currentAlert?.id ?? null}
        actionInProgress={actionInProgress}
        onLearn={handleLearn}
        onMute={handleMute}
        onBlock={handleBlock}
      />

      <SSplit>
        <SColumn>
          <SSectionTitle>Activity Timeline</SSectionTitle>
          {activity !== null ? (
            <ActivityTimeline events={activity} />
          ) : (
            <SEmpty>Activity data unavailable.</SEmpty>
          )}
        </SColumn>
        <SColumn>
          <SSectionTitle>Graph Neighbourhood</SSectionTitle>
          {vizLoading ? (
            <Spinner />
          ) : graph && graph.nodes.length > 0 ? (
            <SCard>
              <GraphNeighborhood nodes={graph.nodes} edges={graph.edges} />
            </SCard>
          ) : (
            <SPlaceholderCard>
              <SPlaceholderIcon>🕸️</SPlaceholderIcon>
              No connections to display
            </SPlaceholderCard>
          )}
        </SColumn>
      </SSplit>
    </SPage>
  )
}

export default InvestigationPage
