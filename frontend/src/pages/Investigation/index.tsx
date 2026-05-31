import React from 'react'
import { useParams } from 'react-router-dom'
import Spinner from '../../components/Spinner'
import ScoreHistoryChart from './ScoreHistoryChart'
import ActivityTimeline from './ActivityTimeline'
import ActionButtons from './ActionButtons'
import { useInvestigation } from './hooks'
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
} from './styled'

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

      <SSectionTitle>Actions</SSectionTitle>
      <ActionButtons
        alertId={currentAlert?.id ?? null}
        actionInProgress={actionInProgress}
        onLearn={handleLearn}
        onMute={handleMute}
        onBlock={handleBlock}
      />

      <SSectionTitle>Activity Timeline</SSectionTitle>
      {activity !== null ? (
        <ActivityTimeline events={activity} />
      ) : (
        <SEmpty>Activity data unavailable.</SEmpty>
      )}
    </SPage>
  )
}

export default InvestigationPage
