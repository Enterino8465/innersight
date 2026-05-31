import React from 'react'
import {
  SPage,
  SPageHeader,
  SPageTitle,
  SCard,
  SCardTitle,
  SMetricsCard,
  SMetricsTitle,
  SImbalanceBanner,
  SBannerIcon,
} from './styled'
import { useTraining } from './hooks'
import TrainingConfig from './TrainingConfig'
import TrainingControls from './TrainingControls'
import MetricsChart from './MetricsChart'
import ConfusionMatrix from './ConfusionMatrix'

const TrainingPage: React.FC = () => {
  const { form, setField, errors, isValid, training, handleStart, handleCancel } = useTraining()
  const { status, epoch, totalEpochs, history, confusionMatrix, classImbalanceWarning } = training

  const isRunning = status === 'running'
  const showImbalanceWarning = classImbalanceWarning !== null && classImbalanceWarning.ratio < 0.05

  return (
    <SPage>
      <SPageHeader>
        <SPageTitle>Model Training</SPageTitle>
      </SPageHeader>

      <SCard>
        <SCardTitle>Hyperparameters</SCardTitle>
        <TrainingConfig
          epochs={form.epochs}
          batch_size={form.batch_size}
          lr={form.lr}
          layer_sizes={form.layer_sizes}
          pos_weight={form.pos_weight}
          patience={form.patience}
          disabled={isRunning}
          errors={errors}
          onChange={(key, value) => setField(key as keyof typeof form)(value)}
        />
      </SCard>

      <SCard>
        <TrainingControls
          status={status}
          epoch={epoch}
          totalEpochs={totalEpochs}
          canStart={isValid}
          onStart={handleStart}
          onCancel={handleCancel}
        />
      </SCard>

      {showImbalanceWarning && (
        <SImbalanceBanner>
          <SBannerIcon>⚠️</SBannerIcon>
          <span>
            Class imbalance detected — only{' '}
            <strong>{(classImbalanceWarning!.ratio * 100).toFixed(2)}%</strong> positive samples.
            Consider increasing <em>Pos Weight</em> or collecting more attack data.
          </span>
        </SImbalanceBanner>
      )}

      <SMetricsCard>
        <SMetricsTitle>Training Metrics</SMetricsTitle>
        <MetricsChart history={history} />
      </SMetricsCard>

      {status === 'done' && (
        <SCard>
          <SCardTitle>Confusion Matrix</SCardTitle>
          <ConfusionMatrix matrix={confusionMatrix} />
        </SCard>
      )}
    </SPage>
  )
}

export default TrainingPage
