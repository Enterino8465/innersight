import React from 'react'
import { SGrid, SField, SLabel, SInput, SHint, SError } from './styled'
import type { ValidationErrors } from '../hooks/useValidation'

interface Props {
  epochs: string
  batch_size: string
  lr: string
  layer_sizes: string
  pos_weight: string
  patience: string
  disabled: boolean
  errors: ValidationErrors
  onChange: (key: string, value: string) => void
}

const TrainingConfig: React.FC<Props> = ({
  epochs, batch_size, lr, layer_sizes, pos_weight, patience, disabled, errors, onChange,
}) => (
  <SGrid>
    <SField>
      <SLabel>Epochs</SLabel>
      <SInput
        type="number" min={1} max={500} step={1}
        value={epochs} disabled={disabled} $hasError={!!errors.epochs}
        onChange={e => onChange('epochs', e.target.value)}
      />
      {errors.epochs && <SError>{errors.epochs}</SError>}
    </SField>
    <SField>
      <SLabel>Batch Size</SLabel>
      <SInput
        type="number" min={1} step={1}
        value={batch_size} disabled={disabled} $hasError={!!errors.batch_size}
        onChange={e => onChange('batch_size', e.target.value)}
      />
      {errors.batch_size ? <SError>{errors.batch_size}</SError> : <SHint>16 · 32 · 64 · 128 · 256</SHint>}
    </SField>
    <SField>
      <SLabel>Learning Rate</SLabel>
      <SInput
        type="number" min={0.0001} max={1} step={0.0001}
        value={lr} disabled={disabled} $hasError={!!errors.lr}
        onChange={e => onChange('lr', e.target.value)}
      />
      {errors.lr && <SError>{errors.lr}</SError>}
    </SField>
    <SField>
      <SLabel>Layer Sizes</SLabel>
      <SInput
        type="text"
        value={layer_sizes} disabled={disabled} $hasError={!!errors.layer_sizes}
        onChange={e => onChange('layer_sizes', e.target.value)}
      />
      {errors.layer_sizes ? <SError>{errors.layer_sizes}</SError> : <SHint>comma-separated integers, last must be 1</SHint>}
    </SField>
    <SField>
      <SLabel>Pos Weight</SLabel>
      <SInput
        type="number" min={1} max={200} step={1}
        value={pos_weight} disabled={disabled} $hasError={!!errors.pos_weight}
        onChange={e => onChange('pos_weight', e.target.value)}
      />
      {errors.pos_weight && <SError>{errors.pos_weight}</SError>}
    </SField>
    <SField>
      <SLabel>Patience</SLabel>
      <SInput
        type="number" min={1} max={50} step={1}
        value={patience} disabled={disabled} $hasError={!!errors.patience}
        onChange={e => onChange('patience', e.target.value)}
      />
      {errors.patience && <SError>{errors.patience}</SError>}
    </SField>
  </SGrid>
)

export default TrainingConfig
