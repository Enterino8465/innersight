import { useState, useMemo, useEffect } from 'react'
import { useAppDispatch, useAppSelector } from '../../../store/hooks'
import {
  startTraining,
  cancelTraining,
  closeStream,
  reset,
  type TrainingConfig,
} from '../../../store/slices/trainingSlice'
import { useValidation } from './useValidation'

interface FormValues {
  epochs: string
  batch_size: string
  lr: string
  layer_sizes: string
  pos_weight: string
  patience: string
}

const DEFAULTS: FormValues = {
  epochs:      '10',
  batch_size:  '256',
  lr:          '0.001',
  layer_sizes: '19, 64, 32, 1',
  pos_weight:  '50',
  patience:    '5',
}

export const useTraining = () => {
  const dispatch = useAppDispatch()
  const training = useAppSelector(s => s.training)
  const [form, setForm] = useState<FormValues>(DEFAULTS)

  useEffect(() => () => closeStream(), [])

  const setField = (key: keyof FormValues) => (value: string) =>
    setForm(prev => ({ ...prev, [key]: value }))

  const parsedConfig = useMemo((): TrainingConfig => ({
    epochs:      Math.max(1, parseInt(form.epochs, 10) || 10),
    batch_size:  Math.max(1, parseInt(form.batch_size, 10) || 256),
    lr:          parseFloat(form.lr) || 0.001,
    layer_sizes: form.layer_sizes
      .split(',')
      .map(s => parseInt(s.trim(), 10))
      .filter(n => !isNaN(n) && n > 0),
    pos_weight:  parseFloat(form.pos_weight) || 50,
    patience:    Math.max(1, parseInt(form.patience, 10) || 5),
  }), [form])

  const handleStart = () => {
    dispatch(reset())
    dispatch(startTraining(parsedConfig))
  }

  const handleCancel = () => dispatch(cancelTraining(undefined))

  const { errors, isValid } = useValidation(form)

  return {
    form,
    setField,
    errors,
    isValid,
    training,
    handleStart,
    handleCancel,
  }
}
