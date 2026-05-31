const VALID_BATCH_SIZES = new Set([16, 32, 64, 128, 256])

export interface FormValues {
  epochs: string
  batch_size: string
  lr: string
  layer_sizes: string
  pos_weight: string
  patience: string
}

export type ValidationErrors = Partial<Record<keyof FormValues, string>>

export const validate = (form: FormValues): ValidationErrors => {
  const errors: ValidationErrors = {}

  const epochsN = Number(form.epochs)
  if (!Number.isFinite(epochsN) || !Number.isInteger(epochsN) || epochsN < 1 || epochsN > 500)
    errors.epochs = 'Whole number between 1 and 500'

  const batchN = Number(form.batch_size)
  if (!VALID_BATCH_SIZES.has(batchN))
    errors.batch_size = 'Must be one of: 16, 32, 64, 128, 256'

  const lrN = Number(form.lr)
  if (!Number.isFinite(lrN) || lrN < 0.0001 || lrN > 1.0)
    errors.lr = 'Float between 0.0001 and 1.0'

  const parts = form.layer_sizes.split(',').map(s => parseInt(s.trim(), 10))
  if (parts.length < 2 || parts.some(n => isNaN(n) || n < 1))
    errors.layer_sizes = 'At least 2 comma-separated positive integers'
  else if (parts[parts.length - 1] !== 1)
    errors.layer_sizes = 'Last value must be 1 (output neuron)'

  const posN = Number(form.pos_weight)
  if (!Number.isFinite(posN) || posN < 1 || posN > 200)
    errors.pos_weight = 'Float between 1 and 200'

  const patN = Number(form.patience)
  if (!Number.isFinite(patN) || !Number.isInteger(patN) || patN < 1 || patN > 50)
    errors.patience = 'Whole number between 1 and 50'

  return errors
}

export const useValidation = (form: FormValues) => {
  const errors = validate(form)
  const isValid = Object.keys(errors).length === 0
  return { errors, isValid }
}
