import { useState, useRef } from 'react'
import { postTrain, createEventSource } from '../api'

export const useTrainingControls = ({ onTrainingStart, onNewEvent, onTrainingDone } = {}) => {
  const [status, setStatus] = useState('idle')
  const [epoch, setEpoch] = useState(0)
  const [total, setTotal] = useState(3)
  const esRef = useRef(null)

  const start = async () => {
    onTrainingStart?.()
    setStatus('running')
    setEpoch(0)

    try {
      await postTrain({ epochs: 3, batch_size: 32, lr: 0.01, layer_sizes: [4, 8, 1] })

      const es = createEventSource()
      esRef.current = es

      es.onmessage = (e) => {
        const data = JSON.parse(e.data)
        if (data.status === 'done') {
          setStatus('done')
          es.close()
          onTrainingDone?.()
        } else {
          setEpoch(data.epoch)
          setTotal(data.total)
          onNewEvent?.(data)
        }
      }

      es.onerror = () => {
        es.close()
      }
    } catch {
      setStatus('idle')
    }
  }

  return { status, epoch, total, start }
}
