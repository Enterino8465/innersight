import { useEffect } from 'react'
import { useAppDispatch, useAppSelector } from '../../../store/hooks'
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
