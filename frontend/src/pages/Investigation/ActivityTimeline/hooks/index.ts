import { useState, useMemo } from 'react'
import type { InvEvent } from '../../../../store/slices/investigationSlice'

export type EventFilter = 'all' | 'logon' | 'file' | 'email' | 'usb' | 'http'

export const useActivityFilter = (events: InvEvent[]) => {
  const [filter, setFilter] = useState<EventFilter>('all')

  const sorted = useMemo(
    () =>
      events
        .filter(e => filter === 'all' || e.type === filter)
        .slice()
        .sort((a, b) => b.timestamp.localeCompare(a.timestamp)),
    [events, filter],
  )

  return { filter, setFilter, sorted }
}
