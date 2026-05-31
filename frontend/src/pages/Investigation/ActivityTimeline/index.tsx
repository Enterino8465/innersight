import React from 'react'
import type { InvEvent } from '../../../store/slices/investigationSlice'
import { useActivityFilter, type EventFilter } from './hooks'
import {
  SFilterRow,
  SChip,
  SList,
  SItem,
  STimestamp,
  STypeTag,
  SDescription,
  SEmpty,
} from './styled'

interface Props {
  events: InvEvent[]
}

const FILTERS: EventFilter[] = ['all', 'logon', 'file', 'email', 'usb', 'http']

const LABELS: Record<EventFilter, string> = {
  all:   'All',
  logon: 'Logon',
  file:  'File',
  email: 'Email',
  usb:   'USB',
  http:  'Web',
}

const ActivityTimeline: React.FC<Props> = ({ events }) => {
  const { filter, setFilter, sorted } = useActivityFilter(events)

  return (
    <div>
      <SFilterRow>
        {FILTERS.map(f => (
          <SChip key={f} $active={filter === f} onClick={() => setFilter(f)}>
            {LABELS[f]}
          </SChip>
        ))}
      </SFilterRow>

      {sorted.length === 0 && <SEmpty>No events match this filter.</SEmpty>}

      {sorted.length > 0 && (
        <SList>
          {sorted.map((event, i) => (
            <SItem key={`${event.timestamp}-${event.type}-${i}`} $suspicious={event.suspicious}>
              <STimestamp>{event.timestamp.replace('T', ' ').slice(0, 19)}</STimestamp>
              <STypeTag $type={event.type}>{event.type}</STypeTag>
              <SDescription title={event.description}>{event.description}</SDescription>
            </SItem>
          ))}
        </SList>
      )}
    </div>
  )
}

export default ActivityTimeline
