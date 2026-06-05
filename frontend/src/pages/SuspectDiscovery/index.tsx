import React from 'react'
import Spinner from '../../components/Spinner'
import { useSuspectDiscovery } from './hooks'
import {
  SPage,
  SHeader,
  SHeaderTitle,
  SSubtitle,
  SControls,
  SSelect,
  SToggle,
  SButton,
  STable,
  STh,
  STr,
  STd,
  SSimBar,
  SError,
  SEmpty,
  SEmptyText,
  SEmptyHint,
} from './styled'

const SuspectDiscovery: React.FC = () => {
  const {
    users, userId, setUserId,
    crossVersion, setCrossVersion,
    results, status, error, search,
  } = useSuspectDiscovery()

  return (
    <SPage>
      <SHeader>
        <SHeaderTitle>Suspect Discovery</SHeaderTitle>
        <SSubtitle>
          Find the users whose behavioural embeddings are nearest to a chosen user — related suspects often cluster together.
        </SSubtitle>
      </SHeader>

      <SControls>
        <SSelect value={userId} onChange={e => setUserId(e.target.value)}>
          <option value="">Select a user…</option>
          {users.map(u => (
            <option key={u} value={u}>{u}</option>
          ))}
        </SSelect>

        <SToggle>
          <input
            type="checkbox"
            checked={crossVersion}
            onChange={e => setCrossVersion(e.target.checked)}
          />
          Cross-version search
        </SToggle>

        <SButton onClick={() => search()} disabled={!userId || status === 'loading'}>
          Find similar users
        </SButton>
      </SControls>

      {status === 'loading' && <Spinner />}

      {status === 'failed' && error && <SError>{error}</SError>}

      {status === 'succeeded' && results.length === 0 && (
        <SEmpty>
          <SEmptyText>No similar users found</SEmptyText>
          <SEmptyHint>
            This user may not have a synced embedding yet — run the embedding sync to populate Qdrant.
          </SEmptyHint>
        </SEmpty>
      )}

      {status === 'succeeded' && results.length > 0 && (
        <STable>
          <thead>
            <tr>
              <STh>User</STh>
              <STh>Similarity</STh>
              <STh>Department</STh>
              <STh>Risk score</STh>
              <STh>Version</STh>
            </tr>
          </thead>
          <tbody>
            {results.map(r => (
              <STr key={`${r.version}:${r.user_id}`}>
                <STd>{r.user_id}</STd>
                <STd>
                  {r.similarity.toFixed(3)}
                  <SSimBar $value={r.similarity} />
                </STd>
                <STd>{r.department || '—'}</STd>
                <STd>{r.score != null ? r.score.toFixed(3) : '—'}</STd>
                <STd>{r.version}</STd>
              </STr>
            ))}
          </tbody>
        </STable>
      )}
    </SPage>
  )
}

export default SuspectDiscovery
