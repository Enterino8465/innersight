# Synthetic Demo Dataset

**This data is entirely synthetic.** It is *not* the CMU CERT Insider Threat
dataset and contains no real people, machines, or events. It exists only so that
InnerSight can boot into a working demo (`docker-compose up` with no environment
variables) without requiring the multi-gigabyte CERT download.

## Format

The files match the **CERT r4.2** layout exactly (the same format
`R4xAdapter` in `backend/innersight/data/adapters.py` expects), so the normal
pipeline (`load_version`, `compute_baselines`, scoring) runs unchanged:

```
synthetic_demo/
├── logon.csv          id,date,user,pc,activity
├── device.csv         id,date,user,pc,activity            (USB Connect/Disconnect)
├── email.csv          id,date,user,pc,to,cc,bcc,from,size,attachments,content
├── http.csv           id,date,user,pc,url,content
├── file.csv           id,date,user,pc,filename,content    (r4.2: every row = removable-media copy)
├── psychometric.csv   employee_name,user_id,O,C,E,A,N      (OCEAN scores)
├── LDAP/2010-01.csv   employee_name,user_id,email,role,department
└── answers/insiders.csv  dataset,scenario,details,user,start,end
```

## Contents

- **5 users** over **10 days** (2010-01-04 → 2010-01-13).
- **4 normal users** with consistent, boring business-hours activity.
- **1 insider — `DEMO_INS01`** (scenario 2), attacking on the **last 3 days**
  (2010-01-11 → 2010-01-13). Their attack pattern is deliberately detectable:
  after-hours logons (~23:00), after-hours USB connect/disconnect, job-search
  and cloud-upload web visits (`monster.com`, `dropbox.com`), a large external
  email attachment, and sensitive-file copies to removable media.

After `compute_baselines`, the insider's mean per-day deviation is roughly
**3× the normal users'** (≈1.0 vs ≈0.3, with peaks above 4σ), so the demo UI
surfaces a clear, explainable alert.
