# CERT Insider Threat Dataset — Complete Reference

> **Purpose:** This document contains every detail about the CMU CERT dataset needed
> to build the InnerSight pipeline. No future conversation should need to access the
> external hard drive. If something is missing from this doc, it's a bug in the doc.

**Dataset location:** `/Volumes/dataset hard drive/dataset/`

---

## 1. Versions at a Glance

| Version | Users | Insiders | Scenarios | Size (approx) | Log types available |
|---------|-------|----------|-----------|---------------|---------------------|
| r1 | 1,000 | 0 | 0 | ~0.5 GB | logon, device, http, LDAP |
| r2 | 1,000 | 1 | 1 | ~1.5 GB | logon, device, email, http, LDAP, psychometric |
| r3.1 | 1,000 | 2 | 2 | ~2 GB | logon, device, email, http, file, LDAP, psychometric |
| r3.2 | 1,000 | 2 | 2 | ~2 GB | logon, device, email, http, file, LDAP, psychometric |
| r4.1 | 1,000 | 3 | 3 | ~8 GB | logon, device, email, http, file, LDAP, psychometric |
| r4.2 | ~1,000 | 70 | 3 | ~16 GB | logon, device, email, http, file, LDAP, psychometric |
| r5.1 | ~2,000 | 4 | 4 | ~19 GB | logon, device, email, http, file, LDAP, psychometric, decoy_file |
| r5.2 | ~2,000 | 99 | 4 | ~38 GB | logon, device, email, http, file, LDAP, psychometric, decoy_file |
| r6.1 | ~4,000 | 5 | 5 | ~47 GB | logon, device, email, http, file, LDAP, psychometric, decoy_file |
| r6.2 | ~4,000 | 5 | 5 | ~94 GB | logon, device, email, http, file, LDAP, psychometric, decoy_file |

**Primary training set:** r4.2 (dense needles — 70 insiders across 3 scenarios)
**Largest trainable:** r5.2 (99 insiders across 4 scenarios)
**Test-only:** r6.2 (only 5 insiders — too few to train on, good for generalization test)

---

## 2. The 5 Attack Scenarios

| # | Name | Duration | Behavior pattern | Present in |
|---|------|----------|-----------------|------------|
| 1 | After-hours USB + WikiLeaks upload | 1-2 days | After-hours logon → USB connect → browse wikileaks.org → USB disconnect → logoff. Quick, 2-6 events per incident. | r2, r3+, r4+, r5+, r6+ |
| 2 | Job hunting → resignation → USB theft | 3-8 weeks | Visits job sites (monster.com, careerbuilder.com, craigslist.org/jobs) → eventually escalates to USB data exfiltration before leaving. Gradual behavioral shift. | r3+, r4+, r5+, r6+ |
| 3 | Sysadmin keylogger on another user's machine | 1-2 days | Searches for keylogger software → downloads exe to USB → logs into ANOTHER user's dedicated PC → installs keylogger. Cross-machine access is the signal. | r4+, r5+, r6+ |
| 4 | Cross-account file snooping + email exfil | 2-5 months | Accesses other users' files over extended period → exfiltrates via email. Long, slow burn. | r5+, r6+ |
| 5 | Post-layoff Dropbox exfil | Minutes | After termination, quickly uploads files to Dropbox. Very short window. | r6+ |

---

## 3. Exact CSV Schemas Per Version

### 3.1 logon.csv

**Identical across all versions:**
```
id,date,user,pc,activity
```
- `id`: Unique event ID in `{XXXX-XXXXXXXXXX-XXXXXXXXXX}` format
- `date`: Timestamp, format `MM/DD/YYYY HH:MM:SS` (zero-padded in r3+; NOT zero-padded in r2: `1/4/2010 8:47:00`)
- `user`: User ID (e.g., `NGF0157`). **EXCEPTION: r1 uses `DTAA/` prefix** (e.g., `DTAA/KEE0997`)
- `pc`: Machine ID (e.g., `PC-6056`)
- `activity`: `Logon` or `Logoff`

**Row counts (approximate):** r1: 849K, r2: 840K, r4.2: ~1M, r5.2: ~2M, r6.2: ~4M

**Sample (r4.2):**
```
{X1D9-S0ES98JV-5357PWMI},01/02/2010 06:49:00,NGF0157,PC-6056,Logon
```

### 3.2 device.csv

**Three different schemas exist:**

**r1:** `id,date,user,pc,activity` (activity = `Connect`/`Disconnect`; user has `DTAA/` prefix)

**r2:** `id,date,user,pc,activity` (activity = **`Insert`/`Remove`** — different from all other versions!)

**r3.x, r4.x:** `id,date,user,pc,activity` (activity = `Connect`/`Disconnect`)

**r5.x, r6.x:** `id,date,user,pc,file_tree,activity` (adds `file_tree` column)
- `file_tree`: Semicolon-delimited list of directories on the USB device, e.g., `R:\;R:\HRE1950;R:\47yHBn0`
- Empty on `Disconnect` events

**Sample (r5.2):**
```
{C9S1-Y8GB42VD-2923GATU},01/02/2010 07:27:19,HRE1950,PC-8025,R:\;R:\HRE1950;R:\47yHBn0;R:\54s7J45,Connect
```

**Row counts (approx):** r1: 65K, r2: 712K, r4.2: ~100K, r5.2: ~200K

### 3.3 email.csv

**Five different schemas exist:**

**r1:** NO EMAIL FILE EXISTS.

**r2:** `id,date,to,from` — extremely minimal. No user, no pc, no cc/bcc, no size, no content. 1.19M rows. Also has `email-supplemental.csv.gz` (7.9 GB compressed).

**r3.x:** `id,date,to,from,size,attachments,content` — still no user or pc column! No cc/bcc.
- `attachments`: integer count
- `content`: bag-of-words keywords

**r4.x:** `id,date,user,pc,to,cc,bcc,from,size,attachments,content` — full schema, first version with user/pc/cc/bcc
- `attachments`: integer count
- `to`/`cc`/`bcc`: semicolon-separated email addresses
- `content`: bag-of-words keywords
- Internal addresses: `Firstname.Middle.Lastname@dtaa.com`
- External addresses: various domains (`@cox.net`, `@netzero.com`, `@lockheedmartin.com`, etc.)

**r5.x:** `id,date,user,pc,to,cc,bcc,from,activity,size,attachments,content` — adds `activity` column
- `activity`: `Send` or `Receive`
- `attachments`: now **filenames** (e.g., `R:\file.doc`), NOT integer count

**r6.x:** `id,date,user,pc,to,cc,bcc,from,activity,size,attachments,content` — same as r5.x
- `activity`: `Send` or `View` (NOT `Receive` — changed from r5)
- `content`: full sentences (not bag-of-words)

**Sample (r4.2):**
```
{R3I7-S4TX96FG-8219JWFF},01/02/2010 07:11:45,LAP0338,PC-5758,Dean.Flynn.Hines@dtaa.com;Wade_Harrison@lockheedmartin.com,Nathaniel.Hunter.Heath@dtaa.com,,Lynn.Adena.Pratt@dtaa.com,25830,0,middle f2 systems 4 july...
```

### 3.4 http.csv

**Three different schemas exist:**

**r1:** NO HEADER ROW! Columns are implicitly `id,date,user,pc,url` (5 columns). User has `DTAA/` prefix. No `content` column. 3.45M rows.

**r2 through r5.x:** `id,date,user,pc,url,content`
- `content`: space-separated topic keywords

**r6.x:** `id,date,user,pc,url,activity,content` — adds `activity` column
- `activity`: `WWW Visit`, `WWW Upload`, or `WWW Download`
- `content`: full sentences (r6.x) not bag-of-words

**Sample (r4.2):**
```
{V1Y4-S2IR20QU-6154HFXJ},01/02/2010 06:55:16,LRR0148,PC-4275,http://msn.com/The_Human_Centipede/katsuro/arjf309875127.htm,remain representatives...
```

**Row counts:** r1: 3.45M, r2: 7.32M, r4.2: ~10M, r6.2: ~85M (HUGE — must stream with chunks)

### 3.5 file.csv

**Three different schemas exist:**

**r1, r2:** NO FILE LOG EXISTS.

**r3.x, r4.x:** `id,date,user,pc,filename,content`
- Every row IS a copy to removable media (implicit — no explicit flag)
- `content`: hex file header + bag-of-words keywords
- File headers: `D0-CF-11-E0-A1-B1-1A-E1` (MS Office), `25-50-44-46-2D` (PDF), `4D-5A-90-00-...` (EXE), `50-4B-03-04-14` (ZIP)

**r5.x:** `id,date,user,pc,filename,activity,to_removable_media,from_removable_media,content`
- `activity`: `File Open`, `File Write`, `File Copy`, `File Delete` (r5 readme says only copy, but actual data has all four)
- `to_removable_media`: `True`/`False`
- `from_removable_media`: `True`/`False`

**r6.x:** Same schema as r5.x
- `content`: full sentences (not bag-of-words)

**Sample (r4.2):**
```
{L9G8-J9QE34VM-2834VDPB},01/02/2010 07:23:14,MOH0273,PC-6699,EYPC9Y08.doc,D0-CF-11-E0-A1-B1-1A-E1 during difficulty overall...
```

**Sample (r6.2):**
```
{F3E2-X3MV05YQ-3516SZDT},01/02/2010 07:19:41,SDH2394,PC-5849,R:\60WBQE7S.doc,File Open,False,True,"D0-CF-11-E0-A1-B1-1A-E1 Ernesztin's brother, Lipot Hoffmann..."
```

### 3.6 LDAP/ (directory of monthly CSVs)

**All versions have 18 monthly snapshots:** `2009-12.csv` through `2011-05.csv`

**Three different schemas exist:**

**r1:** `employee_name,user_id,Domain,Email,Role`
- Note: column names are capitalized (`Domain`, `Email`, `Role`)
- `Domain`: always `dtaa.com`
- No org hierarchy at all

**r2:** `employee_name,user_id,email,role`
- Lowercase column names
- Still no org hierarchy

**r3.x, r4.x:** `employee_name,user_id,email,role,business_unit,functional_unit,department,team,supervisor`
- Full org hierarchy
- `business_unit`: always `1`
- `functional_unit`: e.g., `2 - ResearchAndEngineering`, `5 - SalesAndMarketing`
- `department`: e.g., `2 - SoftwareManagement`, `3 - Assembly`
- `team`: e.g., `3 - Software`, `6 - AssemblyDept` (sometimes empty for Directors)
- `supervisor`: full name of direct supervisor

**r5.x, r6.x:** `employee_name,user_id,email,role,projects,business_unit,functional_unit,department,team,supervisor`
- Adds `projects` column between `role` and `business_unit`
- `projects`: usually empty, occasionally one project name

**Roles observed:** ITAdmin, ComputerProgrammer, SoftwareEngineer, ElectricalEngineer, MechanicalEngineer, SystemsEngineer, MaterialsEngineer, Mathematician, ComputerScientist, Technician, Salesman, ProductionLineWorker, AdministrativeAssistant, SecurityGuard, PurchasingClerk, Director, Manager, VP

**User count per LDAP snapshot:** Decreases over time as employees are terminated. r4.2 Jan 2010: ~1000 users. r5.2 Jan 2010: ~2000 users. r6.2 Jan 2010: ~4000 users.

**Sample (r4.2):**
```
Calvin Edan Love,CEL0561,Calvin.Edan.Love@dtaa.com,ComputerProgrammer,1,2 - ResearchAndEngineering,2 - SoftwareManagement,3 - Software,Stephanie Briar Harrington
```

### 3.7 psychometric.csv

**Identical across all versions that have it (r2+):**
```
employee_name,user_id,O,C,E,A,N
```
- O, C, E, A, N: Big Five personality scores (integers, roughly 10-50 range)
- O = Openness, C = Conscientiousness, E = Extroversion, A = Agreeableness, N = Neuroticism
- **E (Extroversion) drives number of social connections**
- **C (Conscientiousness) drives punctuality / late arrivals**
- One row per user, static (doesn't change over time)

**r1 has NO psychometric file.**

**Sample:**
```
Calvin Edan Love,CEL0561,40,39,36,19,40
```

### 3.8 decoy_file.csv

**Only in r5.x and r6.x.**

```
decoy_filename,pc
```
(r6.x has quoted fields: `"decoy_filename","pc"`)

Lists decoy files placed on specific machines. Filenames are NOT globally unique — the same name can appear on different PCs.

**Sample (r5.2):**
```
C:\46GCY91\9JWV9YSV.doc,PC-8053
```

---

## 4. Answers Directory

**Location:** `/Volumes/dataset hard drive/dataset/answers/`

### 4.1 insiders.csv (Master File)

```
dataset,scenario,details,user,start,end
```
- `dataset`: version number (e.g., `4.2`, `5.2`)
- `scenario`: integer 1-5
- `details`: filename of the per-insider observables CSV
- `user`: insider's user_id
- `start`: attack start timestamp (`M/D/YYYY H:MM:SS`)
- `end`: attack end timestamp

**Insider counts from insiders.csv:**
- r2: 1 insider (scenario 1)
- r3.1: 2 insiders (scenarios 1, 2)
- r3.2: 2 insiders (scenarios 1, 2)
- r4.1: 3 insiders (scenarios 1, 2, 3)
- r4.2: **70 insiders** (30× scenario 1, 30× scenario 2, 10× scenario 3)
- r5.1: 4 insiders (scenarios 1, 2, 3, 4)
- r5.2: **99 insiders** (29× scenario 1, 29× scenario 2, 10× scenario 3, 31× scenario 4)
- r6.1: 5 insiders (1 per scenario)
- r6.2: 5 insiders (1 per scenario)

### 4.2 Per-Insider Observables Files

**These are NOT proper CSVs.** Rows are variable-length with the data type as the first column. They interleave different log types chronologically.

**Format for r2, r3.x, r4.1 (flat files):**
```
logon,{ID},date,user,pc,activity
device,{ID},date,user,pc,activity
http,{ID},date,user,pc,url[,content]
email,{ID},date,user,pc,to,[cc],[bcc],from,size,attachments,content
file,{ID},date,user,pc,filename,content
```

**Format for r4.2, r5.2 (per-insider subdirectories):**
Same row format, but organized in subdirectories by scenario:
- `answers/r4.2-1/r4.2-1-AAM0658.csv` (scenario 1, user AAM0658)
- `answers/r4.2-2/r4.2-2-AAF0535.csv` (scenario 2, user AAF0535)
- `answers/r4.2-3/r4.2-3-CSC0217.csv` (scenario 3, user CSC0217)
- `answers/r5.2-1/r5.2-1-ALT1465.csv` (etc.)
- `answers/r5.2-4/r5.2-4-ACM1770.csv` (scenario 4)

**Format for r6.x (flat files):**
Same variable-length format but with quoted fields and version-specific columns (file_tree, activity on file/http).

**Sample scenario 1 (USB + WikiLeaks, r4.2):**
```
logon,{K3V4-Y4OK65SI-1583GEOQ},10/23/2010 01:34:19,AAM0658,PC-9923,Logon
device,{H1L0-X7RH83FI-5967VUQY},10/23/2010 06:18:48,AAM0658,PC-9923,Connect
http,{Y4Q9-U5VQ11UG-1279ZPTL},10/23/2010 06:26:01,AAM0658,PC-9923,http://wikileaks.org/...,spy bait covert...
device,{W7C0-G1SW41KB-1991BUTL},10/23/2010 06:26:48,AAM0658,PC-9923,Disconnect
logon,{C7T3-B4IY16JF-6945LOFC},10/23/2010 06:28:17,AAM0658,PC-9923,Logoff
```

**Sample scenario 2 (job hunting, r4.2):**
```
http,{O3A5-L8JQ13UN-2649PAKV},06/28/2010 08:51:08,AAF0535,PC-2408,http://monster.com/WboUhagvat...,experience platform resume...
http,{A5W0-L2MU52WY-3024MWHD},06/28/2010 10:33:01,AAF0535,PC-2408,http://craigslist.org/WboUhagvat...,people passion resume...
http,...,http://jobhuntersbible.com/WboUhagvat...,passion responsibilities resume...
```

**Sample scenario 3 (keylogger, r4.2):**
```
email,...,CSC0217,...,fed up i work after-hours i may leave...
http,...,CSC0217,...,http://www.refog.com/free-keylogger/...,covert download keylogger stealth...
device,...,CSC0217,...,Connect
file,...,CSC0217,...,6UQIYOYG.exe,4D-5A-90-00-... stealth surveillance keylogging...
device,...,CSC0217,...,Disconnect
```

---

## 5. Critical Quirks the Adapter Must Handle

### 5.1 User ID Format
- **r1 ONLY:** User IDs are prefixed with `DTAA/` (e.g., `DTAA/KEE0997`). All other versions use bare IDs (e.g., `KEE0997`).
- The adapter must strip `DTAA/` to normalize.

### 5.2 Date Format
- **r2:** Dates are NOT zero-padded: `1/4/2010 8:47:00`
- **All others:** Zero-padded: `01/04/2010 08:47:00`
- `pd.to_datetime()` handles both transparently.

### 5.3 Device Activity Names
- **r2:** Uses `Insert`/`Remove` instead of `Connect`/`Disconnect`
- **All others:** `Connect`/`Disconnect`
- The adapter must normalize `Insert` → `Connect`, `Remove` → `Disconnect`.

### 5.4 HTTP Has No Header in r1
- r1's `http.csv` starts directly with data, no header row.
- Columns are: `id, date, user, pc, url` (5 columns, no content)
- The adapter must supply column names manually.

### 5.5 Email Schema Drift
- **r1:** No email file
- **r2:** Only `id,date,to,from` — no user, no size, no content
- **r3.x:** Adds size, attachments, content — but still no `user` or `pc` column! Sender is in `from` as email address; must be mapped back to user_id via LDAP.
- **r4.x:** First version with full schema including `user`, `pc`, `cc`, `bcc`
- **r5.x:** Adds `activity` column (Send/Receive). `attachments` changes from count to filenames.
- **r6.x:** `activity` changes from Receive to `View`

### 5.6 File Schema Drift
- **r1, r2:** No file log
- **r3.x, r4.x:** Every row IS a removable-media copy (implicit). No activity/to_removable/from_removable columns.
- **r5.x, r6.x:** Explicit `activity`, `to_removable_media`, `from_removable_media` columns. File operations include Open, Write, Copy, Delete — not just copies.

### 5.7 LDAP Column Capitalization
- **r1:** `employee_name,user_id,Domain,Email,Role` (capitals)
- **r2+:** `employee_name,user_id,email,role,...` (lowercase)

### 5.8 Content Format
- **r1-r5:** Bag-of-words (space-separated keywords)
- **r6:** Full natural-language sentences

### 5.9 File Content Hex Headers
All file.csv `content` fields start with a hex-encoded file header:
- `D0-CF-11-E0-A1-B1-1A-E1` = MS Office (.doc, .xls, .ppt)
- `25-50-44-46-2D` = PDF
- `4D-5A-90-00-...` = Windows EXE
- `50-4B-03-04-14` = ZIP
- `FF-D8` = JPEG
- `53-53-42-38` or similar = other formats

### 5.10 r2 email-supplemental.csv.gz
r2 has a 7.9 GB compressed file `email-supplemental.csv.gz` that is NOT needed for the pipeline. It contains expanded email data. Ignore it.

### 5.11 Missing Files Per Version
| Version | Missing logs |
|---------|-------------|
| r1 | email, file, psychometric |
| r2 | file |
| r3.1 | *(has everything for its era)* |
| r3.2 | *(has everything — earlier note about missing email/http was wrong)* |
| r4.1+ | *(all complete)* |

### 5.12 Shared Machines
- r1, r2, r3, r4: 100 shared machines
- r6: 400 shared machines
- Shared machines are computer-lab style — multiple users log in throughout the day
- Each user also has one assigned/dedicated PC

### 5.13 Timestamp Range
All versions span roughly **January 2010 through May 2011** (~17 months).
- Some have events starting December 2009
- Terminated employees stop generating events on their termination day

---

## 6. Canonical Schema for the Universal Pipeline

The adapter layer should normalize all versions to these canonical column sets:

**logon:** `id, date, user, pc, activity`
**device:** `id, date, user, pc, activity` (normalized to Connect/Disconnect; file_tree dropped or kept as optional)
**email:** `id, date, user, pc, to, cc, bcc, from, size, attachments, content` (user inferred from `from` in r2/r3; missing cols filled with None/0)
**http:** `id, date, user, pc, url, content` (activity dropped or kept as optional; content None for r1)
**file:** `id, date, user, pc, filename, content` (only removable-media copies for r3-r4; filter on to_removable_media=True for r5+)
**LDAP:** `employee_name, user_id, email, role, business_unit, functional_unit, department, team, supervisor` (missing org fields filled with None for r1/r2)
**psychometric:** `employee_name, user_id, O, C, E, A, N` (absent in r1)
**answers (insiders.csv):** `dataset, scenario, details, user, start, end`

---

## 7. Adapter Fingerprints

Each version can be identified by checking the header of its CSV files:

| Check | Result | Version family |
|-------|--------|---------------|
| No http.csv header row | → r1 |  |
| logon.csv user contains `DTAA/` | → r1 |  |
| email.csv header = `id,date,to,from` (4 cols) | → r2 |  |
| device.csv has `Insert`/`Remove` | → r2 |  |
| email.csv header has `size` but no `user` | → r3.x |  |
| email.csv header has `user` but no `activity` | → r4.x |  |
| email.csv header has `activity` + file.csv has `to_removable_media` | → r5.x or r6.x |  |
| http.csv header has `activity` column | → r6.x |  |
| LDAP has `projects` column | → r5.x or r6.x |  |
| LDAP has `Domain` (capitalized) | → r1 |  |
| decoy_file.csv exists | → r5.x or r6.x |  |

---

## 8. Key Numbers for Feature Engineering

**Business hours:** 07:00 - 19:00 (from r4.2 readme: "After-hours logins and after-hours thumb drive usage are intended to be significant")

**Internal domain:** `dtaa.com`

**Job search keywords (from r4.2 URLs):** monster.com, careerbuilder.com, craigslist.org (jobs section), jobhuntersbible.com, aol.com/jobs

**Cloud/leak keywords:** wikileaks.org, dropbox (r6 scenario 5)

**Keylogger keywords (scenario 3):** refog.com, softactivity.com, "keylogger", "stealth", "surveillance"

**File extension → type mapping:**
- `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`, `.txt`, `.csv` → documents
- `.pdf` → PDF
- `.exe`, `.dll`, `.bat`, `.cmd`, `.sh` → executables
- `.zip`, `.tar`, `.gz`, `.rar`, `.7z` → archives
- `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp` → images

**Email size:** In bytes (integer). Refers to message body only, not including attachments.

**Attachment field evolution:**
- r2: absent
- r3-r4: integer count
- r5+: semicolon-separated filenames (or empty)

---

## 9. What the Readme Files Say About Signal

From the official CERT r4.2 readme, these behavioral dimensions are "fair game for anomaly detection":

1. Radical changes in behavior (for that specific user)
2. Unusual logon times (for that user)
3. Logins to another user's dedicated machine (for users that don't normally do this)
4. Device usage for non-device-users, or increased device usage
5. Employee termination (as an indicator)
6. Number of emails sent per day
7. Change in web browsing habits (unusual websites)
8. Radical change in social graph behavior (unexpected email recipients)
9. Topics of websites visited, emails, and files copied

The per-user baseline (EMA + z-scored deviations) is designed to capture exactly these signals.
