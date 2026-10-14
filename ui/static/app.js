/* The page's own logic: start a task, follow its lines, render its result.
   No numbers are computed here. Everything displayed arrives from the server.

   On layout: every top-level section is a `block`, separated by a hard rule,
   and every repeated thing inside one — a model's scenarios, one stored run —
   is a `card`. That is not decoration. Two tables of similar numbers running
   together is what made a reader ask whether the page was showing the same
   thing twice, so the structure has to make "these are different things"
   visible before anyone reads a figure. */

"use strict";

const $ = (id) => document.getElementById(id);
const log = $("log");

let stream = null;

/* ---------------------------------------------------------------- tabs */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
    tab.classList.add("active");
    $(tab.dataset.panel).classList.remove("hidden");
  });
});

/* ----------------------------------------------------------- detection */

async function identify(selectId, targetId) {
  const target = $(targetId);
  target.textContent = "reading five header lines…";
  target.className = "detection";
  const response = await fetch("/api/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ release: $(selectId).value }),
  });
  const detection = await response.json();
  target.textContent = detection.report;
  target.classList.add(detection.allowed ? "ok" : "bad");
}

$("run-detect").addEventListener("click", () => identify("run-folder", "run-detection"));
$("train-detect").addEventListener("click", () => identify("train-folder", "train-detection"));

/* The chosen entry's path, shown under the menu so the folder in use is never
   a guess. The menu is a shortcut to a location, not a claim about contents —
   the header read is what decides that, and it runs either way. */
function showPath(selectId, targetId) {
  const option = $(selectId).selectedOptions[0];
  const path = option ? option.dataset.path : "";
  $(targetId).textContent = path
    ? `→ ${path}`
    : "→ not found on this machine. Plug the drive in, or pick the other one.";
}

["run", "train"].forEach((panel) => {
  const select = $(`${panel}-folder`);
  select.addEventListener("change", () => showPath(`${panel}-folder`, `${panel}-path`));
  showPath(`${panel}-folder`, `${panel}-path`);
});

/* ------------------------------------------ the trained-models view */

/* A record of the training sessions started from this page, and nothing else.

   Every number is read out of a run's own manifest.json, written by the run
   that produced it. Nothing here runs, loads a checkpoint or opens
   final_results.json: the command line's results are not this tab's subject,
   so on a machine where nothing has been trained here the tab is empty and
   says so.

   One <section class="session"> per run is what makes that readable. The
   heaviest rule on the page now separates training sessions from each other,
   where before it separated metric types and left the runs as small cards
   inside them — which is why it was impossible to see what belonged to what. */

async function loadModels() {
  const body = $("models-body");
  body.textContent = "reading the stored record…";
  const runs = await (await fetch("/api/models")).json();
  body.innerHTML = "";

  if (!runs.length) {
    body.appendChild(emptyState());
    return;
  }
  body.appendChild(sessionIndex(runs));
  runs.forEach((run, i) => body.appendChild(sessionBlock(run, i + 1, runs.length)));
}

function emptyState() {
  const node = document.createElement("div");
  node.className = "empty";
  const head = document.createElement("h3");
  head.textContent = "Nothing trained here yet";
  node.appendChild(head);
  node.appendChild(paragraph(
    "Train a model on the \u201cTrain a new model on a folder\u201d tab. When it " +
    "finishes, that session and everything it measured appear here."
  ));
  return node;
}

/* The index exists so five sessions can be told apart before any of them is
   read. Its rows link down to the blocks below. */
function sessionIndex(runs) {
  const section = block(
    `${runs.length} training session${runs.length === 1 ? "" : "s"}`,
    "newest first"
  );
  section.appendChild(note(
    "Average precision on the validation split — the only split a model " +
    "trained on this page ever sees."
  ));
  section.appendChild(table(
    ["#", "training session", "when", "dataset", "seeds",
     "model_a — 1 day", "model_b — 28 days"],
    runs.map((run, i) => {
      const link = document.createElement("a");
      link.href = `#run-${run.run_id}`;
      link.textContent = run.run_id;
      return [
        String(i + 1), link, when(run.created), `r${run.release}`,
        run.seeds.join(", "), headlineScore(run, "model_a"),
        headlineScore(run, "model_b"),
      ];
    })
  ));
  return section;
}

function headlineScore(run, name) {
  const model = run.models.find((m) => m.model === name);
  return model ? fixed(model.auprc.mean, 4) : "n/a";
}

function sessionBlock(run, index, total) {
  const section = document.createElement("section");
  section.className = "session";
  section.id = `run-${run.run_id}`;

  const bar = document.createElement("div");
  bar.className = "session-bar";
  const eyebrow = document.createElement("div");
  eyebrow.className = "session-n";
  eyebrow.textContent = `Training session ${index} of ${total}` +
    (index === 1 ? " · most recent" : "");
  const title = document.createElement("h3");
  title.textContent = run.run_id;
  const sub = document.createElement("p");
  sub.className = "session-sub";
  sub.textContent = `${when(run.created)} · ${run.release_label} · ` +
    `seeds ${run.seeds.join(", ")}`;
  bar.append(eyebrow, title, sub);
  section.appendChild(bar);

  if (run.stale) section.appendChild(warn(run.stale));
  (run.warnings || []).forEach((w) => section.appendChild(warn(w)));

  section.appendChild(subhead("What was trained"));
  section.appendChild(kvTable([
    ["dataset", run.release_label],
    ["folder read", run.source_folder],
    ["seeds", run.seeds.join(", ")],
    ["device", run.device +
      (run.device_asked_for && run.device_asked_for !== run.device
        ? ` (asked for ${run.device_asked_for})` : "")],
    ["finished", when(run.created)],
    ["total training time", duration(run.seconds_total)],
    ["history window", run.window_days ? `${run.window_days} days` : "n/a"],
    ["code commit", (run.git_commit || "not a git checkout").slice(0, 7)],
  ]));

  /* A run trained before the manifest kept its measurements says so and names
     the command that repairs it. Showing zeros would be a lie about a number
     nobody wrote. */
  if (!run.measured) {
    section.appendChild(warn(
      "This session recorded no measurements. Rebuild them from its sandbox " +
      `with: python -m pipeline.store --backfill ${run.run_id}`
    ));
    return section;
  }

  section.appendChild(subhead("How well it scored — averaged over the seeds"));
  section.appendChild(table(
    ["model", "context", "parameters", "average precision", "AUROC",
     "P@10", "P@50", "random", "vs random"],
    run.models.map((m) => [
      m.model,
      m.context_days === 1 ? "1 day" : `${m.context_days} days`,
      count(m.n_parameters),
      `${fixed(m.auprc.mean, 4)} ± ${fixed(m.auprc.std, 4)}`,
      fixed(m.auroc.mean, 4),
      fixed(m.precision_at_10, 2),
      fixed(m.precision_at_50, 2),
      fixed(m.random_baseline, 5),
      m.lift_over_random == null ? "n/a" : `${fixed(m.lift_over_random, 0)}×`,
    ])
  ));

  section.appendChild(subhead("Each seed on its own"));
  const seedRows = [];
  run.models.forEach((m) => m.per_seed.forEach((s) => seedRows.push([
    m.model, s.seed, fixed(s.auprc, 4), fixed(s.auroc, 4),
    fixed(s.threshold, 4), fixed(s.precision, 3), fixed(s.recall, 3),
    count(s.days_flagged), s.best_epoch, s.epochs_run,
    duration(s.seconds_total),
  ])));
  section.appendChild(table(
    ["model", "seed", "average precision", "AUROC", "threshold", "precision",
     "recall", "days flagged", "best epoch", "epochs run", "time"],
    seedRows
  ));

  if (run.scenarios.length) {
    section.appendChild(subhead("Which attacks it caught — all seeds added together"));
    const names = run.models.map((m) => m.model);
    section.appendChild(table(
      ["attack type", ...names, "attack days in total"],
      run.scenarios.map((s) => [
        s.label, ...names.map((n) => count(s.hits[n])), count(s.attack_days),
      ])
    ));
  }
  return section;
}

function subhead(text) {
  const node = document.createElement("h4");
  node.className = "session-h";
  node.textContent = text;
  return node;
}

/* Label on the left, value on the right, both left-aligned: the facts about a
   session are not quantities to compare down a column. */
function kvTable(pairs) {
  const wrap = document.createElement("div");
  wrap.className = "scroll";
  const node = document.createElement("table");
  node.className = "kv";
  pairs.forEach(([label, value]) => {
    const line = document.createElement("tr");
    const head = document.createElement("th");
    head.scope = "row";
    head.textContent = label;
    const cell = document.createElement("td");
    cell.textContent = value === null || value === undefined ? "n/a" : value;
    line.append(head, cell);
    node.appendChild(line);
  });
  wrap.appendChild(node);
  return wrap;
}

function when(iso) {
  if (!iso) return "n/a";
  const at = new Date(iso);
  return Number.isNaN(at.getTime()) ? iso : at.toLocaleString(undefined, {
    day: "numeric", month: "long", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function duration(seconds) {
  if (seconds === null || seconds === undefined) return "n/a";
  const whole = Math.round(seconds);
  return whole < 60 ? `${whole}s`
    : `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, "0")}s`;
}

function count(value) {
  return value === null || value === undefined
    ? "n/a" : Number(value).toLocaleString();
}

$("models-refresh").addEventListener("click", loadModels);
document.querySelector('[data-panel="models"]').addEventListener("click", () => {
  if (!$("models-body").hasChildNodes()) loadModels();
});

/* ---------------------------------------------------------- stored models */

async function loadStoredModels() {
  const response = await fetch("/api/runs");
  const runs = await response.json();
  const select = $("run-stored");
  runs.forEach((manifest) => {
    const option = document.createElement("option");
    option.value = manifest.run_id;
    option.textContent = `${manifest.run_id} — r${manifest.release}, seeds ${manifest.seeds.join(", ")}`;
    select.appendChild(option);
  });
}
loadStoredModels();

/* ------------------------------------------------------------- running */

function begin(label) {
  $("status").classList.remove("hidden");
  $("results").classList.add("hidden");
  $("results").innerHTML = "";
  log.textContent = "";
  setState("running", label);
  if (stream) { stream.close(); stream = null; }
}

function setState(state, text) {
  const badge = $("state-badge");
  badge.className = `badge ${state}`;
  badge.textContent = state;
  $("state-text").textContent = text || "";
}

function follow(taskId) {
  stream = new EventSource(`/api/stream/${taskId}`);
  stream.addEventListener("line", (event) => {
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.textContent += event.data + "\n";
    if (atBottom) log.scrollTop = log.scrollHeight;
  });
  stream.addEventListener("done", async () => {
    stream.close();
    stream = null;
    const response = await fetch(`/api/task/${taskId}`);
    render(await response.json());
  });
}

async function start(url, payload, label) {
  begin(label);
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const { task } = await response.json();
  follow(task);
}

$("run-go").addEventListener("click", () =>
  start("/api/infer", {
    release: $("run-folder").value,
    device: $("run-device").value,
    sample: $("run-sample").checked,
    stored_run: $("run-stored").value,
    allow_transfer: $("run-transfer").checked,
  }, "scoring")
);

$("train-go").addEventListener("click", () =>
  start("/api/train", {
    release: $("train-folder").value,
    device: $("train-device").value,
    seeds: $("train-seeds").value.split(/[,\s]+/).filter(Boolean).map(Number),
    max_epochs: Number($("train-epochs").value) || null,
  }, "training")
);

/* ------------------------------------------------------------ rendering */

function render(task) {
  setState(task.state, task.exit_meaning || task.error || "");
  const result = task.result;
  if (!result) return;

  const target = $("results");
  target.classList.remove("hidden");
  target.innerHTML = "";

  if (result.run_id) { renderTraining(target, result); return; }
  renderInference(target, result);
}

/* A finished training run is rendered by the Results tab, not here.

   This used to build the run's own block from `result.manifest` with the same
   function the old models tab used. That meant two renderers for one thing,
   and the manifest is the raw record rather than the shape the tab draws, so
   the two could not stay the same. Now the run is confirmed in one line and
   the Results tab is reloaded, which reads the manifest back through the
   server and so shows exactly what it will show tomorrow. */
function renderTraining(target, result) {
  const done = card(`Trained — ${result.run_id}`);
  done.appendChild(paragraph(
    "Stored under models/" + result.run_id + "/. It is now on the Results " +
    "tab, and in the stored-model menu on the Run tab."
  ));
  const go = document.createElement("button");
  go.className = "secondary";
  go.textContent = "Open the Results tab";
  go.addEventListener("click", () => {
    document.querySelector('[data-panel="models"]').click();
  });
  done.appendChild(go);
  target.appendChild(done);
  loadModels();
  loadStoredModels();
}

function renderInference(target, result) {
  if (result.meaningless) {
    target.appendChild(warn(
      "Sample run — 30 users at roughly a 30% insider rate. Every number " +
      "below proves the code runs end to end and nothing more. None of it is " +
      "a result and none of it may be quoted."
    ));
  }

  /* The same mechanism, a second reason to use it. Note that not one word of
     this banner is written here: the lines come from the run itself, so the
     page and the streamed log say the same sentences about the same run
     rather than two hand-kept copies that can drift apart. The middle lines
     are the degenerate-feature caveat — three of the eighteen features are
     dead on r4.2 and alive on r5.2 — which is the part a green guard block
     does not tell you. */
  if (result.transfer) {
    result.transfer.banner.forEach((line) => target.appendChild(warn(line)));
  }

  const perSeed = block("Per model, per seed", `${result.split} split`);
  perSeed.appendChild(table(
    ["model", "seed", "AUPRC", "AUROC", "P@10", "P@50", "random", "x random", "rows flagged"],
    result.per_seed.map((r) => [
      r.model, r.seed, fixed(r.auprc, 4), fixed(r.auroc, 4),
      fixed(r.precision_at_k["10"], 3), fixed(r.precision_at_k["50"], 3),
      fixed(r.random_baseline, 5), fixed(r.lift_over_random, 1),
      (r.n_flagged || 0).toLocaleString(),
    ])
  ));
  perSeed.appendChild(note(
    `${result.per_seed[0].n_rows.toLocaleString()} test rows on the first seed, ` +
    `of which ${result.per_seed[0].n_positives.toLocaleString()} are attack days. ` +
    `Device ${result.device}. Every threshold above was read from its ` +
    `checkpoint, not chosen here.`
  ));
  target.appendChild(perSeed);

  const across = block("Across seeds", "AUPRC aggregated; recall deliberately not");
  result.models.forEach((model) => {
    const one = card(model.model);
    one.appendChild(paragraph(
      `AUPRC ${fixed(model.auprc.mean, 4)} ± ${fixed(model.auprc.std, 4)} ` +
      `over ${model.seeds.length} seeds, against a random baseline of ` +
      `${fixed(model.random_baseline, 5)} — ${fixed(model.lift_mean, 0)}× random. ` +
      `${model.context_days} day(s) of context, ${model.n_parameters.toLocaleString()} parameters.`
    ));
    one.appendChild(paragraph(
      `Rows flagged per seed: ${model.rows_flagged_per_seed.join(", ")}. Not averaged — ` +
      `the spread is the finding.`
    ));
    one.appendChild(paragraph(latencySentence(model)));
    one.appendChild(table(
      ["scenario", "flagged", "of attack days", "in words"],
      model.per_scenario_pooled.map((s) => [
        `scenario ${s.scenario}`, s.hits, s.attack_days,
        s.attack_days ? `${s.hits} of ${s.attack_days} attack days` : "no positive rows",
      ])
    ));
    across.appendChild(one);
  });
  across.appendChild(note(
    "Per-scenario results are pooled counts across seeds, never averaged " +
    "recalls: scenario 3 has four positive rows per split, and a percentage " +
    "over four rows hides its own denominator."
  ));
  target.appendChild(across);

  const guards = block("Guards", `${result.guards.length} checks`);
  result.guards.forEach((g) => guards.appendChild(guardLine(g)));
  target.appendChild(guards);
}

function latencySentence(model) {
  if (model.median_first_detection_day === null) {
    return `${model.model}: none of ${model.insiders_total} insiders flagged at all, ` +
           `so there is no first-detection day to quote.`;
  }
  return `${model.model}: ${model.insiders_detected} of ${model.insiders_total} ` +
         `insiders flagged, median first detection on day ` +
         `${fixed(model.median_first_detection_day, 0)} of the attack. Recall ` +
         `without latency overstates what a persistence detector does, so the ` +
         `two are quoted together.`;
}

/* -------------------------------------------------------------- helpers */

/* A top-level section: hard rule above it, a title, an optional subtitle. */
function block(title, subtitle) {
  const section = document.createElement("section");
  section.className = "block";
  const head = document.createElement("h3");
  head.textContent = title;
  if (subtitle) {
    const tag = document.createElement("span");
    tag.className = "eyebrow";
    tag.textContent = subtitle;
    head.appendChild(tag);
  }
  section.appendChild(head);
  return section;
}

/* One repeated thing inside a block: a model, a stored run, a guard list. */
function card(title) {
  const node = document.createElement("div");
  node.className = "card";
  if (title) {
    const head = document.createElement("h4");
    head.textContent = title;
    node.appendChild(head);
  }
  return node;
}

function guardLine(guard) {
  const line = document.createElement("div");
  line.className = `guard ${guard.passed ? "pass" : "fail"}`;
  line.innerHTML =
    `<span class="verdict">${guard.passed ? "ok" : "FAIL"}</span>` +
    `<span>${escape(guard.guard)}</span> — <span class="detail">${escape(guard.detail)}</span>`;
  return line;
}

function fixed(value, places) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "n/a" : Number(value).toFixed(places);
}

function paragraph(text) {
  const node = document.createElement("p");
  node.textContent = text;
  return node;
}

function note(text) {
  const node = document.createElement("p");
  node.className = "note";
  node.textContent = text;
  return node;
}

function warn(text) {
  const node = document.createElement("div");
  node.className = "warn";
  node.textContent = text;
  return node;
}

/* Tables scroll inside their own box rather than widening the page: the
   per-seed table has eleven columns and a laptop does not. */
function table(headers, rows) {
  const wrap = document.createElement("div");
  wrap.className = "scroll";
  const node = document.createElement("table");
  const head = document.createElement("tr");
  headers.forEach((h) => {
    const cell = document.createElement("th");
    cell.textContent = h;
    head.appendChild(cell);
  });
  node.appendChild(head);
  rows.forEach((row) => {
    const line = document.createElement("tr");
    row.forEach((value, index) => {
      const cell = document.createElement("td");
      if (index > 0) cell.className = "num";
      if (value instanceof Node) cell.appendChild(value);
      else cell.textContent = value;
      line.appendChild(cell);
    });
    node.appendChild(line);
  });
  wrap.appendChild(node);
  return wrap;
}

function escape(text) {
  const node = document.createElement("div");
  node.textContent = text;
  return node.innerHTML;
}
