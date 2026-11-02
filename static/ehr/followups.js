// ReadmitAI EHR - follow-up to-do list for care coordinators

const { api, esc, tierBadge, riskPill, formatDate, toast, kindIcon, dueLabel, attemptClass, loadNavBadge } = EHR;

const GROUPS = [
  { key: "overdue", title: "Overdue" },
  { key: "today", title: "Due today" },
  { key: "upcoming", title: "Upcoming" },
];
const CHECK_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
const PHONE_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1.9.4 1.8.7 2.7a2 2 0 0 1-.5 2.1L8 9.8a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.7.7a2 2 0 0 1 1.7 2z"/></svg>';

let view = "open";
let outcomes = [];

// ---------- Summary & hero ----------

async function loadSummary() {
  const s = await api("/ehr/followups/summary");
  outcomes = s.call_outcomes;
  document.getElementById("cOverdue").textContent = s.overdue;
  document.getElementById("cToday").textContent = s.due_today;
  document.getElementById("cUpcoming").textContent = s.upcoming;
  document.getElementById("cDone").textContent = s.done;

  const today = new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
  document.getElementById("heroTitle").textContent = `Today's follow-ups · ${today}`;
  const urgent = s.overdue + s.due_today;
  document.getElementById("heroSub").textContent = urgent
    ? `${urgent} task${urgent === 1 ? "" : "s"} need${urgent === 1 ? "s" : ""} attention now (${s.overdue} overdue), ${s.upcoming} coming up.`
    : s.upcoming
      ? `Nothing due right now. ${s.upcoming} task${s.upcoming === 1 ? "" : "s"} coming up.`
      : "All caught up — no open follow-ups.";

  // Ring: share of today's workload (done today vs. done today + still due/overdue)
  const total = s.done_today + urgent;
  const share = total ? s.done_today / total : 1;
  document.getElementById("ringBar").style.strokeDashoffset = 213.6 * (1 - share);
  document.getElementById("ringLabel").textContent = s.done_today;
  loadNavBadge();
}

// ---------- Board ----------

function attemptsHtml(task) {
  if (!task.attempts.length) return "";
  const pills = task.attempts
    .map((a) => `<span class="attempt-pill ${attemptClass(a.outcome)}" title="${esc(a.note || "")}">${esc(a.outcome)} · ${esc(new Date(a.at).toLocaleDateString(undefined, { month: "short", day: "numeric" }))}</span>`)
    .join("");
  return `<div class="attempts">${task.attempts.length} call attempt${task.attempts.length === 1 ? "" : "s"}: ${pills}</div>`;
}

function taskHtml(task) {
  const p = task.patient;
  const e = task.encounter;
  const done = task.status === "done";
  const chartUrl = `patient.html?id=${encodeURIComponent(p.id)}&encounter=${encodeURIComponent(e.id)}`;
  const isCall = task.kind === "call";

  let actions = "";
  if (!done) {
    if (isCall) {
      actions += p.phone
        ? `<a class="btn btn-call btn-sm" href="tel:${esc(p.phone.replace(/[^0-9+]/g, ""))}">${PHONE_SVG} ${esc(p.phone)}</a>`
        : `<div class="phone-missing">No phone on file</div>`;
      actions += `<button class="btn btn-ghost btn-sm" data-action="log">Log call</button>`;
    } else {
      actions += `<button class="btn btn-ghost btn-sm" data-action="complete">Mark done</button>`;
    }
  } else {
    actions += `<button class="btn btn-ghost btn-sm" data-action="reopen">Reopen</button>`;
  }
  actions += `<a class="btn btn-ghost btn-sm" href="${chartUrl}">Open chart</a>`;

  return `
    <article class="task ${task.bucket}" data-id="${esc(task.id)}">
      <button class="check-btn ${done ? "checked" : ""}" data-action="${done ? "reopen" : isCall ? "log" : "complete"}"
        title="${done ? "Reopen" : isCall ? "Log the call" : "Mark done"}" aria-label="${done ? "Reopen task" : "Complete task"}">${CHECK_SVG}</button>
      <div>
        <div class="task-top">
          ${kindIcon(task.kind)}
          <span class="task-title">${esc(task.title)}</span>
          <span class="due ${task.bucket}">${esc(dueLabel(task))}</span>
        </div>
        <div class="task-patient">
          <a href="${chartUrl}">${esc(p.name)}</a>
          <span class="mono muted">${esc(p.mrn)}</span>
          ${riskPill(e.readmission_probability, e.risk_level)}
          ${tierBadge(e.risk_level)}
        </div>
        <div class="task-why">Discharged ${esc(formatDate(e.discharge_date))} · ${esc(e.discharge_disposition)} · ${esc(e.primary_diagnosis)}</div>
        <div class="task-detail">${esc(done && task.note ? `Note: ${task.note}` : task.detail)}</div>
        ${attemptsHtml(task)}
      </div>
      <div class="task-actions">${actions}</div>
    </article>`;
}

async function loadBoard() {
  const q = document.getElementById("search").value.trim();
  const params = new URLSearchParams({ view });
  if (q) params.set("q", q);
  const tasks = await api(`/ehr/followups?${params}`);
  const board = document.getElementById("board");

  if (!tasks.length) {
    board.innerHTML = view === "open" && !q
      ? `<div class="card all-done"><div class="big">✓</div><h2>All caught up</h2><p class="muted">No open follow-ups. New ones appear when a flagged patient is discharged.</p></div>`
      : `<div class="card empty">${q ? "No tasks match this search." : "No completed tasks yet."}</div>`;
    return;
  }

  if (view === "done") {
    board.innerHTML = `<section class="fu-group"><div class="fu-group-head">Completed <span class="count">${tasks.length}</span></div>${tasks.map(taskHtml).join("")}</section>`;
    return;
  }
  board.innerHTML = GROUPS.map((g) => {
    const items = tasks.filter((t) => t.bucket === g.key);
    if (!items.length) return "";
    return `<section class="fu-group" id="group-${g.key}">
      <div class="fu-group-head ${g.key}">${g.title} <span class="count">${items.length}</span></div>
      ${items.map(taskHtml).join("")}
    </section>`;
  }).join("");
}

async function refresh() {
  await Promise.all([loadSummary(), loadBoard()]);
}

// ---------- Actions ----------

function openCallPanel(card) {
  document.querySelectorAll(".call-panel").forEach((p) => p.remove());
  const panel = document.createElement("div");
  panel.className = "call-panel";
  panel.innerHTML = `
    <div class="small muted" style="margin-bottom:8px;font-weight:600">How did the call go?</div>
    <div class="outcomes">${outcomes.map((o) => `<button class="outcome" data-outcome="${esc(o)}">${esc(o)}</button>`).join("")}</div>
    <div class="call-panel-row">
      <input class="call-note" maxlength="500" placeholder="Note (optional), e.g. feeling better, appointment Friday">
      <button class="btn btn-primary btn-sm" data-action="save-call" disabled>Save</button>
      <button class="btn btn-ghost btn-sm" data-action="cancel-call">Cancel</button>
    </div>`;
  card.appendChild(panel);
  panel.querySelector(".call-note").focus();
}

async function saveCall(card) {
  const panel = card.querySelector(".call-panel");
  const selected = panel.querySelector(".outcome.selected");
  if (!selected) return;
  const outcome = selected.dataset.outcome;
  const note = panel.querySelector(".call-note").value.trim() || null;
  const task = await api(`/ehr/followups/${encodeURIComponent(card.dataset.id)}/attempts`, {
    method: "POST",
    body: JSON.stringify({ outcome, note }),
  });
  const messages = {
    Reached: `${task.patient.name} reached — task done`,
    "No answer": "No answer — call moved to tomorrow",
    "Left voicemail": "Voicemail left — call moved to tomorrow",
    "Wrong number": "Logged wrong number — check the phone on the chart",
  };
  toast(messages[outcome] || "Call logged");
  refresh();
}

async function setDone(card, done) {
  await api(`/ehr/followups/${encodeURIComponent(card.dataset.id)}`, {
    method: "PATCH",
    body: JSON.stringify({ done }),
  });
  toast(done ? "Task completed" : "Task reopened");
  refresh();
}

function setView(next) {
  view = next;
  document.querySelectorAll(".segmented button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  return loadBoard();
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("board").addEventListener("click", (e) => {
    const card = e.target.closest(".task");
    if (!card) return;
    const outcomeBtn = e.target.closest(".outcome");
    if (outcomeBtn) {
      card.querySelectorAll(".outcome").forEach((b) => b.classList.toggle("selected", b === outcomeBtn));
      card.querySelector('[data-action="save-call"]').disabled = false;
      return;
    }
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (action === "log") openCallPanel(card);
    else if (action === "complete") setDone(card, true);
    else if (action === "reopen") setDone(card, false);
    else if (action === "save-call") saveCall(card);
    else if (action === "cancel-call") card.querySelector(".call-panel")?.remove();
  });
  document.getElementById("board").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.classList.contains("call-note")) saveCall(e.target.closest(".task"));
  });

  document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  document.querySelectorAll(".fu-chip").forEach((chip) =>
    chip.addEventListener("click", async () => {
      if (chip.dataset.view) return setView(chip.dataset.view);
      if (view !== "open") await setView("open");
      document.getElementById(`group-${chip.dataset.jump}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    })
  );
  let timer;
  document.getElementById("search").addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(loadBoard, 250);
  });

  refresh();
});
