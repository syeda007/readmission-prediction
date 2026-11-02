// ReadmitAI EHR - patient chart: admission history, risk detail, new admission form

const { api, esc, tierClass, tierBadge, statusBadge, riskPill, formatDate, formatAge, yearsSince, showErrors, toast, options, kindIcon, dueLabel, loadNavBadge } = EHR;

const params = new URLSearchParams(location.search);
const patientId = params.get("id");

let formOptions = null;
let patient = null;
let selectedId = params.get("encounter");

const OTHER_MEDS = [
  "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "glipizide", "glyburide",
  "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
  "tolazamide", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
  "metformin-rosiglitazone",
];

const NUMERIC_FIELDS = [
  "number_inpatient", "number_emergency", "number_outpatient", "num_lab_procedures",
  "num_procedures", "num_medications", "number_diagnoses",
];

const ENCOUNTER_FIELD_LABELS = {
  admission_date: "Admission date",
  discharge_date: "Discharge date",
  admission_type_id: "Admission type",
  admission_source_id: "Admitted from",
  discharge_disposition_id: "Discharge destination",
  medical_specialty: "Specialty",
  diag_1: "Primary diagnosis",
  diag_2: "Secondary diagnosis",
  diag_3: "Additional diagnosis",
  number_diagnoses: "Total diagnoses",
  number_inpatient: "Inpatient admissions",
  number_emergency: "Emergency visits",
  number_outpatient: "Outpatient visits",
  num_lab_procedures: "Lab tests",
  num_procedures: "Other procedures",
  num_medications: "Medications given",
  diabetesMed: "Diabetes medication prescribed",
  change: "Changed during stay",
};

const labelOf = (list, value) => (list.find((o) => String(o.value) === String(value)) || {}).label || `code ${value}`;
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// ---------- Banner ----------

function renderBanner() {
  const initials = `${patient.first_name[0]}${patient.last_name[0]}`.toUpperCase();
  document.title = `ReadmitAI EHR — ${patient.last_name}, ${patient.first_name}`;
  document.getElementById("banner").innerHTML = `
    <div class="avatar">${esc(initials)}</div>
    <div>
      <h1>${esc(patient.last_name)}, ${esc(patient.first_name)}</h1>
      <div class="banner-facts">
        <div><div class="fact-label">MRN</div><div class="fact-value mono">${esc(patient.mrn)}</div></div>
        <div><div class="fact-label">Date of birth</div><div class="fact-value">${esc(formatDate(patient.date_of_birth))} (${yearsSince(patient.date_of_birth)} y)</div></div>
        <div><div class="fact-label">Gender</div><div class="fact-value">${esc(patient.gender)}</div></div>
        <div><div class="fact-label">Race</div><div class="fact-value">${esc(patient.labels.race)}</div></div>
        <div><div class="fact-label">Insurance</div><div class="fact-value">${esc(patient.labels.payer)}</div></div>
        <div><div class="fact-label">Phone</div><div class="fact-value">${
          patient.phone ? `<a href="tel:${esc(patient.phone.replace(/[^0-9+]/g, ""))}">${esc(patient.phone)}</a>` : "Not recorded"
        }</div></div>
      </div>
    </div>
    <div class="banner-actions">
      <button class="btn btn-danger" id="btnDeletePatient">Delete patient</button>
    </div>`;
  document.getElementById("btnDeletePatient").addEventListener("click", deletePatient);
}

// ---------- Admission list ----------

function renderAdmissionList() {
  const list = document.getElementById("admList");
  document.getElementById("admEmpty").classList.toggle("hidden", patient.encounters.length > 0);
  list.innerHTML = patient.encounters
    .map(
      (e) => `
      <li class="adm-item ${e.id === selectedId ? "selected" : ""}" data-id="${esc(e.id)}">
        <div>
          <div class="adm-dates">${esc(formatDate(e.discharge_date))}</div>
          <div class="cell-sub">${e.time_in_hospital} day stay · ${esc(e.labels.diagnoses[0].description)}</div>
          <div style="margin-top:4px">${statusBadge(e.review.status)}</div>
        </div>
        <div style="text-align:right">
          ${riskPill(e.prediction.readmission_probability, e.prediction.risk_level)}
          <div style="margin-top:4px">${tierBadge(e.prediction.risk_level)}</div>
        </div>
      </li>`
    )
    .join("");
}

// ---------- Encounter detail ----------

function factorRows(factors, direction, maxAbs) {
  if (!factors.length) return `<p class="muted small">None of note.</p>`;
  return factors
    .map(
      (f) => `
      <div class="factor">
        <span>${esc(f.description)}</span>
        <span class="factor-bar ${direction}" style="width:${Math.max((Math.abs(f.shap_value) / maxAbs) * 100, 6)}%" title="${esc(f.shap_value.toFixed(3))}"></span>
      </div>`
    )
    .join("");
}

function renderDetail() {
  const e = patient.encounters.find((x) => x.id === selectedId);
  if (!e) return;
  const p = e.prediction;
  const cls = tierClass(p.risk_level);
  const threshold = p.decision_threshold;
  const scaleMax = 0.5; // the scale shows 0-50%; risks above that are rare

  const inc = p.explanation.risk_increasing_factors;
  const dec = p.explanation.risk_decreasing_factors;
  const maxAbs = Math.max(...inc.concat(dec).map((f) => Math.abs(f.shap_value)), 0.001);

  const meds = ["insulin", "metformin", ...OTHER_MEDS].filter((m) => e[m] && e[m] !== "No");

  document.getElementById("detailTitle").textContent =
    `Admission ${formatDate(e.admission_date)} → ${formatDate(e.discharge_date)}`;
  document.getElementById("detailMeta").textContent = `${e.id} · scored ${new Date(p.scored_at).toLocaleString()}`;

  document.getElementById("detailBody").innerHTML = `
    <div class="risk-hero">
      <div>
        <div class="fact-label">30-day readmission risk</div>
        <div class="risk-big">${esc(p.readmission_risk_percentage)}</div>
      </div>
      <div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          ${tierBadge(p.risk_level)}
          ${p.requires_review ? '<span class="status pending">Flagged for review</span>' : '<span class="status">Below review threshold</span>'}
        </div>
        <div class="risk-scale">
          <div class="fill fill-${cls}" style="width:${Math.min(p.readmission_probability / scaleMax, 1) * 100}%"></div>
          <div class="mark" style="left:${(threshold / scaleMax) * 100}%" title="Review threshold"></div>
        </div>
        <div class="scale-labels"><span>0%</span><span>Review threshold ${(threshold * 100).toFixed(0)}%</span><span>50%+</span></div>
        <div class="muted small" style="margin-top:6px">Average patient in the training data: about 11.5%.</div>
      </div>
    </div>

    <div class="rec-box ${cls === "low" ? "" : cls}">
      <strong>Suggested follow-up:</strong> ${esc(p.clinical_recommendation)}
    </div>

    <div class="section-title">Follow-up plan</div>
    ${tasksHtml(e)}

    <div class="section-title">Care coordination</div>
    <div class="review-row">
      <div>
        <label for="reviewStatus">Review status</label>
        <select id="reviewStatus">${options(formOptions.review_statuses, e.review.status)}</select>
      </div>
      <div>
        <label for="reviewNote">Coordinator note</label>
        <input id="reviewNote" maxlength="1000" value="${esc(e.review.note || "")}" placeholder="e.g. Called patient, follow-up booked for Friday">
      </div>
      <button class="btn btn-primary" id="btnSaveReview">Save</button>
    </div>
    <div class="muted small" style="margin-top:6px">Last updated ${esc(new Date(e.review.updated_at).toLocaleString())}</div>

    <div class="section-title">Why this estimate</div>
    <div class="factor-cols">
      <div><h4 style="color:var(--high)">▲ Raising risk</h4>${factorRows(inc, "up", maxAbs)}</div>
      <div><h4 style="color:var(--low)">▼ Lowering risk</h4>${factorRows(dec, "down", maxAbs)}</div>
    </div>
    <p class="disclaimer">${esc(p.explanation.clinical_disclaimer)}</p>

    <div class="section-title">Clinical summary</div>
    <div class="kv">
      <div><div class="k">Age at admission</div>${esc(formatAge(e.age))}</div>
      <div><div class="k">Length of stay</div>${e.time_in_hospital} days</div>
      <div><div class="k">Admission type</div>${esc(e.labels.admission_type)}</div>
      <div><div class="k">Admitted from</div>${esc(e.labels.admission_source)}</div>
      <div><div class="k">Discharged to</div>${esc(e.labels.discharge_disposition)}</div>
      <div><div class="k">Specialty</div>${esc(e.labels.medical_specialty)}</div>
      <div><div class="k">Past-year visits</div>${e.number_inpatient} inpatient · ${e.number_emergency} ER · ${e.number_outpatient} outpatient</div>
      <div><div class="k">During stay</div>${e.num_lab_procedures} labs · ${e.num_procedures} procedures · ${e.num_medications} meds</div>
      <div><div class="k">HbA1c / glucose</div>${esc(e.labels.a1c_result)} / ${esc(e.labels.glucose_result)}</div>
      <div><div class="k">Diabetes medication</div>${e.diabetesMed === "Yes" ? "Prescribed" : "None"}${e.change === "Ch" ? " · changed" : ""}</div>
      <div><div class="k">Active diabetes drugs</div>${
        meds.length ? esc(meds.map((m) => `${capitalize(m)} (${labelOf(formOptions.medication_statuses, e[m]).toLowerCase()})`).join(", ")) : "None"
      }</div>
    </div>

    <div class="section-title">Diagnoses</div>
    <ul class="dx-list">
      ${e.labels.diagnoses
        .map((d, i) => `<li><strong>${["Primary", "Secondary", "Additional"][i]}:</strong> ${esc(d.description)} <span class="muted mono">(ICD-9 ${esc(d.code)})</span></li>`)
        .join("")}
      <li class="muted small">${e.number_diagnoses} diagnoses recorded in total</li>
    </ul>

    ${e.notes ? `<div class="section-title">Notes</div><div class="notes-box">${esc(e.notes)}</div>` : ""}

    <p class="muted small" style="margin-top:18px">Model: ${esc(p.model_name)} v${esc(p.model_version)}</p>`;

  document.getElementById("btnSaveReview").addEventListener("click", () => saveReview(e.id));
  document.querySelectorAll(".mini-task [data-task]").forEach((btn) =>
    btn.addEventListener("click", () => toggleTask(btn.dataset.task, btn.dataset.done === "true"))
  );
}

function tasksHtml(e) {
  if (!e.tasks.length) {
    return `<p class="muted small">No follow-up tasks: this discharge is below the review threshold.</p>`;
  }
  const items = e.tasks
    .map((t) => {
      const done = t.status === "done";
      const last = t.attempts.length ? ` · last call: ${t.attempts[t.attempts.length - 1].outcome}` : "";
      return `<li class="mini-task ${done ? "done" : ""}">
        ${kindIcon(t.kind)}
        <div>
          <div class="mini-title"><strong>${esc(t.title)}</strong></div>
          <div class="cell-sub"><span class="due ${t.bucket}">${esc(dueLabel(t))}</span>${esc(last)}</div>
        </div>
        <button class="btn btn-ghost btn-sm" data-task="${esc(t.id)}" data-done="${!done}">${done ? "Reopen" : "Mark done"}</button>
      </li>`;
    })
    .join("");
  return `<ul class="mini-tasks">${items}</ul>
    <p class="small" style="margin-top:6px"><a href="followups.html">Open the follow-up list</a> to log calls.</p>`;
}

async function toggleTask(taskId, done) {
  await api(`/ehr/followups/${encodeURIComponent(taskId)}`, { method: "PATCH", body: JSON.stringify({ done }) });
  await loadPatient();
  renderDetail();
  loadNavBadge();
  toast(done ? "Task completed" : "Task reopened");
}

async function saveReview(encounterId) {
  const status = document.getElementById("reviewStatus").value;
  const note = document.getElementById("reviewNote").value.trim() || null;
  const updated = await api(`/ehr/encounters/${encodeURIComponent(encounterId)}/review`, {
    method: "PATCH",
    body: JSON.stringify({ status, note }),
  });
  const idx = patient.encounters.findIndex((x) => x.id === encounterId);
  patient.encounters[idx] = updated;
  renderAdmissionList();
  renderDetail();
  toast("Review status saved");
}

// ---------- View switching ----------

function selectEncounter(id) {
  selectedId = id;
  document.getElementById("formCard").classList.add("hidden");
  document.getElementById("detailCard").classList.remove("hidden");
  history.replaceState(null, "", `?id=${encodeURIComponent(patientId)}&encounter=${encodeURIComponent(id)}`);
  renderAdmissionList();
  renderDetail();
}

function showForm() {
  selectedId = null;
  renderAdmissionList();
  document.getElementById("detailCard").classList.add("hidden");
  document.getElementById("formCard").classList.remove("hidden");
  resetForm();
  document.getElementById("formCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function hideForm() {
  document.getElementById("formCard").classList.add("hidden");
  if (patient.encounters.length) selectEncounter(patient.encounters[0].id);
}

// ---------- New admission form ----------

function buildFormControls() {
  document.getElementById("admission_type_id").innerHTML =
    `<option value="">Select…</option>${options(formOptions.admission_types)}`;
  document.getElementById("admission_source_id").innerHTML =
    `<option value="">Select…</option>${options(formOptions.admission_sources)}`;
  document.getElementById("discharge_disposition_id").innerHTML = `
    <option value="">Select…</option>
    ${options(formOptions.discharge_dispositions)}
    <optgroup label="Death / hospice (not risk-scored)">${options(formOptions.ineligible_discharge_dispositions)}</optgroup>`;
  document.getElementById("medical_specialty").innerHTML =
    `<option value="">Not recorded</option>${options(formOptions.medical_specialties)}`;
  document.getElementById("A1Cresult").innerHTML = options(formOptions.a1c_results, "None");
  document.getElementById("max_glu_serum").innerHTML = options(formOptions.glucose_results, "None");

  document.getElementById("otherMeds").innerHTML = OTHER_MEDS.map(
    (m) => `<div><label for="${m}">${esc(capitalize(m))}</label><select id="${m}" name="${m}" class="med"></select></div>`
  ).join("");
  document.querySelectorAll("select.med").forEach((s) => (s.innerHTML = options(formOptions.medication_statuses, "No")));

  ["admission_date", "discharge_date"].forEach((id) => document.getElementById(id).addEventListener("change", updateLengthOfStay));
  ["diag_1", "diag_2", "diag_3"].forEach((id) => {
    const input = document.getElementById(id);
    input.addEventListener("input", debounce(() => showDiagnosisHint(input), 300));
  });
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// Shows what an entered ICD-9 code means, so staff can check it before saving
async function showDiagnosisHint(input) {
  const hint = document.getElementById(`${input.id}_hint`);
  const code = input.value.trim();
  hint.classList.remove("bad");
  if (!code) return (hint.textContent = "");
  try {
    const result = await api(`/ehr/icd9/${encodeURIComponent(code)}`);
    if (input.value.trim() === code) hint.textContent = result.description;
  } catch (err) {
    if (input.value.trim() === code) {
      hint.textContent = "Not a recognised ICD-9-CM code";
      hint.classList.add("bad");
    }
  }
}

function resetForm() {
  document.getElementById("encounterForm").reset();
  document.getElementById("encounterError").classList.add("hidden");
  ["diag_1", "diag_2", "diag_3"].forEach((id) => (document.getElementById(`${id}_hint`).textContent = ""));
  document.querySelectorAll("select.med").forEach((s) => (s.value = "No"));
  const today = new Date().toISOString().slice(0, 10);
  document.getElementById("discharge_date").value = today;
  document.getElementById("number_inpatient").value = patient.admissions_last_year;
  document.getElementById("inpatientHint").textContent =
    `${patient.admissions_last_year} recorded in this chart in the past 12 months. Adjust if there were admissions elsewhere.`;
  updateLengthOfStay();
}

function updateLengthOfStay() {
  const a = document.getElementById("admission_date").value;
  const d = document.getElementById("discharge_date").value;
  const out = document.getElementById("losDisplay");
  if (!a || !d) return (out.value = "—");
  const days = Math.round((new Date(d) - new Date(a)) / 86400000);
  if (days < 0) out.value = "Discharge before admission";
  else out.value = `${Math.max(days, 1)} day${Math.max(days, 1) === 1 ? "" : "s"}${days > formOptions.max_stay_days ? " (too long for the model)" : ""}`;
}

async function saveEncounter(event) {
  event.preventDefault();
  const payload = {};
  new FormData(event.target).forEach((value, key) => {
    const v = String(value).trim();
    if (NUMERIC_FIELDS.includes(key)) payload[key] = v === "" ? null : Number(v);
    else payload[key] = v === "" ? null : v;
  });

  const btn = document.getElementById("btnSaveEncounter");
  btn.disabled = true;
  btn.textContent = "Assessing…";
  try {
    const encounter = await api(`/ehr/patients/${encodeURIComponent(patientId)}/encounters`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await loadPatient();
    selectEncounter(encounter.id);
    loadNavBadge();
    toast(`Saved — ${encounter.prediction.risk_level} (${encounter.prediction.readmission_risk_percentage})`);
  } catch (err) {
    showErrors(document.getElementById("encounterError"), err.detail, ENCOUNTER_FIELD_LABELS);
  } finally {
    btn.disabled = false;
    btn.textContent = "Save & assess risk";
  }
}

// ---------- Patient ----------

async function loadPatient() {
  patient = await api(`/ehr/patients/${encodeURIComponent(patientId)}`);
  renderBanner();
  renderAdmissionList();
}

async function deletePatient() {
  const name = `${patient.first_name} ${patient.last_name}`;
  if (!confirm(`Delete ${name} and all ${patient.encounters.length} admission(s)? This cannot be undone.`)) return;
  await api(`/ehr/patients/${encodeURIComponent(patientId)}`, { method: "DELETE" });
  window.location.href = "index.html#patients";
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    formOptions = await api("/ehr/options");
    await loadPatient();
  } catch (err) {
    document.getElementById("notFound").classList.remove("hidden");
    document.getElementById("banner").classList.add("hidden");
    document.getElementById("chartLayout").classList.add("hidden");
    return;
  }

  buildFormControls();
  document.getElementById("admList").addEventListener("click", (e) => {
    const item = e.target.closest(".adm-item");
    if (item) selectEncounter(item.dataset.id);
  });
  document.getElementById("btnNewAdmission").addEventListener("click", showForm);
  document.getElementById("btnCancelForm").addEventListener("click", hideForm);
  document.getElementById("encounterForm").addEventListener("submit", saveEncounter);

  if (params.get("new") || !patient.encounters.length) showForm();
  else selectEncounter(patient.encounters.some((x) => x.id === selectedId) ? selectedId : patient.encounters[0].id);
});
