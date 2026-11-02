// ReadmitAI EHR - discharge worklist & patient registry

const { api, esc, tierBadge, riskPill, formatDate, formatAge, showErrors, toast, options } = EHR;

let formOptions = null;

const PATIENT_FIELD_LABELS = {
  first_name: "First name",
  last_name: "Last name",
  date_of_birth: "Date of birth",
  gender: "Gender",
  race: "Race",
  payer_code: "Insurance",
  phone: "Phone",
};

// ---------- Summary ----------

async function loadSummary() {
  const s = await api("/ehr/summary");
  document.getElementById("statPatients").textContent = s.patients;
  document.getElementById("statEncounters").textContent = `${s.encounters} admissions recorded`;
  document.getElementById("statFlagged").textContent = s.flagged;
  document.getElementById("statPending").textContent = s.pending_review;

  const total = s.encounters || 1;
  const parts = [
    ["High Risk", "fill-high"],
    ["Moderate Risk", "fill-moderate"],
    ["Low Risk", "fill-low"],
  ];
  document.getElementById("tierBar").innerHTML = parts
    .map(([tier, cls]) => `<span class="${cls}" style="width:${(s.tiers[tier] / total) * 100}%"></span>`)
    .join("");
  document.getElementById("tierLegend").innerHTML = parts
    .map(([tier, cls]) => `<span><i class="${cls}"></i>${esc(tier.replace(" Risk", ""))} ${s.tiers[tier]}</span>`)
    .join("");
}

// ---------- Worklist ----------

async function loadWorklist() {
  const params = new URLSearchParams();
  const q = document.getElementById("wlSearch").value.trim();
  const tier = document.getElementById("wlTier").value;
  const status = document.getElementById("wlStatus").value;
  if (q) params.set("q", q);
  if (tier) params.set("tier", tier);
  if (status) params.set("status", status);
  if (document.getElementById("wlFlagged").checked) params.set("flagged_only", "true");

  const rows = await api(`/ehr/worklist?${params}`);
  const body = document.getElementById("wlBody");
  document.getElementById("wlEmpty").classList.toggle("hidden", rows.length > 0);

  body.innerHTML = rows
    .map(
      (r) => `
      <tr class="clickable" data-patient="${esc(r.patient_id)}" data-encounter="${esc(r.encounter_id)}">
        <td>${riskPill(r.readmission_probability, r.risk_level)}</td>
        <td><div class="cell-main">${esc(r.patient_name)}</div><div class="cell-sub mono">${esc(r.mrn)}</div></td>
        <td class="num">${esc(formatAge(r.age))}</td>
        <td class="num">${esc(formatDate(r.discharge_date))}</td>
        <td>${esc(r.discharge_disposition)}</td>
        <td class="dx-cell" title="${esc(r.primary_diagnosis_description)}">
          <div class="dx-desc">${esc(r.primary_diagnosis_description)}</div>
          <div class="cell-sub mono">ICD-9 ${esc(r.primary_diagnosis)}</div>
        </td>
        <td>${tierBadge(r.risk_level)}</td>
        <td>
          <select class="status-select" data-encounter="${esc(r.encounter_id)}" aria-label="Review status">
            ${options(formOptions.review_statuses, r.review_status)}
          </select>
        </td>
      </tr>`
    )
    .join("");
}

async function updateStatus(encounterId, status) {
  await api(`/ehr/encounters/${encodeURIComponent(encounterId)}/review`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
  toast(`Status set to “${status}”`);
  loadSummary();
}

// ---------- Patient registry ----------

async function loadPatients() {
  const q = document.getElementById("ptSearch").value.trim();
  const patients = await api(`/ehr/patients${q ? `?q=${encodeURIComponent(q)}` : ""}`);
  document.getElementById("ptEmpty").classList.toggle("hidden", patients.length > 0);
  document.getElementById("ptBody").innerHTML = patients
    .map(
      (p) => `
      <tr class="clickable" data-patient="${esc(p.id)}">
        <td><div class="cell-main">${esc(p.last_name)}, ${esc(p.first_name)}</div><div class="cell-sub mono">${esc(p.mrn)}</div></td>
        <td class="num">${esc(formatDate(p.date_of_birth))}</td>
        <td>${esc(p.gender)}</td>
        <td class="num">${p.encounter_count}</td>
        <td class="num">${esc(formatDate(p.latest_discharge_date))}</td>
        <td>${p.latest_probability == null ? '<span class="muted">No admissions</span>' : riskPill(p.latest_probability, p.latest_risk_level)}</td>
      </tr>`
    )
    .join("");
}

// ---------- Tabs ----------

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.getElementById("tabWorklist").classList.toggle("hidden", name !== "worklist");
  document.getElementById("tabPatients").classList.toggle("hidden", name !== "patients");
  document.getElementById("navWorklist").classList.toggle("active", name === "worklist");
  document.getElementById("navPatients").classList.toggle("active", name === "patients");
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "patients") loadPatients();
  else loadWorklist();
}

// ---------- New patient modal ----------

function openPatientModal() {
  document.getElementById("patientForm").reset();
  document.getElementById("patientError").classList.add("hidden");
  document.getElementById("date_of_birth").max = new Date().toISOString().slice(0, 10);
  document.getElementById("patientModal").classList.remove("hidden");
  document.getElementById("first_name").focus();
}

function closePatientModal() {
  document.getElementById("patientModal").classList.add("hidden");
}

async function savePatient(event) {
  event.preventDefault();
  const form = event.target;
  const payload = Object.fromEntries(new FormData(form).entries());
  const btn = document.getElementById("btnSavePatient");
  btn.disabled = true;
  try {
    const patient = await api("/ehr/patients", { method: "POST", body: JSON.stringify(payload) });
    window.location.href = `patient.html?id=${encodeURIComponent(patient.id)}&new=1`;
  } catch (err) {
    showErrors(document.getElementById("patientError"), err.detail, PATIENT_FIELD_LABELS);
  } finally {
    btn.disabled = false;
  }
}

// ---------- Wiring ----------

function debounce(fn, ms = 250) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

window.addEventListener("DOMContentLoaded", async () => {
  formOptions = await api("/ehr/options");

  document.getElementById("wlStatus").innerHTML += options(formOptions.review_statuses);
  document.getElementById("race").innerHTML = `<option value="">Not recorded</option>${options(formOptions.races)}`;
  document.getElementById("payer_code").innerHTML = `<option value="">Not recorded</option>${options(formOptions.payer_codes)}`;
  document.getElementById("statThreshold").textContent = `risk ≥ ${(formOptions.decision_threshold * 100).toFixed(0)}%`;

  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  window.addEventListener("hashchange", () => showTab(location.hash === "#patients" ? "patients" : "worklist"));

  const reloadWorklist = debounce(loadWorklist);
  document.getElementById("wlSearch").addEventListener("input", reloadWorklist);
  ["wlTier", "wlStatus", "wlFlagged"].forEach((id) => document.getElementById(id).addEventListener("change", loadWorklist));
  document.getElementById("ptSearch").addEventListener("input", debounce(loadPatients));

  document.getElementById("wlBody").addEventListener("change", (e) => {
    if (e.target.classList.contains("status-select")) updateStatus(e.target.dataset.encounter, e.target.value);
  });
  document.getElementById("wlBody").addEventListener("click", (e) => {
    if (e.target.closest("select")) return;
    const row = e.target.closest("tr[data-patient]");
    if (row) window.location.href = `patient.html?id=${encodeURIComponent(row.dataset.patient)}&encounter=${encodeURIComponent(row.dataset.encounter)}`;
  });
  document.getElementById("ptBody").addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-patient]");
    if (row) window.location.href = `patient.html?id=${encodeURIComponent(row.dataset.patient)}`;
  });

  document.getElementById("btnNewPatient").addEventListener("click", openPatientModal);
  document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closePatientModal));
  document.getElementById("patientModal").addEventListener("click", (e) => {
    if (e.target.id === "patientModal") closePatientModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePatientModal();
  });
  document.getElementById("patientForm").addEventListener("submit", savePatient);

  loadSummary();
  showTab(location.hash === "#patients" ? "patients" : "worklist");
});
