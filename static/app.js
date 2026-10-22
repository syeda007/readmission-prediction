// ReadmitAI - Clinical Assessment Frontend Client

const PRESET_PROFILES = {
  high_risk: {
    age: "[70-80)",
    gender: "Female",
    race: "Caucasian",
    admission_type_id: "1", // Emergency
    admission_source_id: "7", // Emergency Room
    discharge_disposition_id: "3", // SNF
    time_in_hospital: 6,
    number_inpatient: 3,
    number_emergency: 2,
    number_outpatient: 1,
    num_medications: 22,
    num_lab_procedures: 58,
    num_procedures: 1,
    number_diagnoses: 9,
    diag_1: "428", // Congestive Heart Failure
    diag_2: "250.02", // Diabetes with ketoacidosis / complication
    diag_3: "585", // Chronic kidney disease
    A1Cresult: "None", // Unmonitored
    max_glu_serum: ">200",
    insulin: "Up", // Dosage increased
    metformin: "No",
    change: "Ch",
    diabetesMed: "Yes",
    patient_notes: "Elderly patient with heart failure and renal insufficiency. Frequent ED bounce-backs in past 6 months. High fall risk."
  },
  moderate_risk: {
    age: "[60-70)",
    gender: "Male",
    race: "AfricanAmerican",
    admission_type_id: "1", // Emergency
    admission_source_id: "7", // Emergency Room
    discharge_disposition_id: "6", // Home with home health
    time_in_hospital: 4,
    number_inpatient: 1,
    number_emergency: 0,
    number_outpatient: 2,
    num_medications: 15,
    num_lab_procedures: 42,
    num_procedures: 2,
    number_diagnoses: 7,
    diag_1: "414", // Coronary atherosclerosis
    diag_2: "250", // Diabetes
    diag_3: "401", // Hypertension
    A1Cresult: ">7",
    max_glu_serum: "Norm",
    insulin: "Steady",
    metformin: "Steady",
    change: "No",
    diabetesMed: "Yes",
    patient_notes: "Discharged to home with home physical therapy and glucose monitoring nurse."
  },
  low_risk: {
    age: "[40-50)",
    gender: "Female",
    race: "Caucasian",
    admission_type_id: "3", // Elective
    admission_source_id: "1", // Physician referral
    discharge_disposition_id: "1", // Home with self-care
    time_in_hospital: 2,
    number_inpatient: 0,
    number_emergency: 0,
    number_outpatient: 0,
    num_medications: 8,
    num_lab_procedures: 25,
    num_procedures: 1,
    number_diagnoses: 4,
    diag_1: "250.00", // Uncomplicated type 2 diabetes
    diag_2: "401", // Essential hypertension
    diag_3: "272", // Hyperlipidemia
    A1Cresult: "Norm",
    max_glu_serum: "Norm",
    insulin: "No",
    metformin: "Steady",
    change: "No",
    diabetesMed: "Yes",
    patient_notes: "Elective check-up. Well controlled HbA1c. Follow-up scheduled with primary care physician in 30 days."
  },
  ineligible: {
    age: "[80-90)",
    gender: "Male",
    race: "Caucasian",
    admission_type_id: "1",
    admission_source_id: "7",
    discharge_disposition_id: "11", // Expired / Hospice
    time_in_hospital: 5,
    number_inpatient: 2,
    number_emergency: 1,
    number_outpatient: 0,
    num_medications: 14,
    num_lab_procedures: 38,
    num_procedures: 0,
    number_diagnoses: 8,
    diag_1: "486", // Pneumonia
    diag_2: "428",
    diag_3: "250",
    A1Cresult: "None",
    max_glu_serum: "None",
    insulin: "No",
    metformin: "No",
    change: "No",
    diabetesMed: "No",
    patient_notes: "Patient expired during inpatient stay. Disposition ID 11."
  }
};

function loadProfile(key) {
  const data = PRESET_PROFILES[key];
  if (!data) return;

  for (const [field, value] of Object.entries(data)) {
    const el = document.getElementById(field);
    if (el) {
      el.value = value;
    }
  }

  // Automatically trigger assessment on preset selection
  const form = document.getElementById("patientForm");
  if (form) {
    handleAssessment(new Event("submit"));
  }
}

function resetForm() {
  document.getElementById("patientForm").reset();
  document.getElementById("emptyState").classList.remove("hidden");
  document.getElementById("activeResults").classList.add("hidden");
  document.getElementById("assessmentTimestamp").innerText = "Awaiting Assessment";
}

async function handleAssessment(event) {
  if (event) event.preventDefault();

  const form = document.getElementById("patientForm");
  const formData = new FormData(form);
  const payload = {};

  formData.forEach((value, key) => {
    // Cast numeric fields
    if ([
      "time_in_hospital", "num_lab_procedures", "num_procedures",
      "num_medications", "number_outpatient", "number_emergency",
      "number_inpatient", "number_diagnoses"
    ].includes(key)) {
      payload[key] = parseInt(value, 10);
    } else {
      payload[key] = value;
    }
  });

  const btn = document.getElementById("btnAssess");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Running XGBoost & SHAP...`;

  try {
    const response = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      displayEligibilityError(data.detail || "Patient encounter cannot be assessed.");
      return;
    }

    displayResults(data);
  } catch (err) {
    displayEligibilityError("Network or server connection failed. Please ensure the backend is running.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
      Assess Readmission Risk & Explain
    `;
  }
}

function displayEligibilityError(message) {
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("activeResults").classList.remove("hidden");

  const banner = document.getElementById("eligibilityBanner");
  banner.className = "alert-banner danger";
  banner.innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    <div>
      <strong>Cohort Exclusion Rule:</strong> ${message}
    </div>
  `;
  banner.classList.remove("hidden");

  // Reset gauges
  document.getElementById("riskPercentage").innerText = "N/A";
  document.getElementById("riskProbabilityRaw").innerText = "Ineligible encounter";
  document.getElementById("riskBar").style.width = "0%";
  document.getElementById("reviewBadge").className = "status-badge";
  document.getElementById("reviewBadge").innerText = "EXCLUDED";
  document.getElementById("tierBadge").className = "tier-badge";
  document.getElementById("tierBadge").innerText = "Ineligible Cohort";
  document.getElementById("recText").innerText = "No acute readmission risk assessment applicable for deceased or hospice discharges.";
  document.getElementById("increasingFactors").innerHTML = "<div class='factor-item'>Discharge excluded from acute 30-day readmission prediction.</div>";
  document.getElementById("decreasingFactors").innerHTML = "";
}

function displayResults(data) {
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("activeResults").classList.remove("hidden");
  document.getElementById("eligibilityBanner").classList.add("hidden");

  const prob = data.readmission_probability;
  const pctStr = data.readmission_risk_percentage;
  const requiresReview = data.requires_review;
  const riskLevel = data.risk_level;

  // Timestamp
  const now = new Date();
  document.getElementById("assessmentTimestamp").innerText = `Assessed ${now.toLocaleTimeString()}`;

  // Gauge
  document.getElementById("riskPercentage").innerText = pctStr;
  document.getElementById("riskProbabilityRaw").innerText = `Calibrated Prob: ${prob.toFixed(4)}`;

  const bar = document.getElementById("riskBar");
  const barPct = Math.min(Math.round(prob * 100 * 2), 100); // Scale up for visual clarity up to 50%
  bar.style.width = `${barPct}%`;

  if (prob < 0.12) {
    bar.className = "progress-bar low";
  } else if (prob < 0.22) {
    bar.className = "progress-bar moderate";
  } else {
    bar.className = "progress-bar high";
  }

  // Review Status Badge
  const reviewBadge = document.getElementById("reviewBadge");
  if (requiresReview) {
    reviewBadge.className = "status-badge review-required";
    reviewBadge.innerText = "CLINICAL REVIEW REQUIRED";
  } else {
    reviewBadge.className = "status-badge routine";
    reviewBadge.innerText = "ROUTINE DISCHARGE";
  }

  // Risk Tier Badge
  const tierBadge = document.getElementById("tierBadge");
  tierBadge.innerText = riskLevel;
  if (riskLevel.includes("Low")) {
    tierBadge.className = "tier-badge low";
  } else if (riskLevel.includes("Moderate")) {
    tierBadge.className = "tier-badge moderate";
  } else {
    tierBadge.className = "tier-badge high";
  }

  // Recommendation
  const recBox = document.querySelector(".recommendation-box");
  recBox.className = "recommendation-box";
  if (prob >= 0.22) recBox.classList.add("high");
  else if (prob >= 0.12) recBox.classList.add("moderate");

  document.getElementById("recText").innerText = data.clinical_recommendation;

  // SHAP Factors
  renderShapFactors(data.explanation);
}

function renderShapFactors(explanation) {
  const incContainer = document.getElementById("increasingFactors");
  const decContainer = document.getElementById("decreasingFactors");

  incContainer.innerHTML = "";
  decContainer.innerHTML = "";

  if (!explanation) {
    incContainer.innerHTML = "<div class='factor-item'>SHAP explanation unavailable.</div>";
    return;
  }

  const increasing = explanation.risk_increasing_factors || [];
  const decreasing = explanation.risk_decreasing_factors || [];

  if (increasing.length === 0) {
    incContainer.innerHTML = "<div class='factor-item' style='color:#64748B;'>No substantial risk-increasing factors identified.</div>";
  } else {
    increasing.forEach(f => {
      const item = document.createElement("div");
      item.className = "factor-item red";
      item.innerHTML = `
        <span class="factor-desc">${f.description}</span>
        <span class="factor-val red">+${f.shap_value.toFixed(3)}</span>
      `;
      incContainer.appendChild(item);
    });
  }

  if (decreasing.length === 0) {
    decContainer.innerHTML = "<div class='factor-item' style='color:#64748B;'>No substantial protective factors identified.</div>";
  } else {
    decreasing.forEach(f => {
      const item = document.createElement("div");
      item.className = "factor-item green";
      item.innerHTML = `
        <span class="factor-desc">${f.description}</span>
        <span class="factor-val green">${f.shap_value.toFixed(3)}</span>
      `;
      decContainer.appendChild(item);
    });
  }

  if (explanation.clinical_disclaimer) {
    document.getElementById("shapDisclaimer").innerText = explanation.clinical_disclaimer;
  }
}

function toggleMetricsModal() {
  const modal = document.getElementById("metricsModal");
  modal.classList.toggle("hidden");
}

// Load default High Risk profile on initial start
window.addEventListener("DOMContentLoaded", () => {
  loadProfile("high_risk");
});
