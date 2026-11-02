// ReadmitAI EHR - shared helpers used by every page

const EHR = (() => {
  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (response.status === 204) return null;
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error("Request failed");
      error.detail = data.detail;
      throw error;
    }
    return data;
  }

  // Escape any value before it goes into an HTML template
  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function tierClass(riskLevel) {
    if (!riskLevel) return "";
    if (riskLevel.startsWith("High")) return "high";
    if (riskLevel.startsWith("Moderate")) return "moderate";
    return "low";
  }

  function tierBadge(riskLevel) {
    return `<span class="tier ${tierClass(riskLevel)}">${esc(riskLevel)}</span>`;
  }

  function statusClass(status) {
    if (status === "Pending Review") return "pending";
    if (status === "Contacted" || status === "Follow-up Scheduled") return "active";
    if (status === "Closed") return "closed";
    return "";
  }

  function statusBadge(status) {
    return `<span class="status ${statusClass(status)}">${esc(status)}</span>`;
  }

  // Risk % with a small meter; the meter is scaled so 50% fills it (risks above that are rare)
  function riskPill(probability, riskLevel) {
    const width = Math.min(probability * 200, 100);
    return `<span class="risk-pill">
      <span class="risk-meter"><span class="fill-${tierClass(riskLevel)}" style="width:${width}%"></span></span>
      ${(probability * 100).toFixed(1)}%
    </span>`;
  }

  function formatDate(iso) {
    if (!iso) return "—";
    const [y, m, d] = iso.split("-").map(Number);
    return new Date(y, m - 1, d).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }

  function formatAge(bracket) {
    return bracket ? bracket.replace("[", "").replace(")", "").replace("-", "–") : "—";
  }

  function yearsSince(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    const today = new Date();
    let years = today.getFullYear() - y;
    if (today.getMonth() + 1 < m || (today.getMonth() + 1 === m && today.getDate() < d)) years -= 1;
    return years;
  }

  // Turns a FastAPI error body into a readable list of messages
  function errorMessages(detail, fieldLabels = {}) {
    if (!detail) return ["Something went wrong. Is the server running?"];
    if (typeof detail === "string") return [detail];
    return detail.map((err) => {
      const field = err.loc ? err.loc[err.loc.length - 1] : "";
      const label = fieldLabels[field] || (field === "body" || typeof field === "number" ? "" : field);
      const msg = (err.msg || "").replace(/^Value error, /, "");
      return label ? `${label}: ${msg}` : msg;
    });
  }

  function showErrors(container, detail, fieldLabels) {
    const items = errorMessages(detail, fieldLabels).map((m) => `<li>${esc(m)}</li>`).join("");
    container.innerHTML = `<strong>Please fix the following:</strong><ul>${items}</ul>`;
    container.classList.remove("hidden");
    container.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function toast(message) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2500);
  }

  function options(list, selected) {
    return list
      .map((o) => {
        const value = typeof o === "string" ? o : o.value;
        const label = typeof o === "string" ? o : o.label;
        return `<option value="${esc(value)}" ${String(value) === String(selected) ? "selected" : ""}>${esc(label)}</option>`;
      })
      .join("");
  }

  // ---------- Follow-up tasks ----------

  const KIND_ICONS = {
    call: '<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1.9.4 1.8.7 2.7a2 2 0 0 1-.5 2.1L8 9.8a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.7.7a2 2 0 0 1 1.7 2z"/>',
    medication: '<path d="m10.5 20.5 10-10a4.9 4.9 0 1 0-7-7l-10 10a4.9 4.9 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/>',
    visit: '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
  };

  function kindIcon(kind) {
    return `<span class="kind-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${KIND_ICONS[kind] || KIND_ICONS.visit}</svg></span>`;
  }

  function dueLabel(task) {
    if (task.status === "done") return `Done ${task.completed_at ? new Date(task.completed_at).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : ""}`;
    const d = task.days_until_due;
    if (d < -1) return `Overdue by ${-d} days`;
    if (d === -1) return "Overdue since yesterday";
    if (d === 0) return "Due today";
    if (d === 1) return "Due tomorrow";
    return `Due ${formatDate(task.due_date)}`;
  }

  function attemptClass(outcome) {
    return { Reached: "reached", "No answer": "missed", "Left voicemail": "missed", "Wrong number": "wrong" }[outcome] || "";
  }

  // Count of overdue + due-today follow-ups next to the "Follow-ups" menu item
  async function loadNavBadge() {
    const badge = document.getElementById("navFollowupsBadge");
    if (!badge) return;
    try {
      const s = await api("/ehr/followups/summary");
      const n = s.overdue + s.due_today;
      badge.textContent = n;
      badge.classList.toggle("hidden", n === 0);
    } catch (err) {
      badge.classList.add("hidden");
    }
  }

  window.addEventListener("DOMContentLoaded", loadNavBadge);

  return {
    api, esc, tierClass, tierBadge, statusBadge, riskPill,
    formatDate, formatAge, yearsSince, errorMessages, showErrors, toast, options,
    kindIcon, dueLabel, attemptClass, loadNavBadge,
  };
})();
