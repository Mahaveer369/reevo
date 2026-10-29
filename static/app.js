(function () {
  "use strict";

  const PRESET_CODES = {
    reevo: "",
    mwkr: `def priority(features: dict) -> list:
    # Classical Most Work Remaining (MWKR) with SPT tie-breaker
    return [
        m - 0.01 * p
        for m, p in zip(features['rem_work_norm'], features['pt_norm'])
    ]`,
    winq: `def priority(features: dict) -> list:
    # Holthaus & Rajendran PT + WINQ combined with Remaining Work
    return [
        1.25 * m - 0.35 * p - 0.45 * w
        for m, p, w in zip(features['rem_work_norm'], features['pt_norm'], features['winq_norm'])
    ]`,
  };

  const JOB_COLORS = [
    "#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777",
    "#0891b2", "#4f46e5", "#65a30d", "#ea580c", "#0d9488",
    "#9333ea", "#ca8a04", "#0284c7", "#16a34a", "#e11d48",
    "#475569", "#8b5cf6", "#14b8a6", "#f97316", "#64748b"
  ];

  function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      const active = btn.getAttribute("data-tab") === tabId;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === "tab-" + tabId);
    });
  }

  function renderBenchmarkTables(data) {
    const tbody = document.getElementById("benchmark-summary-tbody");
    if (!tbody || !data || !data.summary_rows) return;

    tbody.innerHTML = "";
    const byId = {};

    data.summary_rows.forEach((row) => {
      byId[row.id] = row;
      const tr = document.createElement("tr");
      if (row.id === "multistart_reevo_cpls" || row.id === "custom_user_cpls") {
        tr.className = "row-highlight";
      } else if (row.id === "reevo_gt_single") {
        tr.className = "row-subhighlight";
      }
      const llmInfo = row.llm_calls > 0 ? `${row.llm_calls} / ${(row.llm_tokens / 1000).toFixed(1)}k` : "0 / 0";
      tr.innerHTML = `
        <td>${row.category}</td>
        <td>${row.name}</td>
        <td class="num">${row.mean_cmax.toFixed(1)}</td>
        <td class="num"><strong>${row.mean_gap.toFixed(2)}%</strong> &plusmn; ${row.sd_gap.toFixed(2)}%</td>
        <td class="num">${row.median_gap.toFixed(2)}%</td>
        <td class="num">${row.best_gap.toFixed(2)}%</td>
        <td class="num">${row.mean_runtime_ms.toFixed(2)}</td>
        <td class="num">${row.mean_evaluations.toFixed(0)}</td>
        <td class="num">${llmInfo}</td>
        <td class="num"><span class="badge-ok">${row.feasibility_rate.toFixed(0)}% OK</span></td>
      `;
      tbody.appendChild(tr);
    });

    if (byId.mwkr_gt) {
      document.getElementById("kpi-mwkr-gap").textContent = byId.mwkr_gt.mean_gap.toFixed(2) + "%";
    }
    if (byId.reevo_gt_single && byId.mwkr_gt) {
      document.getElementById("kpi-reevo-single-gap").textContent = byId.reevo_gt_single.mean_gap.toFixed(2) + "%";
      const delta = (byId.reevo_gt_single.mean_gap - byId.mwkr_gt.mean_gap).toFixed(2);
      document.getElementById("kpi-reevo-single-sub").textContent =
        `${delta}% gap change vs MWKR (Single-Pass Giffler–Thompson)`;
    }
    if (byId.multistart_mwkr_cpls) {
      document.getElementById("kpi-ms-mwkr-gap").textContent = byId.multistart_mwkr_cpls.mean_gap.toFixed(2) + "%";
    }
    if (byId.multistart_reevo_cpls) {
      document.getElementById("kpi-ms-reevo-gap").textContent = byId.multistart_reevo_cpls.mean_gap.toFixed(2) + "%";
    }

    const instTbody = document.getElementById("per-instance-tbody");
    if (!instTbody || !data.per_instance) return;
    instTbody.innerHTML = "";

    data.per_instance.forEach((inst) => {
      const s = inst.solvers || {};
      const fmtCell = (key) => {
        if (!s[key]) return "-";
        return `${s[key].best_cmax} <span style="color:#64748b">(${s[key].best_gap.toFixed(1)}%)</span>`;
      };
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${inst.name}</strong></td>
        <td>${inst.family}</td>
        <td>${inst.size}</td>
        <td class="num">${inst.optimum}</td>
        <td class="num">${fmtCell("spt_gt")}</td>
        <td class="num">${fmtCell("mwkr_gt")}</td>
        <td class="num"><strong>${fmtCell("reevo_gt_single")}</strong></td>
        <td class="num">${fmtCell("multistart_mwkr_cpls")}</td>
        <td class="num" style="background:#f0fdf4"><strong>${fmtCell("multistart_reevo_cpls")}</strong></td>
        <td>
          <button type="button" class="secondary-btn inspect-inst-btn" data-instance="${inst.name}">
            Inspect Gantt
          </button>
        </td>
      `;
      instTbody.appendChild(tr);
    });

    instTbody.querySelectorAll(".inspect-inst-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const instName = btn.getAttribute("data-instance");
        const sel = document.getElementById("gantt-instance-select");
        if (sel) sel.value = instName;
        switchTab("gantt");
        loadScheduleView(instName, document.getElementById("gantt-solver-select").value);
      });
    });
  }

  function renderScheduleView(sched) {
    if (!sched || !sched.instance) return;
    const summaryBar = document.getElementById("schedule-summary-bar");
    const v = sched.validation || {};
    summaryBar.innerHTML = `
      <div><strong>Instance:</strong> ${sched.instance.name} (${sched.instance.n_jobs}&times;${sched.instance.n_machines})</div>
      <div><strong>Optimum (<i>C</i>*):</strong> ${sched.instance.optimum}</div>
      <div><strong>Achieved <i>C</i><sub>max</sub>:</strong> <span style="color:#059669;font-weight:700">${sched.makespan}</span> (${sched.gap_percent.toFixed(2)}% gap)</div>
      <div><strong>vs. Single-Pass MWKR (${sched.mwkr_baseline_makespan}):</strong> ${sched.improvement_over_mwkr >= 0 ? "-" + sched.improvement_over_mwkr : "+" + Math.abs(sched.improvement_over_mwkr)} time units</div>
      <div><strong>Critical Path Ops:</strong> ${sched.critical_op_count} (${sched.critical_block_count} critical blocks)</div>
      <div><strong>Runtime:</strong> ${sched.runtime_ms.toFixed(2)} ms (${sched.evaluations} evals)</div>
      <div><span class="badge-ok">${v.feasible ? "FEASIBLE (100% Verified)" : "INFEASIBLE"}</span></div>
    `;

    // Render SVG Gantt Chart
    const container = document.getElementById("gantt-chart-container");
    const nMachines = sched.instance.n_machines;
    const cmax = Math.max(1, sched.makespan);
    const leftPad = 76;
    const rightPad = 28;
    const topPad = 28;
    const rowHeight = 34;
    const chartWidth = 920;
    const plotWidth = chartWidth - leftPad - rightPad;
    const chartHeight = topPad + nMachines * rowHeight + 32;

    let svg = `<svg viewBox="0 0 ${chartWidth} ${chartHeight}" width="100%" height="${chartHeight}" role="img" aria-label="JSSP Gantt Schedule for ${sched.instance.name}">`;

    // Time grid ticks
    const nTicks = 10;
    for (let t = 0; t <= nTicks; t++) {
      const val = Math.round((cmax * t) / nTicks);
      const x = leftPad + (val / cmax) * plotWidth;
      svg += `<line x1="${x}" y1="${topPad - 6}" x2="${x}" y2="${topPad + nMachines * rowHeight}" stroke="#e2e8f0" stroke-width="1" />`;
      svg += `<text x="${x}" y="${topPad - 10}" text-anchor="middle" font-size="11" fill="#64748b" font-family="monospace">${val}</text>`;
    }

    // Machine rows
    for (let m = 0; m < nMachines; m++) {
      const y = topPad + m * rowHeight;
      svg += `<rect x="${leftPad}" y="${y + 3}" width="${plotWidth}" height="${rowHeight - 6}" fill="${m % 2 === 0 ? "#f8fafc" : "#ffffff"}" stroke="#f1f5f9" />`;
      svg += `<text x="${leftPad - 10}" y="${y + rowHeight / 2 + 4}" text-anchor="end" font-size="12" font-weight="600" fill="#334155" font-family="monospace">M${m}</text>`;
    }

    // Operations
    sched.operations.forEach((op) => {
      const x = leftPad + (op.start / cmax) * plotWidth;
      const w = Math.max(2, (op.duration / cmax) * plotWidth);
      const y = topPad + op.machine * rowHeight + 5;
      const h = rowHeight - 10;
      const fill = JOB_COLORS[op.job % JOB_COLORS.length];
      const stroke = op.is_critical ? "#dc2626" : "#ffffff";
      const strokeW = op.is_critical ? 2.2 : 0.8;

      svg += `<g>
        <rect x="${x.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="${h}" rx="3" fill="${fill}" fill-opacity="0.88" stroke="${stroke}" stroke-width="${strokeW}">
          <title>Job J${op.job} Op ${op.op} on M${op.machine}: [${op.start}, ${op.end}] (p=${op.duration})${op.is_critical ? " [CRITICAL PATH]" : ""}</title>
        </rect>`;
      if (w >= 22) {
        svg += `<text x="${(x + w / 2).toFixed(1)}" y="${y + h / 2 + 4}" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff" font-family="monospace">J${op.job}</text>`;
      }
      svg += `</g>`;
    });

    svg += `</svg>`;
    container.innerHTML = svg;

    // Render Validator Checklist
    const vList = document.getElementById("validator-checklist");
    vList.innerHTML = `
      <li><strong>Operation Completeness (${v.total_operations_checked} ops):</strong> ${v.completeness_ok ? "PASS (All operations scheduled once)" : "FAIL"}</li>
      <li><strong>Job Precedence Constraints (<i>S<sub>j,k</sub></i> &ge; <i>C<sub>j,k&minus;1</sub></i>):</strong> ${v.precedence_ok ? "PASS (Zero precedence violations)" : "FAIL"}</li>
      <li><strong>Machine Disjunctive Non-Overlap:</strong> ${v.machine_non_overlap_ok ? "PASS (Zero machine overlaps)" : "FAIL"}</li>
      <li><strong>Exact Processing Durations (<i>C<sub>j,k</sub> &minus; S<sub>j,k</sub> = p<sub>j,k</sub></i>):</strong> ${v.duration_exact_ok ? "PASS" : "FAIL"}</li>
    `;

    // Render Critical Blocks
    const bList = document.getElementById("critical-blocks-list");
    if (!sched.critical_blocks || sched.critical_blocks.length === 0) {
      bList.innerHTML = `<div class="block-item">No multi-operation critical blocks (all critical transitions are cross-machine job arcs).</div>`;
    } else {
      bList.innerHTML = sched.critical_blocks
        .map((blk, idx) => {
          const opsStr = blk.operations.map((o) => `J${o.job}:Op${o.op}`).join(" &rarr; ");
          return `<div class="block-item"><strong>Block #${idx + 1} (Machine M${blk.machine}):</strong> ${opsStr}</div>`;
        })
        .join("");
    }
  }

  async function loadScheduleView(instanceName, solverId) {
    try {
      const resp = await fetch(`/api/schedule?instance=${encodeURIComponent(instanceName)}&solver=${encodeURIComponent(solverId)}`);
      const data = await resp.json();
      if (resp.ok) {
        renderScheduleView(data);
      }
    } catch (err) {
      console.error("Failed to load schedule:", err);
    }
  }

  async function initWorkbench() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
    });

    try {
      const resp = await fetch("/api/initial-state");
      const state = await resp.json();
      PRESET_CODES.reevo = state.default_code || "";
      const codeArea = document.getElementById("custom-code-textarea");
      if (codeArea) {
        codeArea.value = PRESET_CODES.reevo;
      }
      renderBenchmarkTables(state.benchmark);
      renderScheduleView(state.schedule);
    } catch (err) {
      console.error("Error initializing workbench:", err);
    }

    const benchForm = document.getElementById("benchmark-controls-form");
    if (benchForm) {
      benchForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("run-benchmark-btn");
        btn.disabled = true;
        btn.textContent = "Running Controlled Suite...";
        try {
          const payload = {
            subset: document.getElementById("subset-select").value,
            n_starts: parseInt(document.getElementById("starts-input").value, 10),
            ls_steps: parseInt(document.getElementById("ls-steps-input").value, 10),
            n_seeds: 3,
          };
          const resp = await fetch("/api/run-benchmark", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await resp.json();
          if (resp.ok) {
            renderBenchmarkTables(data);
          }
        } finally {
          btn.disabled = false;
          btn.textContent = "Run Controlled Experiment";
        }
      });
    }

    const ganttForm = document.getElementById("gantt-controls-form");
    if (ganttForm) {
      ganttForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const inst = document.getElementById("gantt-instance-select").value;
        const solver = document.getElementById("gantt-solver-select").value;
        await loadScheduleView(inst, solver);
      });
    }

    document.querySelectorAll("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-preset");
        const area = document.getElementById("custom-code-textarea");
        if (area && PRESET_CODES[key]) {
          area.value = PRESET_CODES[key];
        }
      });
    });

    const evalBtn = document.getElementById("eval-custom-btn");
    if (evalBtn) {
      evalBtn.addEventListener("click", async () => {
        const code = document.getElementById("custom-code-textarea").value;
        const statusEl = document.getElementById("sandbox-status-msg");
        const outBox = document.getElementById("custom-eval-output");
        statusEl.textContent = "Compiling & running 6-instance benchmark...";
        statusEl.style.color = "#0284c7";
        evalBtn.disabled = true;
        try {
          const resp = await fetch("/api/evaluate-custom", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              custom_code: code,
              instance: "la01",
              n_starts: 8,
              ls_steps: 30,
            }),
          });
          const data = await resp.json();
          if (!resp.ok || data.error) {
            statusEl.textContent = "Compilation / Validation Error";
            statusEl.style.color = "#dc2626";
            outBox.innerHTML = `<div style="color:#dc2626;font-family:monospace">${data.error || "Unknown error"}</div>`;
            return;
          }
          statusEl.textContent = "Verified 100% Feasible!";
          statusEl.style.color = "#059669";
          renderBenchmarkTables(data.benchmark);
          renderScheduleView(data.schedule);

          const customRow = (data.benchmark.summary_rows || []).find((r) => r.id === "custom_user_cpls");
          const mwkrRow = (data.benchmark.summary_rows || []).find((r) => r.id === "multistart_mwkr_cpls");
          if (customRow && mwkrRow) {
            outBox.innerHTML = `
              <div><strong>Custom Heuristic Result (6 Canonical Instances):</strong></div>
              <div style="margin-top:6px;font-family:monospace">
                &bull; Custom Priority + CP-LS Mean Gap: <strong>${customRow.mean_gap.toFixed(2)}%</strong> (Mean Cmax: ${customRow.mean_cmax.toFixed(1)})<br/>
                &bull; Multi-Start MWKR + CP-LS Mean Gap: <strong>${mwkrRow.mean_gap.toFixed(2)}%</strong><br/>
                &bull; Schedule Feasibility Rate: <span class="badge-ok">${customRow.feasibility_rate.toFixed(0)}% OK</span>
              </div>
              <div style="margin-top:8px;font-size:12px;color:#334155">
                Table 1 (Tab 1) and the Gantt Inspector (Tab 2) have been updated with your custom heuristic.
              </div>
            `;
          }
        } catch (err) {
          statusEl.textContent = "Request failed";
          statusEl.style.color = "#dc2626";
        } finally {
          evalBtn.disabled = false;
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", initWorkbench);
})();
