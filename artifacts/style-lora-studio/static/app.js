const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  step: 0,
  jobId: null,
  analyze: null,
  distill: null,
  defaults: null,
  evalIndex: 0,
};

function log(msg) {
  const el = $("#log");
  const t = new Date().toLocaleTimeString();
  el.textContent = `[${t}] ${msg}\n` + el.textContent;
}

function busy(on, msg) {
  $("#busy").classList.toggle("hidden", !on);
  if (msg) $("#busyMsg").textContent = msg;
}

function lines(el) {
  return el.value.split("\n").map((s) => s.trim()).filter(Boolean);
}

function insertStyle(content, phrase) {
  const c = content.trim();
  const p = phrase.trim();
  if (c.toLowerCase().startsWith("a ")) return `a ${p} ${c.slice(2).trimStart()}`;
  return `${p} ${c}`;
}

function spec() {
  return {
    style_name: $("#styleName").value.trim() || "style",
    style_phrases: lines($("#phrases")),
    extraction_contents: lines($("#extract")),
    evaluation_contents: lines($("#evalset")),
    pairing: $("#pairing").value,
    mode: $("#mode").value,
    te_path: $("#tePath").value.trim(),
    raw_path: $("#rawPath").value.trim(),
    te_template: ($("#teTemplate") && $("#teTemplate").value.trim()) || "",
    layers: $$("input.layer:checked").map((x) => x.value),
    rank: parseInt($("#rank").value, 10),
    alpha: parseFloat($("#alpha").value),
    method: $("#method").value,
    export_format: $("#fmt").value,
    cosine_gate: parseFloat($("#cosGate").value) || 0.4,
    pca_gate: parseFloat($("#pcaGate").value) || 0.35,
    include_control: true,
    export_kohya: $("#exportKohya").checked,
    include_negative: $("#includeNeg").checked,
    include_random: $("#includeRnd").checked,
    include_rank1: $("#includeRank1") ? $("#includeRank1").checked : true,
  };
}

function goto(n) {
  state.step = n;
  $$(".steps button").forEach((b, i) => b.classList.toggle("on", i === n));
  $$(".panel").forEach((p, i) => p.classList.toggle("hidden", i !== n));
}

function applyPreset(p) {
  $("#styleName").value = p.name;
  $("#phrases").value = p.phrases.join("\n");
  $$("#presets .chip").forEach((c) => c.classList.toggle("on", c.dataset.id === p.id));
  renderPairPreview();
  log(`載入預設「${p.label}」`);
}

function renderPresets(presets) {
  const host = $("#presets");
  host.innerHTML = "";
  (presets || []).forEach((p) => {
    const b = document.createElement("button");
    b.className = "chip" + (p.id === "unknown" ? " warn" : "") + (p.id === "watercolor" ? " on" : "");
    b.dataset.id = p.id;
    b.textContent = p.label;
    b.title = p.note || "";
    b.addEventListener("click", () => applyPreset(p));
    host.appendChild(b);
  });
}

function renderPairPreview() {
  const contents = lines($("#extract"));
  const phrases = lines($("#phrases"));
  const cartesian = $("#pairing").value === "cartesian";
  const n = cartesian ? contents.length * Math.max(phrases.length, 1) : contents.length;
  $("#pairCount").textContent = `${n} 組`;
  const rows = [];
  const limit = 6;
  let k = 0;
  outer: for (let i = 0; i < contents.length; i++) {
    const phs = cartesian ? phrases : [phrases[i % Math.max(phrases.length, 1)] || ""];
    for (const ph of phs) {
      rows.push(
        `<div class="tr"><span>${contents[i]}</span><span>${insertStyle(contents[i], ph)}</span></div>`
      );
      k += 1;
      if (k >= limit) break outer;
    }
  }
  $("#pairPreview").innerHTML =
    `<div class="tr head"><span>P<sub>c</sub></span><span>P<sub>c+s</sub></span></div>` + rows.join("");
}

async function loadDefaults() {
  const d = await (await fetch("/api/defaults")).json();
  state.defaults = d;
  $("#styleName").value = d.style_name;
  $("#phrases").value = d.style_phrases.join("\n");
  $("#extract").value = d.extraction_contents.join("\n");
  $("#evalset").value = d.evaluation_contents.join("\n");
  renderPresets(d.presets);
  renderPairPreview();
}

function paintHeat(matrix, labels) {
  const cv = $("#heat");
  const ctx = cv.getContext("2d");
  const n = matrix.length;
  const W = cv.width;
  const pad = 70;
  const cell = (W - pad) / n;
  ctx.fillStyle = "#0e1014";
  ctx.fillRect(0, 0, W, W);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      const t = Math.max(0, Math.min(1, (v + 1) / 2));
      const r = Math.round(40 + 180 * t);
      const g = Math.round(50 + 90 * t);
      const b = Math.round(90 + 40 * (1 - t) * 1.4);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(pad + j * cell, i * cell, Math.max(1, cell - 1), Math.max(1, cell - 1));
    }
  }
  ctx.fillStyle = "#9d9486";
  ctx.font = "10px ui-monospace, monospace";
  labels.forEach((lb, i) => {
    ctx.save();
    ctx.translate(pad + i * cell + cell * 0.2, W - 8);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText((lb || `#${i}`).slice(0, 16), 0, 0);
    ctx.restore();
    ctx.fillText(String(i), 8, i * cell + cell * 0.6);
  });
}

function paintPca(ratios) {
  const host = $("#pcaBars");
  host.innerHTML = "";
  const max = Math.max(...ratios, 0.01);
  ratios.slice(0, 8).forEach((r, i) => {
    const d = document.createElement("div");
    d.className = "bar";
    d.style.height = `${Math.max(6, (r / max) * 100)}px`;
    d.innerHTML = `<span>${(r * 100).toFixed(0)}%</span>`;
    d.title = `PC${i + 1} ${(r * 100).toFixed(1)}%`;
    host.appendChild(d);
  });
}

function paintScatter(xy, proceed) {
  const cv = $("#scatter");
  const ctx = cv.getContext("2d");
  const W = cv.width;
  const H = cv.height;
  ctx.fillStyle = "#0e1014";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#2c2933";
  ctx.beginPath();
  ctx.moveTo(24, H / 2);
  ctx.lineTo(W - 8, H / 2);
  ctx.moveTo(W / 2, 8);
  ctx.lineTo(W / 2, H - 8);
  ctx.stroke();
  ctx.fillStyle = "#9d9486";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("PC1", W - 36, H / 2 - 8);
  ctx.fillText("PC2", W / 2 + 8, 18);
  if (!xy || !xy.length) return;
  const xs = xy.map((p) => p[0]);
  const ys = xy.map((p) => p[1] || 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const sx = (v) => 24 + ((v - minX) / (maxX - minX + 1e-8)) * (W - 40);
  const sy = (v) => H - 16 - ((v - minY) / (maxY - minY + 1e-8)) * (H - 32);
  xy.forEach((p, i) => {
    ctx.beginPath();
    ctx.fillStyle = proceed ? "#d4a24a" : "#5b8ea6";
    ctx.arc(sx(p[0]), sy(p[1] || 0), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#9d9486";
    ctx.fillText(String(i), sx(p[0]) + 6, sy(p[1] || 0) - 4);
  });
}

function setGate(id, pass, label) {
  const el = $(id);
  el.classList.toggle("pass", !!pass);
  el.classList.toggle("fail", pass === false);
  el.querySelector("span").textContent = label;
}

function renderPairs(rows) {
  const host = $("#pairTable");
  const body = (rows || [])
    .map(
      (r) =>
        `<div class="tr"><span title="${r.styled}">${r.label}</span><span>${r.delta_norm.toFixed(3)}</span></div>`
    )
    .join("");
  host.innerHTML = `<div class="tr head"><span>內容</span><span>‖Δ‖</span></div>` + body;
}

function renderNulls(nulls) {
  const host = $("#nulls");
  if (!host) return;
  const order = ["adjective", "non_renderer", "nonce"];
  const items = order.map((k) => nulls && nulls[k]).filter(Boolean);
  if (!items.length) {
    host.innerHTML = "";
    return;
  }
  host.innerHTML = items
    .map((n) => {
      const blocked = !n.proceed;
      const cls = blocked ? "pass" : "fail";
      return `<div class="control-card ${cls}">
        <b>${n.label}</b>
        <span>cos ${Number(n.mean_cosine).toFixed(3)} · λ₁ ${Number(n.lambda1).toFixed(3)} · ${
          blocked ? "低於風格方向（好）" : "太像風格，停"
        }</span>
      </div>`;
    })
    .join("");
}

function renderSoft(s) {
  const el = $("#softCard");
  const text = $("#softText");
  if (!el || !s) {
    if (text) text.textContent = "本次未跑 soft-prompt。";
    return;
  }
  el.className = "control-card " + (s.pass ? "pass" : "fail");
  const imp = Number(s.improvement || 0);
  text.textContent = s.pass
    ? `測試集 n=${s.n}：cos(h, h_s)=${s.cos_base.toFixed(3)} → cos(h+αμ, h_s)=${s.cos_soft.toFixed(3)}（+${imp.toFixed(3)}）。方向本身有用，可以吸收。`
    : `soft-prompt 沒把測試集推向 E(P_c+s)（${s.cos_base.toFixed(3)} → ${s.cos_soft.toFixed(3)}，Δ ${imp.toFixed(3)}）。不該寫 LoRA。`;
}

function renderControl(c) {
  /* kept for old jobs; new UI uses renderNulls */
  renderNulls(c && c.id ? { nonce: c } : null);
}

async function runAnalyze() {
  log("開始編碼與方向分析…");
  $("#runAnalyze").disabled = true;
  busy(true, "編碼配對並做 PCA…");
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    paintAnalyze(data);
    log(
      `分析完成 cosine=${data.stats.mean_cosine.toFixed(3)} λ1=${data.stats.lambda1.toFixed(3)} spec=${data.stats.specificity_pass} soft=${data.stats.soft_pass} proceed=${data.stats.proceed}`
    );
    if (!data.stats.proceed) log("門檻未全過：仍可吸收，但不建議當成成品。");
    await refreshJobs();
  } catch (e) {
    log("分析失敗：" + e.message);
    alert(e.message);
  } finally {
    $("#runAnalyze").disabled = false;
    busy(false);
  }
}

function paintAnalyze(data) {
  state.jobId = data.job_id;
  state.analyze = data;
  const s = data.stats;
  const labels = (s.pair_rows || []).map((r) => r.label);
  paintHeat(s.cosine_matrix, labels);
  paintPca(s.pca_ratios || []);
  paintScatter(s.scatter || [], s.proceed);
  setGate("#gateCos", s.cosine_pass, s.mean_cosine.toFixed(3));
  setGate("#gatePca", s.pca_pass, s.lambda1.toFixed(3));
  const spec = s.specificity || {};
  setGate("#gateSpec", s.specificity_pass, spec.margin != null ? "Δ " + Number(spec.margin).toFixed(3) : "—");
  const soft = s.soft_prompt || {};
  setGate("#gateSoft", s.soft_pass, soft.improvement != null ? (soft.improvement >= 0 ? "+" : "") + Number(soft.improvement).toFixed(3) : "—");
  setGate("#gateGo", s.proceed, s.proceed ? "通過" : "建議停止");
  $("#gateCos small").textContent = `門檻 ${Number(s.cosine_gate).toFixed(2)}`;
  $("#gatePca small").textContent = `門檻 ${Number(s.pca_gate).toFixed(2)}`;
  $("#analyzeNote").textContent =
    (s.notes || []).join(" · ") + ` · job ${data.job_id} · ${s.n_pairs} pairs · dim ${s.dim}`;
  renderPairs(s.pair_rows);
  renderNulls(s.nulls);
  renderSoft(s.soft_prompt);
  $("#toDistill").disabled = false;
}

function syncFormula() {
  const m = $("#method").value;
  $("#eqRank1").classList.toggle("on", m === "rank1" || m === "two_stage");
  $("#eqLstsq").classList.toggle("on", m === "svdl" || m === "lstsq" || m === "two_stage");
}

async function runDistill() {
  if (!state.jobId) {
    alert("請先完成分析");
    return;
  }
  log("TCSA 吸收中…");
  $("#runDistill").disabled = true;
  busy(true, "寫入低秩因子…");
  try {
    const body = {
      job_id: state.jobId,
      rank: parseInt($("#rank").value, 10),
      alpha: parseFloat($("#alpha").value),
      method: $("#method").value,
      export_format: $("#fmt").value,
      layers: $$("input.layer:checked").map((x) => x.value),
      mode: $("#mode").value,
      raw_path: $("#rawPath").value.trim(),
      export_kohya: $("#exportKohya").checked,
      include_negative: $("#includeNeg").checked,
      include_random: $("#includeRnd").checked,
      include_rank1: $("#includeRank1") ? $("#includeRank1").checked : true,
    };
    const res = await fetch("/api/distill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    paintDistill(data);
    log(`LoRA 寫入 ${data.lora_name}  (${data.weight_src})`);
    await refreshJobs();
  } catch (e) {
    log("蒸餾失敗：" + e.message);
    alert(e.message);
  } finally {
    $("#runDistill").disabled = false;
    busy(false);
  }
}

function paintDistill(data) {
  state.distill = data;
  const rows = Object.entries(data.layers).map(
    ([k, v]) =>
      `<div class="tr"><span>${k}</span><span>r=${v.rank}</span><span>${v.out_dim}×${v.in_dim}</span><span>${
        v.recon_error == null ? "—" : v.recon_error.toFixed(3)
      }</span><span>${
        v.rank1_recon_error == null ? "—" : v.rank1_recon_error.toFixed(3)
      }</span></div>`
  );
  $("#layerTable").innerHTML =
    `<div class="tr head"><span>層</span><span>rank</span><span>B×A</span><span>SVD/LS</span><span>Rank-1</span></div>` +
    rows.join("");
  const files = data.files || [{ name: data.lora_name, kind: "main", format: "diffusers" }];
  $("#fileList").innerHTML = files
    .map((f) => {
      const url = `/api/download/${data.job_id}/${f.name}`;
      return `<a href="${url}" download><span>${f.name}</span><span>${f.kind} · ${f.format}</span></a>`;
    })
    .join("");
  $("#toExport").disabled = false;
  renderExport(data);
}

function renderExport(data) {
  const files = data.files || [{ name: data.lora_name, kind: "main", format: "diffusers" }];
  const links = files
    .map((f) => `<a class="primary" href="/api/download/${data.job_id}/${f.name}" download style="text-decoration:none">下載 ${f.kind}/${f.format}</a>`)
    .join("");
  $("#exportBar").innerHTML = `<div><b>${data.lora_name}</b><div class="hint" style="margin:4px 0 0">來源 ${data.weight_src} · job ${data.job_id} · ${data.formula || ""}</div></div><div class="row tight">${links}</div>`;
  state.evalIndex = 0;
  const groups = data.validation || [];
  $("#evalNav").innerHTML = groups
    .map(
      (g, i) =>
        `<button class="ghost ${i === 0 ? "on" : ""}" data-eval="${i}">${g.content.slice(0, 28)}</button>`
    )
    .join("");
  paintEvalGroup();
  const main = files.find((f) => f.kind === "main") || files[0];
  $("#snippet").textContent = [
    `# Krea 2 Turbo · 同一 seed`,
    `lora: ${main ? main.name : data.lora_name}`,
    `strength sweep: 0.25 0.50 0.75 1.00 1.25`,
    `targets: txt_in.linear_1, txt_in.linear_2, text_fusion.projector`,
    `decisive test: A (no style word, LoRA off) → C (no style word, LoRA on)`,
  ].join("\n");
}

function paintEvalGroup() {
  const data = state.distill;
  if (!data) return;
  const groups = data.validation || [];
  const g = groups[state.evalIndex] || groups[0];
  if (!g) return;
  $$("#evalNav button").forEach((b, i) => b.classList.toggle("on", i === state.evalIndex));
  $("#valCards").innerHTML = g.conditions
    .map(
      (c) => `
    <div class="card">
      <div class="k">CONDITION ${c.id} · ${c.title.toUpperCase()}</div>
      <p>${c.prompt}</p>
      <div class="meta">LoRA ${c.lora ? "ON" : "off"}</div>
      <button class="ghost" data-copy="${c.prompt.replaceAll('"', """)}">複製 prompt</button>
    </div>`
    )
    .join("");
}

async function refreshJobs() {
  try {
    const data = await (await fetch("/api/jobs")).json();
    const jobs = data.jobs || [];
    if (!jobs.length) {
      $("#jobs").textContent = "尚無任務";
      return;
    }
    $("#jobs").innerHTML = jobs
      .map((j) => {
        const t = j.created ? new Date(j.created * 1000).toLocaleTimeString() : "";
        const mark = j.proceed ? "通過" : "擋下";
        return `<button class="job ${j.job_id === state.jobId ? "on" : ""}" data-job="${j.job_id}">
          <b>${j.style_name || j.job_id}</b>
          <small>${mark} · cos ${Number(j.mean_cosine || 0).toFixed(2)} · ${j.has_lora ? "已蒸餾" : "僅分析"} · ${t}</small>
        </button>`;
      })
      .join("");
  } catch (e) {
    log("無法讀取任務列表");
  }
}

async function openJob(id) {
  busy(true, "載入任務…");
  try {
    const data = await (await fetch(`/api/jobs/${id}`)).json();
    state.jobId = id;
    state.analyze = data;
    const sp = data.spec || {};
    if (sp.style_name) $("#styleName").value = sp.style_name;
    if (sp.style_phrases) $("#phrases").value = sp.style_phrases.join("\n");
    if (sp.extraction_contents) $("#extract").value = sp.extraction_contents.join("\n");
    if (sp.evaluation_contents) $("#evalset").value = sp.evaluation_contents.join("\n");
    if (sp.mode) $("#mode").value = sp.mode;
    paintAnalyze(data);
    if (data.distill) {
      paintDistill(data.distill);
    }
    goto(data.distill ? 3 : 1);
    log(`載入任務 ${id}`);
    await refreshJobs();
  } catch (e) {
    alert(e.message);
  } finally {
    busy(false);
  }
}

function copyCurrentPrompts() {
  const data = state.distill;
  if (!data) return;
  const g = (data.validation || [])[state.evalIndex];
  if (!g) return;
  const text = g.conditions.map((c) => `${c.id}. ${c.title}\n${c.prompt}\nLoRA ${c.lora ? "ON" : "off"}`).join("\n\n");
  navigator.clipboard.writeText(text);
  log("已複製 A–D prompts");
}

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-copy]");
  if (t) {
    navigator.clipboard.writeText(t.getAttribute("data-copy"));
    log("已複製 prompt");
  }
  const job = e.target.closest("[data-job]");
  if (job) openJob(job.getAttribute("data-job"));
  const ev = e.target.closest("[data-eval]");
  if (ev) {
    state.evalIndex = parseInt(ev.getAttribute("data-eval"), 10);
    paintEvalGroup();
  }
});

$("#mode").addEventListener("change", () => {
  $("#realPaths").classList.toggle("hidden", $("#mode").value !== "real");
  $("#modePill").textContent = $("#mode").value === "real" ? "Real" : "Demo";
});
$("#method").addEventListener("change", syncFormula);
$("#pairing").addEventListener("change", renderPairPreview);
$("#phrases").addEventListener("input", renderPairPreview);
$("#extract").addEventListener("input", renderPairPreview);
$$(".steps button").forEach((b) => b.addEventListener("click", () => goto(+b.dataset.step)));
$$("[data-back]").forEach((b) => b.addEventListener("click", () => goto(+b.dataset.back)));
$("#resetBtn").addEventListener("click", () => loadDefaults());
$("#toAnalyze").addEventListener("click", () => goto(1));
$("#toDistill").addEventListener("click", () => goto(2));
$("#toExport").addEventListener("click", () => goto(3));
$("#runAnalyze").addEventListener("click", runAnalyze);
$("#runDistill").addEventListener("click", runDistill);
$("#copyAll").addEventListener("click", copyCurrentPrompts);

syncFormula();
fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    $("#healthPill").textContent = "後端就緒 v" + (h.version || "");
    $("#healthPill").classList.add("ok");
  })
  .catch(() => {
    $("#healthPill").textContent = "後端未連線";
  });

loadDefaults();
refreshJobs();
