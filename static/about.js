const out = document.getElementById("metrics");
const pct = (v) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "-");

function notice(msg) {
  out.innerHTML = "";
  const p = document.createElement("p");
  p.className = "notice";
  p.textContent = msg;
  out.appendChild(p);
}

function stat(value, label) {
  const d = document.createElement("div");
  d.className = "stat";
  const b = document.createElement("b");
  b.textContent = value;
  const s = document.createElement("span");
  s.textContent = label;
  d.append(b, s);
  return d;
}

function render(m) {
  out.innerHTML = "";

  const stats = document.createElement("div");
  stats.className = "stats";
  stats.append(stat(pct(m.accuracy), "Accuracy"));
  if (typeof m.macro_f1 === "number") stats.append(stat(pct(m.macro_f1), "Macro F1"));
  out.appendChild(stats);

  if (m.dataset || m.test_size) {
    const p = document.createElement("p");
    p.className = "small";
    const size = m.test_size ? m.test_size.toLocaleString() + " test sentences" : "";
    p.textContent = [m.dataset, size].filter(Boolean).join(", ");
    out.appendChild(p);
  }

  const rows = Object.entries(m.per_class || {});
  if (!rows.length) return;
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.className = "metrics";
  table.innerHTML = "<thead><tr><th>Emotion</th><th>Precision</th><th>Recall</th><th>F1</th><th>Sentences</th></tr></thead>";
  const body = document.createElement("tbody");
  rows.forEach(([name, c]) => {
    const tr = document.createElement("tr");
    const first = document.createElement("td");
    const dot = document.createElement("i");
    dot.style.background = "var(--" + name + ")";
    first.append(dot, name);
    tr.appendChild(first);
    [pct(c.precision), pct(c.recall), pct(c.f1), c.support == null ? "-" : String(c.support)].forEach((v) => {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  out.appendChild(wrap);
}

fetch("/api/metrics")
  .then((r) => r.json())
  .then((m) => {
    if (!m.available || typeof m.accuracy !== "number") {
      notice("Evaluation results have not been added yet.");
    } else {
      render(m);
    }
  })
  .catch(() => notice("Could not load the results."));
