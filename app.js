const textEl = document.getElementById("text");
const goEl = document.getElementById("go");
const resultEl = document.getElementById("result");
const emptyEl = document.getElementById("empty");
const errorEl = document.getElementById("error");
const barsEl = document.getElementById("bars");
const countEl = document.getElementById("count");
const emojiEl = document.getElementById("emoji");

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

function render(data) {
  errorEl.hidden = true;
  emptyEl.hidden = true;
  resultEl.style.setProperty("--accent", "var(--" + data.label + ")");

  emojiEl.textContent = data.emoji;
  emojiEl.classList.remove("pop");
  void emojiEl.offsetWidth;
  emojiEl.classList.add("pop");

  document.getElementById("label").textContent = data.label;
  document.getElementById("conf").textContent =
    "Leaning " + (data.confidence * 100).toFixed(1) + "% toward this emotion";

  barsEl.innerHTML = "";
  const fills = [];
  data.probabilities.forEach((p) => {
    const li = document.createElement("li");
    li.className = "bar";

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = p.emoji + " " + p.label;

    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.background = "var(--" + p.label + ")";
    track.appendChild(fill);

    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = (p.probability * 100).toFixed(1) + "%";

    li.append(name, track, pct);
    barsEl.appendChild(li);
    fills.push([fill, p.probability]);
  });

  resultEl.hidden = false;
  requestAnimationFrame(() => {
    fills.forEach(([fill, prob]) => { fill.style.width = (prob * 100) + "%"; });
  });
}

async function analyze() {
  const text = textEl.value.trim();
  if (!text) {
    showError("Write a sentence first.");
    return;
  }

  goEl.disabled = true;
  goEl.textContent = "Analyzing...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    let data;
    try { data = await res.json(); }
    catch (_) { data = { error: "Server error (" + res.status + "). Check the terminal for details." }; }
    if (!res.ok) {
      showError(data.error || "Something went wrong. Try again.");
    } else {
      render(data);
    }
  } catch (e) {
    showError("Can't reach the server. Check that main.py is running.");
  } finally {
    goEl.disabled = false;
    goEl.textContent = "Analyze emotion";
  }
}

function updateCount() {
  countEl.textContent = textEl.value.length + " / 500";
}

goEl.addEventListener("click", analyze);
textEl.addEventListener("input", updateCount);
textEl.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textEl.value = chip.dataset.text;
    updateCount();
    analyze();
  });
});
