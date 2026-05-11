/* HeartGuard — front-end logic
   Multi-step questionnaire, animated risk gauge, AJAX chat */

/* ============================================================
   QUESTIONNAIRE — 30 optional questions across 10 steps
   • At least 10 must be answered
   • Every step has a "Skip step" button that clears it and advances
   • Live "X / 30 answered" counter
   • Soft per-field validation: range only, no required fields
   ============================================================ */
const MIN_ANSWERED = 10;

function initQuestionnaire() {
  const form = document.getElementById("assessForm");
  if (!form) return;

  const steps      = Array.from(form.querySelectorAll(".step-card"));
  const questions  = Array.from(form.querySelectorAll(".question"));
  const backBtn    = document.getElementById("backBtn");
  const nextBtn    = document.getElementById("nextBtn");
  const skipBtn    = document.getElementById("skipBtn");
  const submitBtn  = document.getElementById("submitBtn");
  const fill       = document.getElementById("progressFill");
  const stepNum    = document.getElementById("stepNum");
  const stepTotal  = document.getElementById("stepTotal");
  const answeredNowEl   = document.getElementById("answeredNow");
  const answeredTotalEl = document.getElementById("answeredTotal");
  const minHint    = document.getElementById("minHint");

  stepTotal.textContent      = steps.length;
  answeredTotalEl.textContent = questions.length;

  let current = 0;

  function show(i) {
    steps.forEach((s, idx) => s.classList.toggle("active", idx === i));
    current = i;
    backBtn.disabled = i === 0;
    const last = i === steps.length - 1;
    nextBtn.classList.toggle("hidden", last);
    submitBtn.classList.toggle("hidden", !last);
    fill.style.width = (((i + 1) / steps.length) * 100) + "%";
    stepNum.textContent = i + 1;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* -------- Is a single .question block answered? -------- */
  function questionAnswered(q) {
    // Multi-checkbox group
    const checks = q.querySelectorAll('input[type="checkbox"]');
    if (checks.length) {
      return Array.from(checks).some(c => c.checked);
    }
    // Radio chip group
    const radios = q.querySelectorAll('input[type="radio"]');
    if (radios.length) {
      return Array.from(radios).some(r => r.checked);
    }
    // Numeric / text inputs
    const inputs = q.querySelectorAll('input[type="number"], input[type="text"]');
    for (const inp of inputs) {
      if (inp.value && inp.value.trim() !== "") {
        // basic range check — if out of range, count as unanswered
        if (inp.type === "number") {
          const v = +inp.value;
          if (isNaN(v)) return false;
          if (inp.min !== "" && v < +inp.min) return false;
          if (inp.max !== "" && v > +inp.max) return false;
        }
        return true;
      }
    }
    // Selects
    const selects = q.querySelectorAll("select");
    for (const sel of selects) {
      if (sel.value && sel.value !== "") return true;
    }
    return false;
  }

  function countAnswered() {
    let n = 0;
    questions.forEach(q => { if (questionAnswered(q)) n++; });
    return n;
  }

  function refreshCounter() {
    const n = countAnswered();
    answeredNowEl.textContent = n;
    if (n >= MIN_ANSWERED) {
      answeredNowEl.classList.add("ok");
      minHint.classList.add("ok");
      minHint.innerHTML = `You've answered <b>${n}</b> question${n === 1 ? "" : "s"} — you're ready to submit.`;
    } else {
      answeredNowEl.classList.remove("ok");
      minHint.classList.remove("ok");
      minHint.innerHTML = `Answer at least <b>${MIN_ANSWERED}</b> of the ${questions.length} questions to see your risk. <b>${n}</b> answered so far.`;
    }
  }

  /* -------- Soft per-step validation -------- */
  function softValidateStep(i) {
    let valid = true;
    steps[i].querySelectorAll("input, select").forEach(f => {
      f.style.borderColor = "";
      if (f.type === "number" && f.value !== "") {
        const v = +f.value;
        if (isNaN(v) || (f.min !== "" && v < +f.min) || (f.max !== "" && v > +f.max)) {
          f.style.borderColor = "#ef4444";
          valid = false;
        }
      }
    });
    // BP sanity check on step 3 — only enforce if BOTH provided
    if (i === 2) {
      const hiEl = form.querySelector("[name=ap_hi]");
      const loEl = form.querySelector("[name=ap_lo]");
      const hi = +hiEl.value, lo = +loEl.value;
      if (hiEl.value && loEl.value && hi <= lo) {
        hiEl.style.borderColor = "#ef4444";
        loEl.style.borderColor = "#ef4444";
        valid = false;
      }
    }
    return valid;
  }

  /* -------- Live BMI hint on step 2 -------- */
  const hint = document.getElementById("bmiHint");
  const h = form.querySelector("[name=height_cm]");
  const w = form.querySelector("[name=weight_kg]");
  function updateBmi() {
    const hv = +h.value, wv = +w.value;
    if (hv > 0 && wv > 0) {
      const bmi = wv / Math.pow(hv / 100, 2);
      let label = "Normal";
      if (bmi < 18.5)      label = "Underweight";
      else if (bmi < 25)   label = "Normal";
      else if (bmi < 30)   label = "Overweight";
      else if (bmi < 35)   label = "Obese I";
      else                 label = "Obese II+";
      hint.textContent = `BMI = ${bmi.toFixed(1)} (${label})`;
    } else {
      hint.textContent = "BMI will be computed automatically when both fields are filled.";
    }
  }
  if (h && w) { h.addEventListener("input", updateBmi); w.addEventListener("input", updateBmi); }

  /* -------- Recount whenever anything changes -------- */
  form.addEventListener("input",  refreshCounter);
  form.addEventListener("change", refreshCounter);

  /* -------- Buttons -------- */
  nextBtn.addEventListener("click", () => {
    if (!softValidateStep(current)) return;
    if (current < steps.length - 1) show(current + 1);
  });

  backBtn.addEventListener("click", () => {
    if (current > 0) show(current - 1);
  });

  skipBtn.addEventListener("click", () => {
    // Clear every input / select / radio / checkbox in the current step
    steps[current].querySelectorAll("input, select").forEach(f => {
      if (f.type === "checkbox" || f.type === "radio") f.checked = false;
      else f.value = "";
      f.style.borderColor = "";
    });
    if (hint) hint.textContent = "BMI will be computed automatically when both fields are filled.";
    refreshCounter();
    if (current < steps.length - 1) show(current + 1);
  });

  form.addEventListener("submit", e => {
    if (!softValidateStep(current)) { e.preventDefault(); return; }
    const n = countAnswered();
    if (n < MIN_ANSWERED) {
      e.preventDefault();
      minHint.classList.add("err");
      minHint.innerHTML =
        `Please answer at least <b>${MIN_ANSWERED}</b> questions before submitting. ` +
        `You've answered only <b>${n}</b>.`;
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });

  show(0);
  refreshCounter();
}


/* ============================================================
   RESULT PAGE — animate gauge, wire up chat
   ============================================================ */
function initResultPage() {
  const gauge = document.getElementById("gauge");
  if (gauge) {
    const prob = parseFloat(gauge.dataset.prob);
    const valueEl = document.getElementById("gaugeValue");
    const fillCircle = gauge.querySelector(".gauge-fill");
    const C = 2 * Math.PI * 85;                    // 533.99
    const targetOffset = C * (1 - prob);

    // Start at empty then animate fill + numeric counter together
    requestAnimationFrame(() => {
      fillCircle.style.strokeDashoffset = targetOffset;
    });
    const target = prob * 100;
    let displayed = 0;
    const start = performance.now();
    const dur = 1600;
    function tick(now) {
      const t = Math.min(1, (now - start) / dur);
      const eased = t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      displayed = eased * target;
      valueEl.textContent = displayed.toFixed(1) + "%";
      if (t < 1) requestAnimationFrame(tick);
      else valueEl.textContent = target.toFixed(1) + "%";
    }
    requestAnimationFrame(tick);
  }

  // chat
  const chatCard = document.getElementById("chatCard");
  if (!chatCard) return;
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const stream = document.getElementById("chatStream");
  const tier = chatCard.dataset.tier;
  const summary = chatCard.dataset.summary;
  const ollamaUp = chatCard.dataset.ollama === "true";

  if (!ollamaUp) return;

  function appendMsg(cls, html) {
    const div = document.createElement("div");
    div.className = "chat-msg " + cls;
    div.innerHTML = html;
    stream.prepend(div);
    return div;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    appendMsg("q", `<div class="who">You</div>${escapeHtml(q)}`);
    input.value = "";

    const thinking = appendMsg("a",
      `<div class="who">AI</div>
       <span class="typing"><span></span><span></span><span></span></span> Retrieving and generating…`);

    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, tier, summary }),
      });
      const j = await r.json();
      thinking.remove();
      if (!j.ok) {
        appendMsg("err", `<div class="who">AI</div>${escapeHtml(j.answer || j.error || "Unknown error.")}`);
        return;
      }
      let retrievedHtml = "";
      if (j.retrieved && j.retrieved.length) {
        retrievedHtml = `<div class="ret"><b>Retrieved ${j.retrieved.length} knowledge entries:</b><br>` +
          j.retrieved.map(x => `${x.emoji} ${escapeHtml(x.title)} <small>(${x.kind}, score ${x.score.toFixed(3)})</small>`).join("<br>") +
          `</div>`;
      }
      appendMsg("a", `<div class="who">AI</div>${escapeHtml(j.answer)}${retrievedHtml}`);
    } catch (err) {
      thinking.remove();
      appendMsg("err", `<div class="who">AI</div>Network error: ${escapeHtml(err.message)}`);
    }
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
