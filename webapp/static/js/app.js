/* HeartGuard — front-end logic
   Multi-step questionnaire, animated risk gauge, AJAX chat */

/* ============================================================
   QUESTIONNAIRE — step-by-step with validation + progress bar
   ============================================================ */
function initQuestionnaire() {
  const form = document.getElementById("assessForm");
  if (!form) return;

  const steps = Array.from(form.querySelectorAll(".step-card"));
  const backBtn = document.getElementById("backBtn");
  const nextBtn = document.getElementById("nextBtn");
  const submitBtn = document.getElementById("submitBtn");
  const fill = document.getElementById("progressFill");
  const stepNum = document.getElementById("stepNum");
  const stepTotal = document.getElementById("stepTotal");
  stepTotal.textContent = steps.length;

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

  function validateStep(i) {
    const fields = steps[i].querySelectorAll("input[required], select[required]");
    let valid = true;
    fields.forEach(f => {
      f.style.borderColor = "";
      if (!f.value || (f.type === "number" && (isNaN(+f.value) || +f.value < +f.min || +f.value > +f.max))) {
        f.style.borderColor = "#ef4444";
        valid = false;
      }
    });
    // BP sanity check on step 3
    if (i === 2) {
      const hi = +form.querySelector("[name=ap_hi]").value;
      const lo = +form.querySelector("[name=ap_lo]").value;
      if (hi && lo && hi <= lo) {
        form.querySelector("[name=ap_hi]").style.borderColor = "#ef4444";
        form.querySelector("[name=ap_lo]").style.borderColor = "#ef4444";
        valid = false;
      }
    }
    return valid;
  }

  // live BMI hint on step 2
  const hint = document.getElementById("bmiHint");
  const h = form.querySelector("[name=height_cm]");
  const w = form.querySelector("[name=weight_kg]");
  function updateBmi() {
    const hv = +h.value, wv = +w.value;
    if (hv > 0 && wv > 0) {
      const bmi = wv / Math.pow(hv / 100, 2);
      let label = "Normal";
      if (bmi < 18.5) label = "Underweight";
      else if (bmi < 25) label = "Normal";
      else if (bmi < 30) label = "Overweight";
      else if (bmi < 35) label = "Obese I";
      else label = "Obese II+";
      hint.textContent = `BMI = ${bmi.toFixed(1)} (${label})`;
    } else {
      hint.textContent = "BMI will be computed automatically.";
    }
  }
  if (h && w) { h.addEventListener("input", updateBmi); w.addEventListener("input", updateBmi); }

  nextBtn.addEventListener("click", () => {
    if (!validateStep(current)) return;
    if (current < steps.length - 1) show(current + 1);
  });
  backBtn.addEventListener("click", () => { if (current > 0) show(current - 1); });

  form.addEventListener("submit", e => {
    if (!validateStep(current)) { e.preventDefault(); }
  });

  show(0);
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
