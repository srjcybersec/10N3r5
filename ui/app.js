const BASE = "";

function setActiveNav() {
  const links = document.querySelectorAll("[data-nav]");
  const current = window.location.pathname.split("/").pop() || "index.html";
  links.forEach((link) => {
    const href = link.getAttribute("href") || "";
    const hrefName = href.split("/").pop();
    if (hrefName === current) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });
}

function setStatus(ok) {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  if (!dot || !text) return;
  dot.style.background = ok ? "#22c55e" : "#ef4444";
  text.textContent = ok ? "Environment online" : "Environment offline";
}

async function checkHealth() {
  try {
    const r = await fetch(`${BASE}/health`);
    setStatus(r.ok);
    return r.ok;
  } catch (_) {
    setStatus(false);
    return false;
  }
}

function createAmbient() {
  if (document.querySelector(".ambient")) return;
  const wrap = document.createElement("div");
  wrap.className = "ambient";
  wrap.innerHTML = '<div class="blob one"></div><div class="blob two"></div>';
  document.body.appendChild(wrap);
}

function byId(id) {
  return document.getElementById(id);
}

function nowStamp() {
  return new Date().toISOString().split("T")[1].slice(0, 8);
}

function animateNumber(el, to, duration = 700, decimals = 0) {
  if (!el) return;
  const target = Number(to);
  if (!Number.isFinite(target)) {
    el.textContent = "--";
    return;
  }
  const from = Number(el.dataset.value || 0);
  const start = performance.now();
  function step(ts) {
    const p = Math.min(1, (ts - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    const value = from + (target - from) * eased;
    el.textContent = value.toFixed(decimals);
    if (p < 1) {
      requestAnimationFrame(step);
    } else {
      el.dataset.value = String(target);
    }
  }
  requestAnimationFrame(step);
}

createAmbient();
setActiveNav();
checkHealth();
