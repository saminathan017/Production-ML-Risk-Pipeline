/**
 * NeuralNexus — Advanced ML Risk Intelligence Platform
 * app.js  •  Particles · Neural Canvas · Chart.js · WebSocket · Animations
 */

'use strict';

const API = '/api';
const POLL_MS = 28_000;   // refresh interval for stats

// ─── Global state ─────────────────────────────────────────────────────────────
const state = {
  metrics: null,
  modelInfo: null,
  monStats: null,
  featureImportance: [],
  predictionHistory: [],
  charts: {},
  ws: null,
  wsConnected: false,
};

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initParticles();
  initNeuralCanvas();
  setupFeatureCounter();
  loadAll().then(() => {
    renderCharts();
    startPolling();
    connectWebSocket();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
//  CLOCK
// ═══════════════════════════════════════════════════════════════════════════════
function initClock() {
  const el = document.getElementById('topbar-clock');
  if (!el) return;
  const tick = () => {
    const d = new Date();
    el.textContent = d.toLocaleTimeString('en-US', { hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PARTICLE CANVAS
// ═══════════════════════════════════════════════════════════════════════════════
function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, particles;

  const PARTICLE_COUNT = 90;
  const MAX_DIST = 130;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r:  Math.random() * 1.6 + 0.4,
      alpha: Math.random() * 0.5 + 0.1,
    };
  }

  function boot() {
    resize();
    particles = Array.from({ length: PARTICLE_COUNT }, mkParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;

      // Draw dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,245,255,${p.alpha})`;
      ctx.fill();

      // Draw connections
      for (let j = i + 1; j < particles.length; j++) {
        const q = particles[j];
        const dx = p.x - q.x, dy = p.y - q.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DIST) {
          const opacity = (1 - dist / MAX_DIST) * 0.12;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(q.x, q.y);
          ctx.strokeStyle = `rgba(0,245,255,${opacity})`;
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  boot();
  draw();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  NEURAL NETWORK CANVAS
// ═══════════════════════════════════════════════════════════════════════════════
function initNeuralCanvas() {
  const canvas = document.getElementById('neuralCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const LAYERS = [6, 8, 8, 6, 2];   // representative topology
  let W, H, nodes, t = 0;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    W = canvas.width  = rect.width;
    H = canvas.height = 180;
    buildNodes();
  }

  function buildNodes() {
    nodes = [];
    const cols = LAYERS.length;
    const colW = W / (cols + 1);
    for (let l = 0; l < cols; l++) {
      const count = LAYERS[l];
      const rowH = H / (count + 1);
      for (let n = 0; n < count; n++) {
        nodes.push({ l, n, x: colW * (l + 1), y: rowH * (n + 1), pulse: Math.random() * Math.PI * 2 });
      }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    t += 0.016;

    const byLayer = {};
    for (const nd of nodes) {
      if (!byLayer[nd.l]) byLayer[nd.l] = [];
      byLayer[nd.l].push(nd);
    }

    // Draw edges
    for (let l = 0; l < LAYERS.length - 1; l++) {
      const from = byLayer[l], to = byLayer[l + 1];
      for (const f of from) {
        for (const tN of to) {
          const pulse = Math.sin(t * 1.4 + f.pulse) * 0.5 + 0.5;
          ctx.beginPath();
          ctx.moveTo(f.x, f.y);
          ctx.lineTo(tN.x, tN.y);
          ctx.strokeStyle = `rgba(123,47,255,${0.04 + pulse * 0.06})`;
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
      }
    }

    // Animated data packets on edges (random layer each frame)
    const activeLayer = Math.floor(t * 0.5) % (LAYERS.length - 1);
    const srcLayer = byLayer[activeLayer];
    const dstLayer = byLayer[activeLayer + 1];
    if (srcLayer && dstLayer) {
      const src = srcLayer[Math.floor(t * 3) % srcLayer.length];
      const dst = dstLayer[Math.floor(t * 2) % dstLayer.length];
      if (src && dst) {
        const pct = (Math.sin(t * 2) * 0.5 + 0.5);
        const px = src.x + (dst.x - src.x) * pct;
        const py = src.y + (dst.y - src.y) * pct;
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,245,255,0.9)';
        ctx.shadowBlur = 8;
        ctx.shadowColor = '#00f5ff';
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Draw nodes
    for (const nd of nodes) {
      const glow = Math.sin(t + nd.pulse) * 0.5 + 0.5;
      const r = 4 + glow * 1.5;
      const alpha = 0.5 + glow * 0.4;
      const isInput  = nd.l === 0;
      const isOutput = nd.l === LAYERS.length - 1;
      const color = isOutput ? '#00ff88' : isInput ? '#00f5ff' : '#7b2fff';
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${isOutput?'0,255,136':isInput?'0,245,255':'123,47,255'},${alpha * 0.3})`;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  draw();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  FEATURE COUNTER
// ═══════════════════════════════════════════════════════════════════════════════
function setupFeatureCounter() {
  const ta = document.getElementById('featuresInput');
  const counter = document.getElementById('featureCounter');
  if (!ta || !counter) return;

  ta.addEventListener('input', () => {
    const vals = ta.value.split(',').map(s => s.trim()).filter(Boolean);
    counter.textContent = `${vals.length} / 30`;
    counter.className = 'feature-counter';
    if (vals.length === 30) counter.classList.add('valid');
    else if (vals.length > 30) counter.classList.add('invalid');
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PRESET FILLS
// ═══════════════════════════════════════════════════════════════════════════════
const PRESETS = {
  malignant: [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189],
  benign:    [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259],
};

window.fillPreset = function(type) {
  const ta = document.getElementById('featuresInput');
  if (!ta) return;
  if (type === 'random') {
    const vals = Array.from({ length: 30 }, () => +(Math.random() * 2).toFixed(4));
    ta.value = vals.join(', ');
  } else {
    ta.value = PRESETS[type]?.join(', ') ?? '';
  }
  ta.dispatchEvent(new Event('input'));
};

window.clearInput = function() {
  const ta = document.getElementById('featuresInput');
  if (ta) { ta.value = ''; ta.dispatchEvent(new Event('input')); }
  const res = document.getElementById('predictionResult');
  if (res) res.innerHTML = '';
};

// ═══════════════════════════════════════════════════════════════════════════════
//  PREDICTION
// ═══════════════════════════════════════════════════════════════════════════════
window.makePrediction = async function() {
  const ta = document.getElementById('featuresInput');
  const resultEl = document.getElementById('predictionResult');
  const btn = document.getElementById('predictBtn');

  const raw = ta?.value.trim();
  if (!raw) { showResultError(resultEl, 'No feature vector provided.'); return; }

  const features = raw.split(',').map(s => parseFloat(s.trim()));
  if (features.length !== 30 || features.some(isNaN)) {
    showResultError(resultEl, `Expected 30 numeric values — got ${features.length}.`);
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spinSlow 1s linear infinite">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
    </svg>
    PROCESSING...`;

  try {
    const resp = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || 'Inference failed');
    const data = await resp.json();
    renderPredictionResult(resultEl, data);
    addFeedEntry(data);
    incrementHeaderTotal();
  } catch (err) {
    showResultError(resultEl, err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
      </svg>
      EXECUTE INFERENCE`;
  }
};

function renderPredictionResult(container, data) {
  // In breast-cancer dataset: prediction=1 → benign, 0 → malignant
  const isBenign  = data.prediction === 1;
  const cls       = isBenign ? 'Benign'    : 'Malignant';
  const riskCls   = isBenign ? 'benign'    : data.risk_level === 'Medium' ? 'medium' : 'malign';
  const riskLabel = isBenign ? 'LOW RISK'  : data.risk_level === 'Medium' ? 'MEDIUM RISK' : 'HIGH RISK';
  const fillCls   = isBenign ? 'benign'    : data.risk_level === 'Medium' ? 'medium' : 'malign';
  const pct       = (data.probability * 100).toFixed(1);

  container.innerHTML = `
    <div class="prediction-result">
      <div class="result-header ${riskCls}">
        <span>◈ CLASSIFICATION: ${cls.toUpperCase()}</span>
        <span>${riskLabel}</span>
      </div>
      <div class="result-grid">
        <div class="result-cell">
          <div class="result-cell-label">Prediction</div>
          <div class="result-cell-value">${cls}</div>
        </div>
        <div class="result-cell">
          <div class="result-cell-label">Confidence</div>
          <div class="result-cell-value">${pct}%</div>
        </div>
        <div class="result-cell">
          <div class="result-cell-label">Risk Level</div>
          <div class="result-cell-value">${data.risk_level}</div>
        </div>
        <div class="result-cell">
          <div class="result-cell-label">Latency</div>
          <div class="result-cell-value">${data.latency_ms.toFixed(2)} ms</div>
        </div>
      </div>
      <div class="confidence-bar-wrap">
        <div class="confidence-label-row">
          <span>CONFIDENCE SCORE</span>
          <span>${pct}%</span>
        </div>
        <div class="confidence-track">
          <div class="confidence-fill ${fillCls}" id="confFill" style="width:0%"></div>
        </div>
      </div>
    </div>`;

  // Animate bar after render
  setTimeout(() => {
    const fill = document.getElementById('confFill');
    if (fill) fill.style.width = pct + '%';
  }, 50);
}

function showResultError(container, msg) {
  if (!container) return;
  container.innerHTML = `<div class="state-error" style="margin-top:14px">⚠ ${msg}</div>`;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DATA LOADING
// ═══════════════════════════════════════════════════════════════════════════════
async function loadAll() {
  await Promise.allSettled([
    loadModelInfo(),
    loadMetrics(),
    loadMonitoringStats(),
    loadFeatureImportance(),
    loadRecentPredictions(),
    loadSystemStatus(),
  ]);
}

async function apiFetch(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function loadModelInfo() {
  try {
    const d = await apiFetch('/model');
    state.modelInfo = d;
    renderModelInfo(d);

    // Header
    const hm = document.getElementById('hdr-model');
    if (hm) hm.textContent = fmtModelType(d.model_type);
  } catch (e) {
    const el = document.getElementById('modelInfoContent');
    if (el) el.innerHTML = `<div class="state-error">Failed to load model: ${e.message}</div>`;
  }
}

async function loadMetrics() {
  try {
    const d = await apiFetch('/metrics');
    state.metrics = d;
    const m = d.metrics;

    animateCounter('stat-accuracy', m.accuracy * 100, '%', 1);
    animateCounter('stat-auc',      m.roc_auc  * 100, '%', 1);
    animateCounter('stat-f1',       m.f1       * 100, '%', 1);

    const hacc = document.getElementById('hdr-accuracy');
    if (hacc) hacc.textContent = (m.accuracy * 100).toFixed(1) + '%';
    const hauc = document.getElementById('hdr-auc');
    if (hauc) hauc.textContent = (m.roc_auc * 100).toFixed(1) + '%';

    const nn = document.getElementById('nn-accuracy');
    if (nn) nn.textContent = (m.f1 * 100).toFixed(1) + '%';
  } catch (e) { console.warn('Metrics load error:', e.message); }
}

async function loadMonitoringStats() {
  try {
    const d = await apiFetch('/monitoring/stats');
    state.monStats = d;

    if (d.total_predictions > 0) {
      animateCounter('stat-total', d.total_predictions, '', 0);
      animateCounter('stat-latency', d.latency_stats.mean_ms, ' ms', 1);

      const htot = document.getElementById('hdr-total');
      if (htot) htot.textContent = d.total_predictions;
    } else {
      setText('stat-total', '0');
      setText('stat-latency', '— ms');
    }
  } catch (e) { console.warn('Monitor stats error:', e.message); }
}

async function loadFeatureImportance() {
  try {
    const d = await apiFetch('/feature-importance');
    state.featureImportance = d.feature_importance || [];
    renderFeatureImportance(d.feature_importance || []);
  } catch (e) {
    const el = document.getElementById('featureImportanceContent');
    if (el) el.innerHTML = `<div class="state-error">Feature importance unavailable: ${e.message}</div>`;
  }
}

async function loadRecentPredictions() {
  try {
    const d = await apiFetch('/predictions/recent');
    state.predictionHistory = d.predictions || [];
    renderFeed(state.predictionHistory);
  } catch (e) { console.warn('Recent predictions error:', e.message); }
}

async function loadSystemStatus() {
  try {
    const d = await apiFetch('/system/status');
    const pf = document.getElementById('footer-platform');
    if (pf) pf.textContent = d.platform;
    const up = document.getElementById('footer-uptime');
    if (up) up.textContent = `UPTIME: ${fmtUptime(d.uptime_seconds)}`;
    const fv = document.getElementById('footer-version');
    if (fv) fv.textContent = '2.0.0';
  } catch (e) { console.warn('System status error:', e.message); }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RENDERERS
// ═══════════════════════════════════════════════════════════════════════════════
function renderModelInfo(d) {
  const el = document.getElementById('modelInfoContent');
  if (!el) return;

  const metrics = d.metrics || {};
  const metricRows = ['accuracy','precision','recall','f1','roc_auc']
    .filter(k => metrics[k] !== undefined)
    .map(k => `
      <div class="gauge-item">
        <span class="gauge-name">${k.replace('_','-').toUpperCase()}</span>
        <div class="gauge-track">
          <div class="gauge-fill" data-target="${metrics[k]*100}" style="width:0%"></div>
        </div>
        <span class="gauge-pct">${(metrics[k]*100).toFixed(1)}%</span>
      </div>`).join('');

  el.innerHTML = `
    <div class="intel-grid">
      <div class="intel-box">
        <div class="intel-box-label">Algorithm</div>
        <div class="intel-box-value">${fmtModelType(d.model_type)}</div>
      </div>
      <div class="intel-box">
        <div class="intel-box-label">Version</div>
        <div class="intel-box-value" style="font-size:0.7rem">${d.model_version?.split('_')[1] || '—'}</div>
      </div>
      <div class="intel-box">
        <div class="intel-box-label">Features</div>
        <div class="intel-box-value">${d.feature_count} dims</div>
      </div>
      <div class="intel-box">
        <div class="intel-box-label">Dataset Hash</div>
        <div class="intel-box-value" style="font-size:0.65rem">${(d.dataset_hash||'').substring(0,10)}…</div>
      </div>
      <div class="intel-box">
        <div class="intel-box-label">Trained</div>
        <div class="intel-box-value" style="font-size:0.72rem">${fmtDate(d.trained_at)}</div>
      </div>
      <div class="intel-box">
        <div class="intel-box-label">Status</div>
        <div class="intel-box-value" style="color:var(--green);font-size:0.8rem">● ACTIVE</div>
      </div>
    </div>
    <div class="gauge-row">${metricRows}</div>`;

  // Animate gauges
  requestAnimationFrame(() => {
    document.querySelectorAll('.gauge-fill[data-target]').forEach(bar => {
      setTimeout(() => { bar.style.width = bar.dataset.target + '%'; }, 100);
    });
  });
}

function renderFeatureImportance(list) {
  const el = document.getElementById('featureImportanceContent');
  if (!el) return;
  if (!list.length) { el.innerHTML = '<div class="state-loading">NO IMPORTANCE DATA AVAILABLE</div>'; return; }

  const top = list.slice(0, 10);
  const maxVal = top[0].importance;

  el.innerHTML = `<div class="feature-list">${
    top.map(item => {
      const pct = ((item.importance / maxVal) * 100).toFixed(1);
      const raw = (item.importance * 100).toFixed(2);
      return `
        <div class="feature-item">
          <span class="feature-name">${item.feature}</span>
          <div class="feature-bar-track">
            <div class="feature-bar-fill" data-w="${pct}" style="width:0%"></div>
          </div>
          <span class="feature-pct">${raw}%</span>
        </div>`;
    }).join('')
  }</div>`;

  requestAnimationFrame(() => {
    document.querySelectorAll('.feature-bar-fill[data-w]').forEach((bar, i) => {
      setTimeout(() => { bar.style.width = bar.dataset.w + '%'; }, i * 60);
    });
  });
}

function renderFeed(predictions) {
  const el = document.getElementById('liveFeed');
  if (!el) return;
  if (!predictions.length) {
    el.innerHTML = '<div class="feed-empty">NO PREDICTIONS YET — EXECUTE INFERENCE ABOVE</div>';
    return;
  }
  el.innerHTML = predictions.slice(0, 30).map(p => feedRow(p)).join('');
}

function feedRow(p) {
  const ts = new Date(p.timestamp).toLocaleTimeString('en-US', { hour12: false });
  const ver = (p.model_version || '').replace('model_', '').replace('_', ' ');
  const prob = (p.probability * 100).toFixed(1) + '%';
  const lat  = p.latency_ms.toFixed(1) + 'ms';
  return `
    <div class="feed-item">
      <span class="feed-ts">${ts}</span>
      <span class="feed-model">${ver}</span>
      <span class="feed-prob">${prob}</span>
      <span class="feed-lat">${lat}</span>
      <span class="feed-risk ${p.risk_level}">${p.risk_level.toUpperCase()}</span>
    </div>`;
}

function addFeedEntry(data) {
  const el = document.getElementById('liveFeed');
  if (!el) return;
  const empty = el.querySelector('.feed-empty');
  if (empty) el.innerHTML = '';

  const entry = {
    timestamp: new Date().toISOString(),
    model_version: data.model_version,
    probability: data.probability,
    latency_ms: data.latency_ms,
    risk_level: data.risk_level,
  };
  state.predictionHistory.unshift(entry);

  el.insertAdjacentHTML('afterbegin', feedRow(entry));
  if (el.children.length > 30) el.lastElementChild?.remove();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CHARTS
// ═══════════════════════════════════════════════════════════════════════════════
const CHART_DEFAULTS = {
  color: '#c8d6e8',
  font: { family: "'JetBrains Mono', monospace", size: 10 },
};
Chart.defaults.color = CHART_DEFAULTS.color;
Chart.defaults.font  = CHART_DEFAULTS.font;

function chartGrid() {
  return { color: 'rgba(200,214,232,0.06)', drawBorder: false };
}

function renderCharts() {
  renderRadarChart();
  renderDoughnutChart();
  renderBarChart();
}

function renderRadarChart() {
  const ctx = document.getElementById('radarChart');
  if (!ctx) return;
  const m = state.metrics?.metrics;
  if (!m) return;

  if (state.charts.radar) state.charts.radar.destroy();

  state.charts.radar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
      datasets: [{
        data: [m.accuracy, m.precision, m.recall, m.f1, m.roc_auc].map(v => +(v*100).toFixed(1)),
        fill: true,
        backgroundColor: 'rgba(0,245,255,0.07)',
        borderColor: '#00f5ff',
        borderWidth: 1.5,
        pointBackgroundColor: '#00f5ff',
        pointBorderColor: 'transparent',
        pointRadius: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 85, max: 100,
          grid:      { color: 'rgba(0,245,255,0.07)' },
          angleLines:{ color: 'rgba(0,245,255,0.07)' },
          ticks: {
            backdropColor: 'transparent',
            color: 'rgba(200,214,232,0.35)',
            font: { size: 9 },
            stepSize: 5,
          },
          pointLabels: { color: 'rgba(200,214,232,0.55)', font: { size: 9 } },
        },
      },
    },
  });
}

function renderDoughnutChart() {
  const ctx = document.getElementById('doughnutChart');
  if (!ctx) return;
  const ms = state.monStats;

  const low    = ms?.risk_distribution?.low    ?? 1;
  const medium = ms?.risk_distribution?.medium ?? 1;
  const high   = ms?.risk_distribution?.high   ?? 1;

  if (state.charts.doughnut) state.charts.doughnut.destroy();

  state.charts.doughnut = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['LOW', 'MEDIUM', 'HIGH'],
      datasets: [{
        data: [low, medium, high],
        backgroundColor: [
          'rgba(0,255,136,0.7)',
          'rgba(255,122,0,0.7)',
          'rgba(255,45,85,0.7)',
        ],
        borderColor: ['#00ff88','#ff7a00','#ff2d55'],
        borderWidth: 1.5,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 10, padding: 12, font: { size: 9 } },
        },
      },
    },
  });
}

function renderBarChart() {
  const ctx = document.getElementById('barChart');
  if (!ctx) return;
  const m = state.metrics?.metrics;
  if (!m) return;

  if (state.charts.bar) state.charts.bar.destroy();

  const vals = [m.accuracy, m.precision, m.recall, m.f1, m.roc_auc].map(v => +(v*100).toFixed(1));
  const colors = ['#00f5ff','#7b2fff','#00ff88','#ff7a00','#ff2d55'];
  const borders = colors;

  state.charts.bar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['ACC', 'PREC', 'REC', 'F1', 'AUC'],
      datasets: [{
        data: vals,
        backgroundColor: colors.map(c => c + '33'),
        borderColor: borders,
        borderWidth: 1.5,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: chartGrid(), ticks: { font: { size: 9 } } },
        y: {
          grid: chartGrid(),
          min: 85, max: 100,
          ticks: { font: { size: 9 }, callback: v => v + '%' },
        },
      },
    },
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  WEBSOCKET
// ═══════════════════════════════════════════════════════════════════════════════
function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url   = `${proto}://${location.host}/api/ws/live`;
  const badge = document.getElementById('wsBadge');

  try {
    const ws = new WebSocket(url);
    state.ws = ws;

    ws.onopen = () => {
      state.wsConnected = true;
      if (badge) badge.className = 'ws-badge connected';
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === 'prediction') {
          addFeedEntry(msg);
          incrementHeaderTotal();
        }
      } catch (_) {}
    };

    ws.onclose = () => {
      state.wsConnected = false;
      if (badge) badge.className = 'ws-badge';
      // Reconnect after 8 s
      setTimeout(connectWebSocket, 8000);
    };

    ws.onerror = () => ws.close();
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════════════════════
//  POLLING
// ═══════════════════════════════════════════════════════════════════════════════
function startPolling() {
  setInterval(async () => {
    await Promise.allSettled([
      loadMonitoringStats(),
      loadRecentPredictions(),
      loadSystemStatus(),
    ]);
    // Refresh doughnut with updated risk distribution
    renderDoughnutChart();
  }, POLL_MS);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════
function animateCounter(id, target, suffix = '', decimals = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = 0, duration = 1200;
  const startTime = performance.now();

  function step(now) {
    const pct = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - pct, 3);
    const val = start + (target - start) * eased;
    el.textContent = val.toFixed(decimals) + suffix;
    if (pct < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function incrementHeaderTotal() {
  const el = document.getElementById('hdr-total');
  if (!el) return;
  const cur = parseInt(el.textContent, 10) || 0;
  el.textContent = cur + 1;

  const stat = document.getElementById('stat-total');
  if (stat) stat.textContent = (parseInt(stat.textContent, 10) || 0) + 1;
}

function fmtModelType(t) {
  return ({ gradient_boosting: 'GradBoost', random_forest: 'Rnd Forest', logistic_regression: 'LogReg' })[t] || (t || '—');
}

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' });
}

function fmtUptime(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}
