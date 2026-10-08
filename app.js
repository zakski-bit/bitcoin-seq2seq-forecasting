// BTC-CORE-RPC // Cypherpunk Terminal Controller

let forecastData = null;
let currentScenario = null;
let chartInstance = null;

const formatUSD = (val) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(val);
};

// Initialize Application
async function initApp() {
  try {
    if (window.FORECAST_DATA) {
      forecastData = window.FORECAST_DATA;
    } else {
      const response = await fetch('data/forecast_data.json');
      if (!response.ok) throw new Error('Failed to load forecast data stream');
      forecastData = await response.json();
    }

    // Default to scenario 1 (Primary Benchmark)
    switchScenario('scenario_1');

    // Register Keyboard Shortcuts for authentic terminal navigation
    window.addEventListener('keydown', (e) => {
      if (e.key === '1') switchScenario('scenario_1');
      if (e.key === '2') switchScenario('scenario_2');
      if (e.key === '3') switchScenario('scenario_3');
    });

  } catch (err) {
    console.error('Fatal initialization error:', err);
    const desc = document.getElementById('scenario-desc');
    if (desc) {
      desc.innerHTML = `<span class="text-term-red">[ERROR] Could not connect to RPC telemetry: ${err.message}</span>`;
    }
  }
}

// Switch Active Forecast Scenario
function switchScenario(scenarioId) {
  if (!forecastData) return;
  currentScenario = forecastData.scenarios.find((s) => s.id === scenarioId);
  if (!currentScenario) return;

  // Update Button States
  ['scenario_1', 'scenario_2', 'scenario_3'].forEach((id, idx) => {
    const btn = document.getElementById(`btn-sc${idx + 1}`);
    if (btn) {
      if (id === scenarioId) {
        btn.className = 'px-2.5 py-1 text-xs font-mono font-bold border border-term-cyan bg-term-cyan/20 text-term-cyan transition';
      } else {
        btn.className = 'px-2.5 py-1 text-xs font-mono border border-term-borderDim bg-[#0a0f18] hover:border-term-cyan/60 text-slate-400 transition';
      }
    }
  });

  // Update Telemetry Header
  const nameEl = document.getElementById('scenario-name');
  const descEl = document.getElementById('scenario-desc');
  if (nameEl) nameEl.textContent = `${currentScenario.name} [${currentScenario.tag}]`;
  if (descEl) descEl.textContent = currentScenario.description;

  // Update Dynamic KPI values
  const kpiBase = document.getElementById('kpi-base-mae');
  const kpiSeq = document.getElementById('kpi-seq-mae');
  if (kpiBase && kpiSeq) {
    kpiBase.textContent = `$${currentScenario.metrics.baseline_mae_usd.toFixed(2)}`;
    kpiSeq.textContent = `$${currentScenario.metrics.seq2seq_mae_usd.toFixed(2)}`;
  }

  // Render Charts & Matrix
  renderTerminalChart();
  renderMatrixTable();
}

// Update Visibility from Checkboxes
function updateChartVisibility() {
  if (!chartInstance) return;
  renderTerminalChart();
}

// Render High-Contrast Terminal Chart
function renderTerminalChart() {
  const canvas = document.getElementById('terminalChart');
  if (!canvas || !currentScenario) return;
  const ctx = canvas.getContext('2d');

  const showActual = document.getElementById('chk-actual')?.checked ?? true;
  const showSeq2Seq = document.getElementById('chk-seq2seq')?.checked ?? true;
  const showBaseline = document.getElementById('chk-baseline')?.checked ?? true;
  const showHistory = document.getElementById('chk-history')?.checked ?? true;

  const hist = currentScenario.historical;
  const fore = currentScenario.forecast;

  let labels = [];
  let actualData = [];
  let seq2seqData = [];
  let baselineData = [];

  if (showHistory) {
    labels = [...hist.dates, ...fore.dates];
    actualData = [...hist.prices, ...fore.actual_usd];

    const nullPads = new Array(hist.prices.length).fill(null);
    // Anchor t0 to last historical point for continuous step progression
    nullPads[nullPads.length - 1] = hist.prices[hist.prices.length - 1];

    seq2seqData = [...nullPads, ...fore.seq2seq_usd];
    baselineData = [...nullPads, ...fore.baseline_usd];
  } else {
    labels = fore.dates;
    actualData = fore.actual_usd;
    seq2seqData = fore.seq2seq_usd;
    baselineData = fore.baseline_usd;
  }

  const datasets = [];

  // Ground Truth (Actual Price) - Neon Green
  if (showActual) {
    datasets.push({
      label: 'ACTUAL_GROUND_TRUTH',
      data: actualData,
      borderColor: '#00ff66',
      backgroundColor: 'rgba(0, 255, 102, 0.04)',
      borderWidth: 2,
      pointRadius: (ctx) => {
        const idx = ctx.dataIndex;
        if (showHistory && idx < hist.prices.length) return 0;
        return 3;
      },
      pointHoverRadius: 5,
      pointBackgroundColor: '#00ff66',
      fill: true,
      tension: 0.1
    });
  }

  // Seq2Seq Autoregressive - Neon Cyan
  if (showSeq2Seq) {
    datasets.push({
      label: 'SEQ2SEQ_AUTOREGRESSIVE',
      data: seq2seqData,
      borderColor: '#00e5ff',
      borderDash: [5, 4],
      backgroundColor: 'transparent',
      borderWidth: 2.2,
      pointRadius: 3.5,
      pointHoverRadius: 6,
      pointBackgroundColor: '#00e5ff',
      tension: 0.15
    });
  }

  // Baseline LSTM - Vivid Magenta/Purple
  if (showBaseline) {
    datasets.push({
      label: 'BASELINE_LSTM_DIRECT',
      data: baselineData,
      borderColor: '#d946ef',
      borderDash: [2, 2],
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 2.5,
      pointHoverRadius: 5,
      pointBackgroundColor: '#d946ef',
      tension: 0.2
    });
  }

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#060a12',
          titleColor: '#00e5ff',
          bodyColor: '#f8fafc',
          borderColor: '#0e3a47',
          borderWidth: 1,
          padding: 10,
          titleFont: { family: 'JetBrains Mono', size: 11, weight: 'bold' },
          bodyFont: { family: 'JetBrains Mono', size: 11 },
          callbacks: {
            title: function (items) {
              return `[ STEP // ${items[0].label} ]`;
            },
            label: function (context) {
              const val = context.raw;
              if (val === null || val === undefined) return null;
              return `> ${context.dataset.label.padEnd(23, ' ')} : ${formatUSD(val)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(14, 58, 71, 0.4)',
            drawBorder: true,
            borderColor: '#0e3a47'
          },
          ticks: {
            color: '#64748b',
            maxTicksLimit: 14,
            font: { family: 'JetBrains Mono', size: 10 }
          }
        },
        y: {
          grid: {
            color: 'rgba(14, 58, 71, 0.4)',
            drawBorder: true,
            borderColor: '#0e3a47'
          },
          ticks: {
            color: '#00e5ff',
            callback: function (val) {
              return '$' + val.toLocaleString();
            },
            font: { family: 'JetBrains Mono', size: 10 }
          }
        }
      }
    }
  });
}

// Render Hour-by-Hour Evaluation Matrix Table
function renderMatrixTable() {
  const tbody = document.getElementById('matrix-tbody');
  if (!tbody || !currentScenario) return;
  tbody.innerHTML = '';

  const fore = currentScenario.forecast;

  for (let i = 0; i < fore.hours.length; i++) {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-[#0c1422] transition border-b border-term-borderDim/20';

    const actual = fore.actual_usd[i];
    const seq = fore.seq2seq_usd[i];
    const diffSeq = fore.diff_seq2seq_usd[i];
    const base = fore.baseline_usd[i];
    const diffBase = fore.diff_baseline_usd[i];

    // Status indicator
    const seqColorClass = diffSeq < 100 ? 'text-term-green font-bold' : 'text-term-cyan';

    tr.innerHTML = `
      <td class="py-1.5 px-3 border-r border-term-borderDim/40 font-bold text-slate-300">${fore.hours[i]}</td>
      <td class="py-1.5 px-3 border-r border-term-borderDim/40 text-slate-400">${fore.dates[i]}</td>
      <td class="py-1.5 px-3 border-r border-term-borderDim/40 text-term-green font-bold">${formatUSD(actual)}</td>
      <td class="py-1.5 px-3 border-r border-term-borderDim/40 text-term-cyan font-bold">${formatUSD(seq)}</td>
      <td class="py-1.5 px-3 border-r border-term-borderDim/40 ${seqColorClass}">$${diffSeq.toFixed(2)}</td>
      <td class="py-1.5 px-3 border-r border-term-borderDim/40 text-term-purple">${formatUSD(base)}</td>
      <td class="py-1.5 px-3 text-term-purple/80">$${diffBase.toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  }
}

// Run on DOM ready
document.addEventListener('DOMContentLoaded', initApp);
