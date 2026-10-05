// BTC-Forecaster AI - Interactive Frontend Controller

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
      if (!response.ok) throw new Error('Gagal memuat data prediksi');
      forecastData = await response.json();
    }

    // Render Scenarios
    renderScenarioButtons();

    // Select default scenario (primary test window)
    selectScenario(forecastData.scenarios[0].id);

    // Setup chart toggles
    setupToggles();
  } catch (err) {
    console.error('Error loading data:', err);
    document.getElementById('scenario-description-text').innerHTML =
      '<span class="text-rose-400 font-semibold">Gagal memuat dataset prediksi: ' + err.message + '</span>';
  }
}

// Render Scenario Selector Buttons
function renderScenarioButtons() {
  const container = document.getElementById('scenario-buttons');
  container.innerHTML = '';

  forecastData.scenarios.forEach((sc, idx) => {
    const btn = document.createElement('button');
    btn.id = `btn-${sc.id}`;
    btn.className = `px-3.5 py-1.5 rounded-xl text-xs font-semibold transition flex items-center space-x-1.5 ${
      idx === 0
        ? 'bg-sky-500 text-white shadow-lg shadow-sky-500/25'
        : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700'
    }`;
    btn.innerHTML = `<span>${sc.name}</span>`;
    btn.addEventListener('click', () => selectScenario(sc.id));
    container.appendChild(btn);
  });
}

// Select and Activate Scenario
function selectScenario(scenarioId) {
  currentScenario = forecastData.scenarios.find((s) => s.id === scenarioId);
  if (!currentScenario) return;

  // Update button active state
  forecastData.scenarios.forEach((sc) => {
    const btn = document.getElementById(`btn-${sc.id}`);
    if (btn) {
      if (sc.id === scenarioId) {
        btn.className = 'px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-sky-500 text-white shadow-lg shadow-sky-500/25 transition';
      } else {
        btn.className = 'px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition';
      }
    }
  });

  // Update info boxes
  document.getElementById('scenario-description-text').innerHTML = `
    <strong>${currentScenario.name} (${currentScenario.tag}):</strong> ${currentScenario.description}
  `;

  document.getElementById('scenario-summary-badge').innerHTML = `
    <span class="text-sky-400 font-bold">Seq2Seq MAE: $${currentScenario.metrics.seq2seq_mae_usd}</span> &bull; 
    <span class="text-purple-400">Baseline MAE: $${currentScenario.metrics.baseline_mae_usd}</span>
  `;

  // Render Chart
  renderChart();

  // Render Table
  renderTable();
}

// Render Chart.js
function renderChart() {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  const showHistory = document.getElementById('toggle-history').checked;
  const showActual = document.getElementById('toggle-actual').checked;
  const showSeq2Seq = document.getElementById('toggle-seq2seq').checked;
  const showBaseline = document.getElementById('toggle-baseline').checked;

  const hist = currentScenario.historical;
  const fore = currentScenario.forecast;

  let labels = [];
  let actualData = [];
  let seq2seqData = [];
  let baselineData = [];

  if (showHistory) {
    labels = [...hist.dates, ...fore.dates];
    actualData = [...hist.prices, ...fore.actual_usd];
    // For forecast lines, prepend null for historical period
    const nullPads = new Array(hist.prices.length).fill(null);
    // Connect seamless starting point at t0 (last historical point)
    const t0 = hist.prices[hist.prices.length - 1];
    nullPads[nullPads.length - 1] = t0;

    seq2seqData = [...nullPads, ...fore.seq2seq_usd];
    baselineData = [...nullPads, ...fore.baseline_usd];
  } else {
    labels = fore.dates;
    actualData = fore.actual_usd;
    seq2seqData = fore.seq2seq_usd;
    baselineData = fore.baseline_usd;
  }

  const datasets = [];

  if (showActual) {
    datasets.push({
      label: 'Data Aktual (Ground Truth)',
      data: actualData,
      borderColor: '#10b981',
      backgroundColor: 'rgba(16, 185, 129, 0.08)',
      borderWidth: 2.5,
      pointRadius: (ctx) => {
        // Larger point on the forecast portion
        const idx = ctx.dataIndex;
        if (showHistory && idx < hist.prices.length) return 0;
        return 3.5;
      },
      pointHoverRadius: 6,
      fill: true,
      tension: 0.2
    });
  }

  if (showSeq2Seq) {
    datasets.push({
      label: 'Prediksi Seq2Seq Autoregressive',
      data: seq2seqData,
      borderColor: '#38bdf8',
      borderDash: [5, 4],
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.2
    });
  }

  if (showBaseline) {
    datasets.push({
      label: 'Prediksi Baseline LSTM',
      data: baselineData,
      borderColor: '#a855f7',
      borderDash: [2, 2],
      backgroundColor: 'transparent',
      borderWidth: 1.8,
      pointRadius: 3,
      pointHoverRadius: 5,
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
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: '#111827',
          titleColor: '#f3f4f6',
          bodyColor: '#e5e7eb',
          borderColor: '#374151',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: function (context) {
              const val = context.raw;
              if (val === null || val === undefined) return null;
              return ` ${context.dataset.label}: ${formatUSD(val)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(55, 65, 81, 0.35)',
            drawBorder: false
          },
          ticks: {
            color: '#9ca3af',
            maxTicksLimit: 14,
            font: { size: 10 }
          }
        },
        y: {
          grid: {
            color: 'rgba(55, 65, 81, 0.35)',
            drawBorder: false
          },
          ticks: {
            color: '#9ca3af',
            callback: function (val) {
              return '$' + val.toLocaleString();
            },
            font: { size: 10 }
          }
        }
      }
    }
  });
}

// Render Table
function renderTable() {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';

  const fore = currentScenario.forecast;

  for (let i = 0; i < fore.hours.length; i++) {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-800/40 transition';

    const actual = fore.actual_usd[i];
    const seq = fore.seq2seq_usd[i];
    const diffSeq = fore.diff_seq2seq_usd[i];
    const base = fore.baseline_usd[i];
    const diffBase = fore.diff_baseline_usd[i];

    tr.innerHTML = `
      <td class="py-2.5 px-4 font-semibold text-slate-300">${fore.hours[i]}</td>
      <td class="py-2.5 px-4 text-slate-400 text-xs">${fore.dates[i]}</td>
      <td class="py-2.5 px-4 text-emerald-400 font-semibold">${formatUSD(actual)}</td>
      <td class="py-2.5 px-4 text-sky-400 font-semibold">${formatUSD(seq)}</td>
      <td class="py-2.5 px-4">
        <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-sky-500/10 text-sky-300">
          $${diffSeq.toFixed(2)}
        </span>
      </td>
      <td class="py-2.5 px-4 text-purple-400">${formatUSD(base)}</td>
      <td class="py-2.5 px-4">
        <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-500/10 text-purple-300">
          $${diffBase.toFixed(2)}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  }
}

// Event Listeners for Filters
function setupToggles() {
  ['toggle-history', 'toggle-actual', 'toggle-seq2seq', 'toggle-baseline'].forEach((id) => {
    document.getElementById(id).addEventListener('change', () => {
      renderChart();
    });
  });
}

// Start
document.addEventListener('DOMContentLoaded', initApp);
