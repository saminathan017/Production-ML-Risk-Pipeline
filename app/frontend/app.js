/**
 * ML Risk Pipeline - Creative Modern JavaScript
 */

const API_BASE = '/api';

document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    loadMetrics();
    document.getElementById('predictBtn')?.addEventListener('click', makePrediction);
});

async function makePrediction() {
    const input = document.getElementById('featuresInput');
    const result = document.getElementById('predictionResult');
    const btn = document.getElementById('predictBtn');

    try {
        const text = input.value.trim();
        if (!text) {
            showError(result, 'Please enter feature values');
            return;
        }

        const features = text.split(',').map(f => parseFloat(f.trim()));
        if (features.some(isNaN) || features.length !== 30) {
            showError(result, 'Please enter exactly 30 valid numbers');
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="loading-pulse">Analyzing...</span>';

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });

        if (!response.ok) throw new Error((await response.json()).detail);

        const data = await response.json();
        displayPrediction(data, result);

    } catch (error) {
        showError(result, error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <span>Analyze Risk</span>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M4 10h12m-6-6l6 6-6 6" stroke="currentColor" stroke-width="2"/>
            </svg>
        `;
    }
}

function displayPrediction(data, container) {
    // Note: sklearn breast cancer dataset has inverted labels
    // 0 = malignant (bad), 1 = benign (good)
    // So we need to invert what we show
    const isHighRisk = data.prediction === 0;  // 0 means malignant in sklearn
    const classification = isHighRisk ? 'Malignant' : 'Benign';
    const riskLevel = isHighRisk ? 'High' : 'Low';

    const colors = {
        'Low': { bg: 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)', border: '#10b981' },
        'Medium': { bg: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)', border: '#f59e0b' },
        'High': { bg: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)', border: '#ef4444' }
    };

    const color = colors[riskLevel];

    container.innerHTML = `
        <div class="prediction-display" style="background: ${color.bg}; border-left-color: ${color.border}">
            <div class="prediction-grid">
                <div class="prediction-item">
                    <div class="prediction-label">Classification</div>
                    <div class="prediction-value">${classification}</div>
                </div>
                <div class="prediction-item">
                    <div class="prediction-label">Confidence</div>
                    <div class="prediction-value">${(data.probability * 100).toFixed(1)}%</div>
                </div>
                <div class="prediction-item">
                    <div class="prediction-label">Risk Level</div>
                    <div class="prediction-value" style="color: ${color.border}">${riskLevel}</div>
                </div>
                <div class="prediction-item">
                    <div class="prediction-label">Latency</div>
                    <div class="prediction-value">${data.latency_ms.toFixed(1)}ms</div>
                </div>
            </div>
        </div>
    `;
}

async function loadModelInfo() {
    const container = document.getElementById('modelInfoResult');
    try {
        const response = await fetch(`${API_BASE}/model`);
        if (!response.ok) throw new Error('Failed to load model');

        const data = await response.json();

        container.innerHTML = `
            <div class="model-info-grid">
                <div class="info-box">
                    <div class="info-label">Version</div>
                    <div class="info-value">${data.model_version.split('_')[1] || 'v1.0'}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Algorithm</div>
                    <div class="info-value">${formatModelType(data.model_type)}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Training Date</div>
                    <div class="info-value">${new Date(data.trained_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Features</div>
                    <div class="info-value">${data.feature_count} Dims</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Dataset ID</div>
                    <div class="info-value">${data.dataset_hash.substring(0, 8)}...</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Status</div>
                    <div class="info-value"><span class="status-badge">Active</span></div>
                </div>
            </div>
        `;
    } catch (error) {
        container.innerHTML = `<div class="error-display">${error.message}</div>`;
    }
}

async function loadMetrics() {
    const container = document.getElementById('metricsResult');
    try {
        const response = await fetch(`${API_BASE}/metrics`);
        if (!response.ok) throw new Error('Failed to load metrics');

        const data = await response.json();

        const metricOrder = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
        const sortedMetrics = metricOrder.map(key => [key, data.metrics[key]]);

        container.innerHTML = `
            <div class="metrics-grid-modern">
                ${sortedMetrics.map(([key, value]) => `
                    <div class="metric-box-modern">
                        <div class="metric-number">${(value * 100).toFixed(1)}%</div>
                        <div class="metric-name">${key.replace('_', '-')}</div>
                    </div>
                `).join('')}
            </div>
            <div class="metrics-footer">
                <span><strong>Model:</strong> ${data.model_version}</span>
                <span><strong>Evaluated:</strong> ${new Date(data.evaluated_at).toLocaleDateString()}</span>
                <span><strong>Test Samples:</strong> ${data.dataset_size}</span>
            </div>
        `;
    } catch (error) {
        container.innerHTML = `<div class="error-display">${error.message}</div>`;
    }
}

function formatModelType(type) {
    const names = {
        'gradient_boosting': 'Gradient Boost',
        'random_forest': 'Random Forest',
        'logistic_regression': 'Logistic Reg'
    };
    return names[type] || type;
}

function showError(container, message) {
    container.innerHTML = `<div class="error-display">⚠️ ${message}</div>`;
}
