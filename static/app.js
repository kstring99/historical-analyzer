let selectedFiles = [];
let currentJobId = null;
let pollInterval = null;

// --- Settings ---
function toggleSettings() {
    document.getElementById('settings-panel').classList.toggle('hidden');
}

// --- File Upload ---
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
});

function handleFiles(files) {
    selectedFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (selectedFiles.length === 0) {
        alert('Please select PDF files.');
        return;
    }
    renderFileList();
}

function renderFileList() {
    const container = document.getElementById('file-items');
    const fileList = document.getElementById('file-list');
    const addressSection = document.getElementById('address-section');
    
    container.innerHTML = '';
    let hasCityDir = false;

    selectedFiles.forEach((f, i) => {
        const type = classifyFile(f.name);
        if (type === 'city_directory') hasCityDir = true;
        
        const icon = { aerial: '🛩️', topo: '🗺️', city_directory: '📒', unknown: '📄' }[type];
        const label = { aerial: 'Aerial Photos', topo: 'Topographic Maps', city_directory: 'City Directory', unknown: 'Unknown' }[type];
        
        container.innerHTML += `
            <div class="file-item">
                <span class="file-icon">${icon}</span>
                <div class="file-info">
                    <span class="file-name">${f.name}</span>
                    <span class="file-type">${label} — ${(f.size / 1024 / 1024).toFixed(1)} MB</span>
                </div>
                <button class="remove-btn" onclick="removeFile(${i})">✕</button>
            </div>
        `;
    });

    fileList.classList.remove('hidden');
    
    // Show address inputs if city directory is included
    if (hasCityDir) {
        addressSection.classList.remove('hidden');
    } else {
        addressSection.classList.add('hidden');
    }
}

function classifyFile(name) {
    const n = name.toLowerCase();
    if (n.includes('aerial')) return 'aerial';
    if (n.includes('topo')) return 'topo';
    if (n.includes('cd') || n.includes('city') || n.includes('directory')) return 'city_directory';
    return 'unknown';
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    if (selectedFiles.length === 0) {
        document.getElementById('file-list').classList.add('hidden');
        document.getElementById('address-section').classList.add('hidden');
    } else {
        renderFileList();
    }
}

// --- Processing ---
async function startProcessing() {
    if (selectedFiles.length === 0) return;

    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('files', f));
    formData.append('provider', document.getElementById('provider-select').value);
    formData.append('api_key', document.getElementById('api-key-input').value);
    formData.append('subject_address', document.getElementById('subject-address')?.value || '');
    formData.append('adjoining_addresses', document.getElementById('adjoining-addresses')?.value || '');

    document.getElementById('upload-section').classList.add('hidden');
    document.getElementById('progress-section').classList.remove('hidden');

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        currentJobId = data.job_id;
        startPolling();
    } catch (err) {
        showError('Upload failed: ' + err.message);
    }
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${currentJobId}`);
            const data = await res.json();

            const pct = data.total_pages > 0 ? Math.round((data.progress / data.total_pages) * 100) : 0;
            document.getElementById('progress-pct').textContent = pct + '%';
            document.getElementById('progress-bar').style.width = pct + '%';
            document.getElementById('progress-status').textContent = data.current_file || 'Processing...';
            document.getElementById('progress-pages').textContent = `${data.progress} / ${data.total_pages} pages`;

            if (data.status === 'complete') {
                clearInterval(pollInterval);
                loadResults();
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                showError(data.error);
            }
        } catch (err) {
            console.error('Poll error:', err);
        }
    }, 2000);
}

// --- Results ---
async function loadResults() {
    try {
        const res = await fetch(`/api/results/${currentJobId}`);
        const data = await res.json();

        document.getElementById('progress-section').classList.add('hidden');
        document.getElementById('results-section').classList.remove('hidden');

        const container = document.getElementById('results-container');
        container.innerHTML = '';

        // Summary paragraph
        if (data.summary) {
            container.innerHTML += `
                <div class="doc-result summary-section">
                    <h2 class="section-title">Historical Documentation Summary</h2>
                    <div class="summary-text">${data.summary.replace(/\n/g, '<br>')}</div>
                </div>
            `;
        }

        if (!data.results || !Array.isArray(data.results)) {
            showError('No results available. The analysis may have failed — check job status.');
            return;
        }

        data.results.forEach(doc => {
            const sectionTitle = {
                aerial: 'Aerial Photographs',
                topo: 'Topographic Maps',
                city_directory: 'Street Directories',
            }[doc.doc_type] || doc.doc_type;

            const tableTitle = {
                aerial: 'AERIAL PHOTOGRAPH SUMMARY',
                topo: 'TOPOGRAPHIC MAP SUMMARY',
                city_directory: 'STREET DIRECTORY SUMMARY',
            }[doc.doc_type] || 'SUMMARY';

            const obsCol = doc.doc_type === 'city_directory' ? 'Occupants' : 'Observations';

            let html = `<div class="doc-result">`;
            html += `<h2 class="section-title">${sectionTitle}</h2>`;
            
            if (doc.years_reviewed && doc.years_reviewed.length > 0) {
                html += `<p class="years-reviewed">Years reviewed: <strong>${doc.years_reviewed.join(', ')}</strong></p>`;
            }

            doc.tables.forEach(table => {
                html += `<div class="zone-section">`;
                html += `<h3 class="zone-title">${table.zone_label}</h3>`;
                html += `<p class="table-caption">${tableTitle} - ${table.zone_label}</p>`;
                html += `<table class="result-table">`;
                html += `<thead><tr><th class="col-year">Year</th><th class="col-issues">Issues Noted</th><th class="col-obs">${obsCol}</th></tr></thead>`;
                html += `<tbody>`;

                if (table.rows.length === 0) {
                    html += `<tr><td colspan="3" class="no-data">No data available</td></tr>`;
                } else {
                    table.rows.forEach(row => {
                        const issueClass = row.issues_noted === 'Yes' ? 'issue-yes' : 'issue-no';
                        html += `<tr>`;
                        html += `<td class="col-year">${row.year_range}</td>`;
                        html += `<td class="col-issues ${issueClass}">${row.issues_noted}</td>`;
                        html += `<td class="col-obs">${row.observations}</td>`;
                        html += `</tr>`;
                    });
                }

                html += `</tbody></table></div>`;
            });

            html += `</div>`;
            container.innerHTML += html;
        });
    } catch (err) {
        showError('Failed to load results: ' + err.message);
    }
}

function showError(msg) {
    document.getElementById('progress-section').classList.add('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('results-container').innerHTML = `
        <div class="error-card">
            <h3>⚠️ Error</h3>
            <p>${msg}</p>
            <button class="action-btn" onclick="resetApp()">Try Again</button>
        </div>
    `;
}

// --- Export ---
function copyAllTables() {
    const tables = document.querySelectorAll('.result-table');
    let text = '';
    tables.forEach(table => {
        const caption = table.closest('.zone-section')?.querySelector('.table-caption')?.textContent || '';
        text += caption + '\n';
        table.querySelectorAll('tr').forEach(row => {
            const cells = Array.from(row.querySelectorAll('th, td')).map(c => c.textContent.trim());
            text += cells.join('\t') + '\n';
        });
        text += '\n';
    });
    navigator.clipboard.writeText(text).then(() => {
        alert('Tables copied to clipboard!');
    });
}

async function exportHTML() {
    if (!currentJobId) return;
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = `/api/export/${currentJobId}`;
    form.target = '_blank';
    const input = document.createElement('input');
    input.name = 'format';
    input.value = 'html';
    input.type = 'hidden';
    form.appendChild(input);
    document.body.appendChild(form);
    form.submit();
    document.body.removeChild(form);
}

function resetApp() {
    currentJobId = null;
    selectedFiles = [];
    if (pollInterval) clearInterval(pollInterval);
    document.getElementById('upload-section').classList.remove('hidden');
    document.getElementById('progress-section').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('file-list').classList.add('hidden');
    document.getElementById('address-section').classList.add('hidden');
    document.getElementById('file-items').innerHTML = '';
    document.getElementById('file-input').value = '';
}
