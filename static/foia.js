// ── State ──
let generatedRequests = [];
let currentProjectNumber = '';

// ── Init ──
document.addEventListener('DOMContentLoaded', init);

async function init() {
    loadSavedConsultantInfo();
    await loadJurisdictions();
    await loadRegistryAgencies();
    await checkFollowUps();
}

// ── Utility ──
function formatJurisdiction(key) {
    if (!key) return '';
    const parts = key.split('_');
    const state = parts.pop().toUpperCase();
    const county = parts.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
    return `${county}, ${state}`;
}

function normalizeJurisdiction(county, state) {
    return `${county.toLowerCase().replace(/\s+/g, '_')}_${state.toLowerCase().trim()}`;
}

function formatAgencyType(type) {
    const labels = {
        'fire_department': 'Fire Department',
        'health_department': 'Health Department',
        'building_department': 'Building Department',
        'state_environmental': 'State Environmental',
        'water_board': 'Water Board',
        'public_works': 'Public Works',
        'air_quality': 'Air Quality',
        'county_environmental': 'County Environmental',
    };
    return labels[type] || type;
}

function formatMethod(method) {
    return { email: 'Email', form: 'Form', portal: 'Portal', mail: 'Mail' }[method] || method;
}

function formatStatus(status) {
    const labels = {
        'draft': 'Draft',
        'sent': 'Sent',
        'follow_up_due': 'Follow-up Due',
        'follow_up_sent': 'Follow-up Sent',
        'received': 'Received',
        'no_records': 'No Records',
        'no_response': 'No Response',
    };
    return labels[status] || status;
}

function getStatusClass(status) {
    const classes = {
        'draft': 'status-draft',
        'sent': 'status-sent',
        'follow_up_due': 'status-followup',
        'follow_up_sent': 'status-sent',
        'received': 'status-received',
        'no_records': 'status-no-records',
        'no_response': 'status-no-response',
    };
    return classes[status] || 'status-draft';
}

function getMethodClass(method) {
    return `method-${method}`;
}

function formatDate(iso) {
    if (!iso) return '\u2014';
    return iso.substring(0, 10);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type) {
    type = type || 'success';
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function () {
        toast.classList.add('toast-fade');
        setTimeout(function () { toast.remove(); }, 300);
    }, 3000);
}

// ── LocalStorage ──
function loadSavedConsultantInfo() {
    const saved = localStorage.getItem('foia_consultant');
    if (saved) {
        try {
            const info = JSON.parse(saved);
            document.getElementById('consultant-name').value = info.name || '';
            document.getElementById('consultant-title').value = info.title || '';
            document.getElementById('consultant-email').value = info.email || '';
            document.getElementById('consultant-phone').value = info.phone || '';
            document.getElementById('company-name').value = info.company || '';
            document.getElementById('company-address').value = info.companyAddress || '';
        } catch (e) { /* ignore corrupt data */ }
    }
}

function saveConsultantInfo() {
    const info = {
        name: document.getElementById('consultant-name').value,
        title: document.getElementById('consultant-title').value,
        email: document.getElementById('consultant-email').value,
        phone: document.getElementById('consultant-phone').value,
        company: document.getElementById('company-name').value,
        companyAddress: document.getElementById('company-address').value,
    };
    localStorage.setItem('foia_consultant', JSON.stringify(info));
}

// ── Jurisdictions ──
async function loadJurisdictions() {
    try {
        const res = await fetch('/api/foia/jurisdictions');
        const data = await res.json();
        const jurisdictions = data.jurisdictions || [];
        populateJurisdictionDropdown('jurisdiction-select', jurisdictions);
        populateJurisdictionDropdown('agency-jurisdiction', jurisdictions);
        populateJurisdictionFilter(jurisdictions);
    } catch (err) {
        console.error('Failed to load jurisdictions:', err);
    }
}

function populateJurisdictionDropdown(selectId, jurisdictions) {
    const select = document.getElementById(selectId);
    const placeholder = select.options[0].cloneNode(true);
    const addNew = select.querySelector('option[value="__new__"]').cloneNode(true);
    select.innerHTML = '';
    select.appendChild(placeholder);
    jurisdictions.forEach(function (j) {
        const opt = document.createElement('option');
        opt.value = j;
        opt.textContent = formatJurisdiction(j);
        select.appendChild(opt);
    });
    select.appendChild(addNew);
}

function populateJurisdictionFilter(jurisdictions) {
    const select = document.getElementById('registry-jurisdiction-filter');
    const allOpt = select.options[0].cloneNode(true);
    select.innerHTML = '';
    select.appendChild(allOpt);
    jurisdictions.forEach(function (j) {
        const opt = document.createElement('option');
        opt.value = j;
        opt.textContent = formatJurisdiction(j);
        select.appendChild(opt);
    });
}

function handleJurisdictionChange() {
    const select = document.getElementById('jurisdiction-select');
    const isNew = select.value === '__new__';
    document.getElementById('new-jurisdiction-group').classList.toggle('hidden', !isNew);
    document.getElementById('new-state-group').classList.toggle('hidden', !isNew);
}

function handleAgencyJurisdictionChange() {
    const select = document.getElementById('agency-jurisdiction');
    const isNew = select.value === '__new__';
    document.getElementById('agency-new-jurisdiction-group').classList.toggle('hidden', !isNew);
    document.getElementById('agency-new-state-group').classList.toggle('hidden', !isNew);
}

// ── Follow-ups ──
async function checkFollowUps() {
    try {
        const res = await fetch('/api/foia/follow-ups');
        const data = await res.json();
        if (data.count > 0) {
            document.getElementById('followup-banner').classList.remove('hidden');
            document.getElementById('followup-count').textContent = data.count;
        }
    } catch (err) { /* silent */ }
}

function scrollToTracker() {
    document.getElementById('tracker-section').scrollIntoView({ behavior: 'smooth' });
}

// ── Form Submission ──
async function handleGenerateSubmit(e) {
    e.preventDefault();
    saveConsultantInfo();

    const jurisdictionSelect = document.getElementById('jurisdiction-select');
    var jurisdiction = jurisdictionSelect.value;

    if (jurisdiction === '__new__') {
        var county = document.getElementById('new-county').value.trim();
        var st = document.getElementById('new-state').value.trim();
        if (!county || !st) {
            showToast('Please enter county and state for the new jurisdiction.', 'error');
            return;
        }
        jurisdiction = normalizeJurisdiction(county, st);
    }

    if (!jurisdiction) {
        showToast('Please select a jurisdiction.', 'error');
        return;
    }

    var formData = new FormData();
    formData.append('jurisdiction', jurisdiction);
    formData.append('address', document.getElementById('property-address').value);
    formData.append('city', document.getElementById('city').value);
    formData.append('state', document.getElementById('state').value);
    formData.append('zip_code', document.getElementById('zip-code').value);
    formData.append('parcel_number', document.getElementById('parcel-number').value);
    formData.append('business_names', document.getElementById('business-names').value);
    formData.append('start_year', document.getElementById('start-year').value);
    formData.append('project_number', document.getElementById('project-number').value);
    formData.append('consultant_name', document.getElementById('consultant-name').value);
    formData.append('consultant_title', document.getElementById('consultant-title').value);
    formData.append('consultant_email', document.getElementById('consultant-email').value);
    formData.append('consultant_phone', document.getElementById('consultant-phone').value);
    formData.append('company_name', document.getElementById('company-name').value);
    formData.append('company_address', document.getElementById('company-address').value);

    var btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.textContent = 'Generating...';

    try {
        var res = await fetch('/api/foia/generate', { method: 'POST', body: formData });
        var data = await res.json();

        generatedRequests = data.requests || [];
        currentProjectNumber = document.getElementById('project-number').value;

        renderGeneratedRequests(generatedRequests);

        // Auto-load tracker
        document.getElementById('tracker-project').value = currentProjectNumber;
        document.getElementById('roc-project').value = currentProjectNumber;
        await loadTracker(currentProjectNumber);

        showToast('Generated ' + data.count + ' request(s)');
    } catch (err) {
        showToast('Failed to generate requests: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Requests';
    }
}

// ── Generated Requests ──
function renderGeneratedRequests(requests) {
    var section = document.getElementById('generated-section');
    var container = document.getElementById('requests-container');

    section.classList.remove('hidden');
    container.innerHTML = '';

    // Check if these are "no agency" results
    var noAgency = requests.filter(function (r) { return r.status === 'no_agency_registered'; });
    if (noAgency.length > 0 && noAgency.length === requests.length) {
        container.innerHTML =
            '<div class="no-agency-alert">' +
            '<h3>No Agencies Registered</h3>' +
            '<p>No agencies are registered for this jurisdiction. Register agencies in the Agency Registry section below, then generate requests again.</p>' +
            '<div class="needed-agencies"><h4>Standard agencies needed:</h4><ul>' +
            noAgency.map(function (a) { return '<li>' + formatAgencyType(a.agency_type) + '</li>'; }).join('') +
            '</ul></div></div>';
        document.getElementById('generated-count').textContent = '';
        return;
    }

    document.getElementById('generated-count').textContent = '(' + requests.length + ')';

    requests.forEach(function (req) {
        container.innerHTML += renderRequestCard(req);
    });
}

function renderRequestCard(req) {
    var methodBadge = '<span class="badge ' + getMethodClass(req.method) + '">' + formatMethod(req.method) + '</span>';
    var typeBadge = '<span class="badge badge-type">' + formatAgencyType(req.agency_type) + '</span>';

    var bodyHtml = '';

    if (req.method === 'email') {
        bodyHtml =
            '<div class="card-field"><label>To: ' + escapeHtml(req.email_to || '') + '</label></div>' +
            '<div class="card-field"><label>Subject: ' + escapeHtml(req.email_subject || '') + '</label></div>' +
            '<textarea class="email-body" id="email-' + req.request_id + '" readonly>' + escapeHtml(req.email_body || '') + '</textarea>' +
            '<div class="card-actions">' +
            '<button class="card-btn card-btn-primary" onclick="copyEmailBody(\'' + req.request_id + '\')">Copy to Clipboard</button>' +
            '<button class="card-btn card-btn-success" id="sent-btn-' + req.request_id + '" onclick="markSent(\'' + req.request_id + '\')">Mark as Sent</button>' +
            '</div>';
    } else if (req.method === 'form') {
        var fields = req.form_fields || {};
        var fieldsHtml = Object.keys(fields).map(function (k) {
            return '<div class="field-row"><span class="field-key">' + escapeHtml(k) + ':</span> <span class="field-val">' + escapeHtml(fields[k]) + '</span></div>';
        }).join('');
        bodyHtml =
            '<div class="card-field"><label>Form URL:</label> <a href="' + escapeHtml(req.form_url || '') + '" target="_blank" rel="noopener">' + escapeHtml(req.form_url || 'Not set') + '</a></div>' +
            '<div class="card-field"><label>Pre-filled Values:</label><div class="form-fields-preview">' + fieldsHtml + '</div></div>' +
            '<div class="card-actions">' +
            '<a href="' + escapeHtml(req.form_url || '') + '" target="_blank" rel="noopener" class="card-btn card-btn-primary">Open Form</a>' +
            '<button class="card-btn card-btn-success" id="sent-btn-' + req.request_id + '" onclick="markSent(\'' + req.request_id + '\')">Mark as Sent</button>' +
            '</div>';
    } else if (req.method === 'portal') {
        bodyHtml =
            '<div class="card-field"><label>Portal:</label> <a href="' + escapeHtml(req.portal_url || '') + '" target="_blank" rel="noopener">' + escapeHtml(req.portal_url || 'Not set') + '</a></div>' +
            '<div class="card-field"><label>Instructions:</label><pre class="instructions-text">' + escapeHtml(req.instructions || '') + '</pre></div>' +
            '<div class="card-actions">' +
            '<a href="' + escapeHtml(req.portal_url || '') + '" target="_blank" rel="noopener" class="card-btn card-btn-primary">Open Portal</a>' +
            '<button class="card-btn card-btn-success" id="sent-btn-' + req.request_id + '" onclick="markSent(\'' + req.request_id + '\')">Mark as Sent</button>' +
            '</div>';
    } else if (req.method === 'mail') {
        bodyHtml =
            '<div class="card-field"><label>Mailing Address: ' + escapeHtml(req.mailing_address || '') + '</label></div>' +
            '<textarea class="email-body" id="letter-' + req.request_id + '" readonly>' + escapeHtml(req.letter_body || '') + '</textarea>' +
            '<div class="card-actions">' +
            '<button class="card-btn card-btn-primary" onclick="printLetter(\'' + req.request_id + '\')">Print</button>' +
            '<button class="card-btn card-btn-success" id="sent-btn-' + req.request_id + '" onclick="markSent(\'' + req.request_id + '\')">Mark as Sent</button>' +
            '</div>';
    }

    return '<div class="request-card" id="card-' + req.request_id + '">' +
        '<div class="request-card-header"><div class="card-title-row">' +
        '<h3>' + escapeHtml(req.agency_name || 'Unknown Agency') + '</h3>' +
        '<div class="card-badges">' + typeBadge + ' ' + methodBadge + '</div>' +
        '</div></div>' +
        '<div class="request-card-body">' + bodyHtml + '</div></div>';
}

async function copyEmailBody(requestId) {
    var textarea = document.getElementById('email-' + requestId);
    if (!textarea) return;
    try {
        await navigator.clipboard.writeText(textarea.value);
        showToast('Email body copied to clipboard');
    } catch (err) {
        textarea.select();
        document.execCommand('copy');
        showToast('Email body copied to clipboard');
    }
}

function printLetter(requestId) {
    var textarea = document.getElementById('letter-' + requestId);
    if (!textarea) return;
    var printWindow = window.open('', '_blank');
    printWindow.document.write(
        '<html><head><title>Records Request Letter</title>' +
        '<style>body{font-family:serif;white-space:pre-wrap;padding:1in;font-size:12pt;line-height:1.6;}</style>' +
        '</head><body>' + escapeHtml(textarea.value) + '</body></html>'
    );
    printWindow.document.close();
    printWindow.print();
}

async function markSent(requestId) {
    try {
        var res = await fetch('/api/foia/requests/' + requestId + '/sent', { method: 'POST' });
        var data = await res.json();
        var btn = document.getElementById('sent-btn-' + requestId);
        if (btn) {
            btn.textContent = 'Sent!';
            btn.disabled = true;
            btn.classList.add('card-btn-disabled');
        }
        showToast('Request marked as sent. Follow-up due: ' + data.follow_up_due);
        if (currentProjectNumber) {
            await loadTracker(currentProjectNumber);
        }
    } catch (err) {
        showToast('Failed to mark as sent: ' + err.message, 'error');
    }
}

// ── Tracker ──
function loadTrackerFromInput() {
    var project = document.getElementById('tracker-project').value.trim();
    if (!project) {
        showToast('Enter a project number', 'error');
        return;
    }
    currentProjectNumber = project;
    loadTracker(project);
}

async function loadTracker(projectNumber) {
    try {
        var res = await fetch('/api/foia/project/' + encodeURIComponent(projectNumber));
        var data = await res.json();
        renderTrackerTable(data.requests || []);
    } catch (err) {
        showToast('Failed to load tracker: ' + err.message, 'error');
    }
}

function renderTrackerTable(requests) {
    var container = document.getElementById('tracker-container');

    if (requests.length === 0) {
        container.innerHTML = '<p class="tracker-empty">No requests found for this project.</p>';
        return;
    }

    var html =
        '<div class="tracker-table-wrap"><table class="tracker-table">' +
        '<thead><tr>' +
        '<th>Agency</th><th>Type</th><th>Method</th><th>Status</th>' +
        '<th>Date Sent</th><th>Follow-up Due</th><th>Actions</th>' +
        '</tr></thead><tbody>';

    requests.forEach(function (req) {
        var statusClass = getStatusClass(req.status);
        var isTerminal = ['received', 'no_records', 'no_response'].indexOf(req.status) !== -1;

        var actions = '';
        if (!isTerminal) {
            actions =
                '<button class="table-action-btn" onclick="openReceivedModal(\'' + req.request_id + '\')">Received</button>' +
                '<button class="table-action-btn" onclick="markNoRecordsAction(\'' + req.request_id + '\')">No Records</button>';
            if (req.status === 'sent' || req.status === 'follow_up_due') {
                actions += '<button class="table-action-btn" onclick="sendFollowUp(\'' + req.request_id + '\')">Follow-up</button>';
            }
        } else {
            var summary = req.response_summary ? escapeHtml(req.response_summary.substring(0, 50)) : '\u2014';
            actions = '<span class="action-complete">' + summary + '</span>';
        }

        html +=
            '<tr>' +
            '<td>' + escapeHtml(req.agency_name || '') + '</td>' +
            '<td>' + formatAgencyType(req.agency_type || '') + '</td>' +
            '<td><span class="badge ' + getMethodClass(req.method) + '">' + formatMethod(req.method) + '</span></td>' +
            '<td><span class="status-badge ' + statusClass + '">' + formatStatus(req.status) + '</span></td>' +
            '<td>' + formatDate(req.sent_at) + '</td>' +
            '<td>' + formatDate(req.follow_up_due) + '</td>' +
            '<td class="actions-cell">' + actions + '</td>' +
            '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// ── Modal ──
function openReceivedModal(requestId) {
    document.getElementById('received-request-id').value = requestId;
    document.getElementById('received-summary').value = '';
    document.getElementById('received-modal').classList.remove('hidden');
}

function closeReceivedModal() {
    document.getElementById('received-modal').classList.add('hidden');
}

async function submitReceived() {
    var requestId = document.getElementById('received-request-id').value;
    var summary = document.getElementById('received-summary').value;

    var formData = new FormData();
    formData.append('summary', summary);

    try {
        await fetch('/api/foia/requests/' + requestId + '/received', { method: 'POST', body: formData });
        closeReceivedModal();
        showToast('Request marked as received');
        if (currentProjectNumber) {
            await loadTracker(currentProjectNumber);
        }
    } catch (err) {
        showToast('Failed to mark received: ' + err.message, 'error');
    }
}

async function markNoRecordsAction(requestId) {
    try {
        await fetch('/api/foia/requests/' + requestId + '/no-records', { method: 'POST' });
        showToast('Marked as no records found');
        if (currentProjectNumber) {
            await loadTracker(currentProjectNumber);
        }
    } catch (err) {
        showToast('Failed to update: ' + err.message, 'error');
    }
}

async function sendFollowUp(requestId) {
    try {
        await fetch('/api/foia/requests/' + requestId + '/sent', { method: 'POST' });
        showToast('Follow-up sent \u2014 timer reset to 14 days');
        if (currentProjectNumber) {
            await loadTracker(currentProjectNumber);
        }
    } catch (err) {
        showToast('Failed: ' + err.message, 'error');
    }
}

// ── Agency Registry ──
function toggleRegistryForm() {
    var container = document.getElementById('registry-form-container');
    var btn = document.getElementById('registry-toggle-btn');
    var isHidden = container.classList.contains('hidden');
    container.classList.toggle('hidden');
    btn.textContent = isHidden ? '- Cancel' : '+ Add Agency';
}

function handleMethodChange() {
    var method = document.getElementById('agency-method').value;
    document.getElementById('agency-form-url-group').classList.toggle('hidden', method !== 'form');
    document.getElementById('agency-portal-url-group').classList.toggle('hidden', method !== 'portal');
    document.getElementById('agency-mailing-group').classList.toggle('hidden', method !== 'mail');
}

async function handleAddAgency(e) {
    e.preventDefault();

    var jurisdiction = document.getElementById('agency-jurisdiction').value;
    if (jurisdiction === '__new__') {
        var county = document.getElementById('agency-new-county').value.trim();
        var st = document.getElementById('agency-new-state').value.trim();
        if (!county || !st) {
            showToast('Enter county and state', 'error');
            return;
        }
        jurisdiction = normalizeJurisdiction(county, st);
    }
    if (!jurisdiction) {
        showToast('Select a jurisdiction', 'error');
        return;
    }

    var formData = new FormData();
    formData.append('jurisdiction', jurisdiction);
    formData.append('agency_type', document.getElementById('agency-type').value);
    formData.append('name', document.getElementById('agency-name').value);
    formData.append('method', document.getElementById('agency-method').value);
    formData.append('contact_email', document.getElementById('agency-contact-email').value);
    formData.append('contact_phone', document.getElementById('agency-contact-phone').value);
    formData.append('form_url', document.getElementById('agency-form-url').value);
    formData.append('portal_url', document.getElementById('agency-portal-url').value);
    formData.append('mailing_address', document.getElementById('agency-mailing-address').value);
    formData.append('notes', document.getElementById('agency-notes').value);

    try {
        var res = await fetch('/api/foia/agencies', { method: 'POST', body: formData });
        var data = await res.json();
        showToast('Agency registered: ' + data.agency_id);

        document.getElementById('agency-form').reset();
        toggleRegistryForm();
        await loadJurisdictions();
        await loadRegistryAgencies();
    } catch (err) {
        showToast('Failed to add agency: ' + err.message, 'error');
    }
}

async function loadRegistryAgencies() {
    var jurisdiction = document.getElementById('registry-jurisdiction-filter').value;

    if (!jurisdiction) {
        try {
            var res = await fetch('/api/foia/jurisdictions');
            var data = await res.json();
            var jurisdictions = data.jurisdictions || [];
            if (jurisdictions.length === 0) {
                renderRegistryAgencies([]);
                return;
            }
            var results = await Promise.all(
                jurisdictions.map(function (j) {
                    return fetch('/api/foia/agencies/' + encodeURIComponent(j)).then(function (r) { return r.json(); });
                })
            );
            var allAgencies = [];
            results.forEach(function (r) {
                (r.agencies || []).forEach(function (a) { allAgencies.push(a); });
            });
            renderRegistryAgencies(allAgencies);
        } catch (err) {
            console.error('Failed to load agencies:', err);
        }
        return;
    }

    try {
        var res2 = await fetch('/api/foia/agencies/' + encodeURIComponent(jurisdiction));
        var data2 = await res2.json();
        renderRegistryAgencies(data2.agencies || []);
    } catch (err) {
        showToast('Failed to load agencies', 'error');
    }
}

function renderRegistryAgencies(agencies) {
    var container = document.getElementById('registry-list');

    if (agencies.length === 0) {
        container.innerHTML = '<p class="tracker-empty">No agencies registered yet. Click "+ Add Agency" to get started.</p>';
        return;
    }

    var html = '';
    agencies.forEach(function (a) {
        html +=
            '<div class="agency-card">' +
            '<div class="agency-card-header">' +
            '<strong>' + escapeHtml(a.name) + '</strong>' +
            '<div class="card-badges">' +
            '<span class="badge badge-type">' + formatAgencyType(a.agency_type) + '</span>' +
            '<span class="badge ' + getMethodClass(a.method) + '">' + formatMethod(a.method) + '</span>' +
            '</div></div>' +
            '<div class="agency-card-body">' +
            '<span class="agency-detail">Jurisdiction: ' + formatJurisdiction(a.jurisdiction) + '</span>' +
            (a.contact_email ? '<span class="agency-detail">Email: ' + escapeHtml(a.contact_email) + '</span>' : '') +
            (a.contact_phone ? '<span class="agency-detail">Phone: ' + escapeHtml(a.contact_phone) + '</span>' : '') +
            (a.form_url ? '<span class="agency-detail">Form: <a href="' + escapeHtml(a.form_url) + '" target="_blank">' + escapeHtml(a.form_url) + '</a></span>' : '') +
            (a.portal_url ? '<span class="agency-detail">Portal: <a href="' + escapeHtml(a.portal_url) + '" target="_blank">' + escapeHtml(a.portal_url) + '</a></span>' : '') +
            (a.mailing_address ? '<span class="agency-detail">Mail: ' + escapeHtml(a.mailing_address) + '</span>' : '') +
            '</div>' +
            '<div class="agency-card-stats">' +
            '<span>Used: ' + (a.times_used || 0) + ' times</span>' +
            (a.avg_response_days ? '<span>Avg response: ' + a.avg_response_days + ' days</span>' : '') +
            '</div></div>';
    });
    container.innerHTML = html;
}

// ── ROC ──
async function generateROC() {
    var project = document.getElementById('roc-project').value.trim();
    if (!project) {
        showToast('Enter a project number', 'error');
        return;
    }

    try {
        var res = await fetch('/api/foia/roc/' + encodeURIComponent(project));
        var html = await res.text();
        document.getElementById('roc-container').innerHTML = html;
        document.getElementById('roc-actions').classList.remove('hidden');
        showToast('ROC table generated');
    } catch (err) {
        showToast('Failed to generate ROC: ' + err.message, 'error');
    }
}

function copyROCHtml() {
    var container = document.getElementById('roc-container');
    navigator.clipboard.writeText(container.innerHTML).then(function () {
        showToast('ROC HTML copied to clipboard');
    }).catch(function () {
        showToast('Failed to copy', 'error');
    });
}
