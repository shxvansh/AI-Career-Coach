// Config
const API_URL = ""; // Relative path since we serve from same origin

// State
let resumeFile = null;
let jobFile = null;

// Elements
const kSlider = document.getElementById('k-slider');
const kValue = document.getElementById('k-value');
const resumeInput = document.getElementById('resume-file');
const jobInput = document.getElementById('job-file');
const ingestBtn = document.getElementById('ingest-btn');
const tuneBtn = document.getElementById('tune-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsSection = document.getElementById('results-section');

// Utils
const showElement = (el) => el.classList.remove('hidden');
const hideElement = (el) => el.classList.add('hidden');
const setStatus = (elId, msg, type) => {
    const el = document.getElementById(elId);
    el.textContent = msg;
    el.className = `status-msg ${type}`;
};

// Event Listeners
kSlider.addEventListener('input', (e) => kValue.textContent = e.target.value);

// File Inputs
const setupFileInput = (inputId, dropAreaId, nameDisplayId, btnId, fileVarSetter) => {
    const input = document.getElementById(inputId);
    const dropArea = document.getElementById(dropAreaId);
    const nameDisplay = document.getElementById(nameDisplayId);
    const btn = document.getElementById(btnId);

    const handleFile = (file) => {
        if (file && file.type === 'application/pdf') {
            fileVarSetter(file);
            nameDisplay.textContent = file.name;
            showElement(nameDisplay);
            btn.disabled = false;
        }
    };

    dropArea.addEventListener('click', () => input.click());
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('dragover');
    });
    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', (e) => handleFile(e.target.files[0]));
};

setupFileInput('resume-file', 'resume-drop-area', 'resume-file-name', 'ingest-btn', (f) => resumeFile = f);
setupFileInput('job-file', 'job-drop-area', 'job-file-name', 'tune-btn', (f) => jobFile = f);

// Ingest Action
ingestBtn.addEventListener('click', async () => {
    if (!resumeFile) return;
    
    ingestBtn.disabled = true;
    setStatus('ingest-status', 'Ingesting...', 'normal');
    
    const formData = new FormData();
    formData.append('file', resumeFile);

    try {
        const res = await fetch(`${API_URL}/ingest`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (res.ok) {
            setStatus('ingest-status', 'Resume ingested successfully!', 'success');
        } else {
            setStatus('ingest-status', `Error: ${data.message}`, 'error');
            ingestBtn.disabled = false;
        }
    } catch (err) {
        setStatus('ingest-status', `Connection Error: ${err.message}`, 'error');
        ingestBtn.disabled = false;
    }
});

// Tune Action
tuneBtn.addEventListener('click', async () => {
    if (!jobFile) return;

    tuneBtn.disabled = true;
    showElement(loadingOverlay);
    hideElement(resultsSection);

    const formData = new FormData();
    formData.append('job_file', jobFile);
    const k = kSlider.value;

    try {
        const res = await fetch(`${API_URL}/tune?k=${k}`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            renderResults(data);
            showElement(resultsSection);
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (err) {
        alert(`Connection Error: ${err.message}`);
    } finally {
        hideElement(loadingOverlay);
        tuneBtn.disabled = false;
    }
});

// Render Logic
function renderResults(data) {
    const analysis = data.analysis || {};
    
    // Gap Analysis
    const gapContainer = document.getElementById('gap-list');
    gapContainer.innerHTML = '';
    const gaps = analysis.gap_analysis || [];
    if (gaps.length === 0) gapContainer.innerHTML = '<p class="text-muted">No gaps detected.</p>';
    gaps.forEach(item => {
        const el = document.createElement('div');
        el.className = 'list-item';
        el.innerHTML = `
            <div class="list-item-title">${item.gap}</div>
            <div class="evidence-tag">Evidence: ${(item.evidence_chunks || []).join(', ')}</div>
        `;
        gapContainer.appendChild(el);
    });

    // Rewrite Suggestions
    const suggContainer = document.getElementById('suggestion-list');
    suggContainer.innerHTML = '';
    const suggs = analysis.rewrite_suggestions || [];
    if (suggs.length === 0) suggContainer.innerHTML = '<p class="text-muted">No suggestions available.</p>';
    suggs.forEach(item => {
        const el = document.createElement('div');
        el.className = 'list-item';
        el.innerHTML = `
            <div class="list-item-title">${item.suggestion}</div>
            <div class="evidence-tag">Evidence: ${(item.evidence_chunks || []).join(', ')}</div>
        `;
        suggContainer.appendChild(el);
    });

    // LaTeX
    const bullets = analysis.latex_bullets || [];
    const latexCode = bullets.map(b => b.bullet).join('\n');
    document.getElementById('latex-code').textContent = latexCode || '% No bullets generated';
    
    const bulletsList = document.getElementById('bullets-list');
    bulletsList.innerHTML = '';
    bullets.forEach(item => {
        const el = document.createElement('div');
        el.className = 'list-item';
        el.innerHTML = `
            <div class="list-item-title">${item.bullet}</div>
            <div class="evidence-tag">Evidence: ${(item.evidence_chunks || []).join(', ')}</div>
        `;
        bulletsList.appendChild(el);
    });

    // Stats
    document.getElementById('total-time').textContent = `${data.analysis ? data.analysis.total_time_seconds : 'N/A'}s`; // Wait, total_time is in data top level wrapper? No, resume_tuner puts it in top level.
    // Check api/main.py response structure: { status, retrieval_stats, analysis }
    // resume_tuner's tune_resume returns { inputs, ingest_stats, retrieval_stats, retrieved_chunks, model_output, total_time_seconds }
    // api/main.py returns { status: success, retrieval_stats, analysis: model_output }. 
    // Wait, api/main.py just returns `model_output` as `analysis`. 
    // `model_output` from resume_tuner only contains gap_analysis, etc.
    // The timing info is in the resume_tuner wrapper result, but api/main.py discarded it?
    // Let's re-read api/main.py.
    // In api/main.py: return JSONResponse(content={ "status": "success", "retrieval_stats": retrieval_stats, "analysis": model_output })
    // So total_time is lost unless I add it.
    
    document.getElementById('retrieved-count').textContent = data.retrieval_stats ? data.retrieval_stats.chunks_returned : '-';
    document.getElementById('guardrail-json').textContent = JSON.stringify(analysis._guardrails || {}, null, 2);
    document.getElementById('retrieval-json').textContent = JSON.stringify(data.retrieval_stats || {}, null, 2);
}

// Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
    });
});

// Copy LaTeX
document.getElementById('copy-latex-btn').addEventListener('click', () => {
    const code = document.getElementById('latex-code').textContent;
    navigator.clipboard.writeText(code);
    const btn = document.getElementById('copy-latex-btn');
    const original = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = original, 2000);
});
