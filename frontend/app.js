// Config
const API_URL = ""; // Relative path since we serve from same origin

// State
let uploadedFiles = [];

// Elements
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const resetChatBtn = document.getElementById('reset-chat-btn');
const fileInput = document.getElementById('file-upload-input');
const attachmentPreview = document.getElementById('attachment-preview');
const loadingOverlay = document.getElementById('loading-overlay');

// Utils
function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
    return container;
}

const toastContainer = createToastContainer();

function showToast(message, type = 'normal') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-content">${message}</span>
        <span class="toast-close">&times;</span>
    `;

    toastContainer.appendChild(toast);

    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);

    // Close button
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    });

    // Auto dismiss
    setTimeout(() => {
        if (document.body.contains(toast)) {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
}

function addMessage(role, text) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    el.textContent = text;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Event Listeners

// Chat Logic
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage('user', text);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    try {
        const res = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        
        const data = await res.json();
        
        if (data.response) {
             addMessage('ai', data.response);
        } else {
             // Fallback if response structure is different
             addMessage('ai', JSON.stringify(data));
        }
        
    } catch (err) {
        addMessage('ai', `Error: ${err.message}`);
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

resetChatBtn.addEventListener('click', async () => {
    await fetch(`${API_URL}/reset_chat`, { method: 'POST' });
    chatMessages.innerHTML = '<div class="message ai">Chat reset. Hello! I am your AI Career Coach. Upload your resume and a job description to get started.</div>';
    uploadedFiles = [];
    renderAttachments();
    showToast('Chat history reset', 'normal');
});

// File Upload Logic
fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    // Show loading
    const loadingId = 'uploading-toast';
    // We don't have a persistent toast ID logic, but multiple toasts are fine.
    showToast('Uploading files...', 'normal');

    for (const file of files) {
        const formData = new FormData();
        
        // Determine endpoint based on file type or just use generic ingest?
        // The current backend has /ingest for Resume and /tune for Job.
        // But /tune triggers a full analysis which we might not want yet.
        // Let's deduce: if name contains "resume", use /ingest?
        // Or better: Use /ingest for everything and let backend handle it?
        // Backend /ingest only expects 'file'.
        // Backend /tune expects 'job_file'.
        
        // Let's try to be smart.
        let endpoint = '/ingest';
        let paramName = 'file';
        
        // Simple heuristic: if we already have a resume, assume next is job?
        // Or check filename.
        if (file.name.toLowerCase().includes('job') || file.name.toLowerCase().includes('description')) {
             endpoint = '/tune'; // This endpoint is heavy, it triggers analysis.
             paramName = 'job_file';
             // For now, let's just upload it. 
             // Ideally we should have a lightweight /upload_job endpoint.
             // But sticking to /tune is okay if we ignore the result or use it.
        }
        
        formData.append(paramName, file);

        try {
            // Note: If using /tune, we might need to suppress the analysis result if we just want to chat.
            // But getting the analysis is fine.
            
            // If we are sending job_file, /tune requires 'k' param? Defaults to 4.
            const res = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                uploadedFiles.push(file.name);
                showToast(`${file.name} uploaded successfully!`, 'success');
                
                // If it was a job file via /tune, the backend returns "analysis". 
                // We could display it, or just let the chat know.
                if (endpoint === '/tune') {
                    addMessage('ai', "I've analyzed the job description. I'm ready to discuss gaps in your resume.");
                }
                
                // Enable chat
                chatInput.disabled = false;
                sendBtn.disabled = false;
                
            } else {
                showToast(`Error uploading ${file.name}: ${data.message}`, 'error');
            }
        } catch (err) {
            showToast(`Connection error: ${err.message}`, 'error');
        }
    }
    
    renderAttachments();
    // clear input
    fileInput.value = '';
});

function renderAttachments() {
    attachmentPreview.innerHTML = '';
    uploadedFiles.forEach(name => {
        const chip = document.createElement('div');
        chip.className = 'file-chip';
        chip.innerHTML = `<span>📄 ${name}</span>`;
        attachmentPreview.appendChild(chip);
    });
}
