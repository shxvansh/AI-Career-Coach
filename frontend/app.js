// Config
const API_URL = ""; 

// State
let uploadedFiles = [];

// Elements
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-upload-input');
const attachmentPreview = document.getElementById('attachment-preview');
const loadingOverlay = document.getElementById('loading-overlay');
const uploadTrigger = document.getElementById('upload-trigger');
const uploadMenu = document.getElementById('upload-menu');

function openUploadMenu() {
    if (uploadMenu.classList.contains('hidden')) {
        uploadMenu.classList.remove('hidden');
        // allow layout before animating
        requestAnimationFrame(() => uploadMenu.classList.add('show'));
    } else {
        uploadMenu.classList.add('show');
    }
}

function closeUploadMenu() {
    uploadMenu.classList.remove('show');
    setTimeout(() => uploadMenu.classList.add('hidden'), 200);
}

// Toggle Upload Menu
uploadTrigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (uploadMenu.classList.contains('show')) {
        closeUploadMenu();
    } else {
        openUploadMenu();
    }
});

// Close menu when clicking outside
document.addEventListener('click', (e) => {
    if (!uploadMenu.contains(e.target) && !uploadTrigger.contains(e.target)) {
        closeUploadMenu();
    }
});

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
    setTimeout(() => toast.classList.add('show'), 10);

    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    });

    setTimeout(() => {
        if (document.body.contains(toast)) {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
}

function escapeHtml(unsafe) {
    return (unsafe || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderMarkdown(mdText) {
    const escaped = escapeHtml(mdText);

    // Inline transforms
    const withInline = escaped
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    const lines = withInline.split('\n');
    const out = [];
    let inList = false;
    let inCodeBlock = false;

    function closeList() {
        if (inList) {
            out.push('</ul>');
            inList = false;
        }
    }

    for (const rawLine of lines) {
        const line = rawLine.trimEnd();

        // Code blocks
        if (line.startsWith('```')) {
            if (inCodeBlock) {
                out.push('</code></pre>');
                inCodeBlock = false;
            } else {
                out.push('<pre><code>');
                inCodeBlock = true;
            }
            continue;
        }
        if (inCodeBlock) {
            out.push(line + '\n');
            continue;
        }

        // Blank line
        if (!line.trim()) {
            closeList();
            continue;
        }

        // Headings
        if (line.startsWith('### ')) {
            closeList();
            out.push(`<h3>${line.slice(4).trim()}</h3>`);
            continue;
        }
        if (line.startsWith('## ')) {
            closeList();
            out.push(`<h2>${line.slice(3).trim()}</h2>`);
            continue;
        }
        if (line.startsWith('# ')) {
            closeList();
            out.push(`<h1>${line.slice(2).trim()}</h1>`);
            continue;
        }

        // Bullet list
        if (line.startsWith('- ') || line.startsWith('* ')) {
            if (!inList) {
                out.push('<ul>');
                inList = true;
            }
            out.push(`<li>${line.slice(2).trim()}</li>`);
            continue;
        }

        // Default paragraph
        closeList();
        out.push(`<p>${line.trim()}</p>`);
    }

    closeList();
    if (inCodeBlock) out.push('</code></pre>');
    return out.join('');
}

function addMessage(role, text) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    
    // Avatar
    const avatar = document.createElement('div');
    avatar.className = `avatar ${role}`;
    avatar.textContent = role === 'ai' ? '✨' : '👤';
    
    // Content
    const content = document.createElement('div');
    content.className = 'message-content';
    
    if (role === 'ai') {
        content.innerHTML = text ? renderMarkdown(text) : '';
    } else {
        content.textContent = text;
    }
    
    el.appendChild(avatar);
    el.appendChild(content);
    
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return content;
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage('user', text);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Create AI placeholder with loading dots
    const aiContentEl = addMessage('ai', '');
    aiContentEl.innerHTML = '<div class="loading-dots"><span>.</span><span>.</span><span>.</span></div>';

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        let firstChunk = true;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            if (firstChunk) {
                aiContentEl.innerHTML = ""; // Clear loading dots
                firstChunk = false;
            }

            const chunk = decoder.decode(value, { stream: true });
            fullText += chunk;
            
            // Re-render markdown on every chunk (simple but effective for short chats)
            aiContentEl.innerHTML = renderMarkdown(fullText);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

    } catch (err) {
        aiContentEl.innerHTML = `<span style="color: var(--error)">Error: ${err.message}</span>`;
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

// File Upload Logic
fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    showToast('Uploading files...', 'normal');

    for (const file of files) {
        const formData = new FormData();
        let endpoint = '/ingest';
        let paramName = 'file';
        
        if (file.name.toLowerCase().includes('job') || file.name.toLowerCase().includes('description')) {
             endpoint = '/tune';
             paramName = 'job_file';
        }
        
        formData.append(paramName, file);

        try {
            const res = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                uploadedFiles.push(file.name);
                showToast(`${file.name} uploaded successfully!`, 'success');
                
                // Add system message
                const sysMsg = endpoint === '/tune' 
                    ? "I've analyzed the job description. I'm ready to discuss gaps in your resume."
                    : "I've received your resume. How can I help you?";
                
                addMessage('ai', sysMsg);
                
            } else {
                showToast(`Error uploading ${file.name}: ${data.message}`, 'error');
            }
        } catch (err) {
            showToast(`Connection error: ${err.message}`, 'error');
        }
    }
    
    renderAttachments();
    fileInput.value = '';
    
    // Close menu
    closeUploadMenu();
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
