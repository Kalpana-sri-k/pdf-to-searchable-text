document.addEventListener('DOMContentLoaded', function () {
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const processBtn = document.getElementById('processBtn');
const fileList = document.getElementById('fileList');
const statusIndicator = document.getElementById('statusIndicator');
const progressBar = document.getElementById('progressBar');
const processedStats = document.getElementById('processedStats');
const totalPagesEl = document.getElementById('totalPages');
const totalCharsEl = document.getElementById('totalChars');
const totalImagesEl = document.getElementById('totalImages');
const tocEl = document.getElementById('toc');
const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');

let uploadedFiles = [];
let lastStats = null;

// --- Drag & Drop Handlers ---
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
    dropArea.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); })
);
['dragenter', 'dragover'].forEach(evt => dropArea.addEventListener(evt, () => dropArea.classList.add('active')));
['dragleave', 'drop'].forEach(evt => dropArea.addEventListener(evt, () => dropArea.classList.remove('active')));

dropArea.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

function handleFiles(files) {
    Array.from(files).forEach(file => {
        if (file.type === 'application/pdf' && !uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
            uploadedFiles.push(file);
            addFileToList(file);
        }
    });
    processBtn.disabled = uploadedFiles.length === 0;
}

function addFileToList(file) {
    const d = document.createElement('div');
    d.className = 'file-item';
    d.textContent = file.name;
    fileList.appendChild(d);
}

// --- Process PDFs ---
processBtn.addEventListener('click', async () => {
    if (!uploadedFiles.length) return;
    statusIndicator.textContent = 'Processing...';
    processBtn.disabled = true;
    progressBar.style.width = '0%';

    let allPages = 0, allChars = 0, allImages = 0;
    let tocCombined = [];

    for (let i = 0; i < uploadedFiles.length; i++) {
        const file = uploadedFiles[i];
        const fd = new FormData();
        fd.append('file', file);
        try {
            const resp = await fetch('/upload_pdf', { method: 'POST', body: fd });
            const data = await resp.json();
            const stats = data.stats || {};
            lastStats = stats;

            allPages += stats.pages || 0;
            allChars += stats.characters || 0;
            allImages += stats.images || 0;

            if (Array.isArray(stats.toc)) tocCombined = tocCombined.concat(stats.toc.map(t => t[1]));

            progressBar.style.width = `${Math.round(((i + 1) / uploadedFiles.length) * 100)}%`;
        } catch (err) {
            console.error('Upload error', err);
        }
    }

    totalPagesEl.textContent = allPages;
    totalCharsEl.textContent = allChars;
    totalImagesEl.textContent = allImages;
    tocEl.innerHTML = tocCombined.length ? tocCombined.map(t => `<div>${t}</div>`).join('') : '<div>No TOC found</div>';
    processedStats.style.display = 'block';
    statusIndicator.textContent = 'Completed';
});

// --- Debounce Helper ---
function debounce(fn, wait) {
    let t = null;
    return function (...args) {
        clearTimeout(t);
        t = setTimeout(() => fn.apply(this, args), wait);
    };
}

async function doSearch(query) {
    if (!query || !query.trim()) {
        searchResults.innerHTML = '';
        return;
    }

    const fd = new FormData();
    fd.append('query', query);
    fd.append('top_k', '20');

    try {
        const res = await fetch('/search/semantic', { method: 'POST', body: fd });
        const json = await res.json();
        displayResults(json.results || []);
    } catch (err) {
        console.error('search error', err);
    }
}

const debouncedSearch = debounce(e => doSearch(e.target.value), 250);
searchInput.addEventListener('input', debouncedSearch);

// --- Display Results ---
function displayResults(results) {
    searchResults.innerHTML = '';
    if (!results.length) {
        searchResults.innerHTML = '<div class="result-card"><div class="result-title">No results found</div></div>';
        return;
    }

    const seen = new Set();
    results.forEach(r => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const header = document.createElement('div');
        header.className = 'result-title';
        header.textContent = r.subheading || `Page ${r.section_id || 'N/A'}`;
        card.appendChild(header);

        // Check if the result is an image entry
        if (r.image_url) {
    const image = document.createElement('img');
    image.src = r.image_url;
    image.alt = r.caption || 'Image from PDF';
    image.className = 'result-image';
    card.appendChild(image);

    const content = document.createElement('div');
    content.className = 'result-content';
    content.textContent = r.caption || '';
    card.appendChild(content);
} else {
    const text = r.chunk_text || '';
    if (seen.has(text)) return;
    seen.add(text);

    const content = document.createElement('div');
    content.className = 'result-content';
    content.textContent = text;
    card.appendChild(content);
}

        searchResults.appendChild(card);
    });
}

});