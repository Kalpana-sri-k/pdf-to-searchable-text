document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const status = document.getElementById('statusIndicator');
    const searchInput = document.getElementById('searchInput');
    const resultsBox = document.getElementById('searchResults');
    const previewArea = document.getElementById('previewArea');
    
    // Track current PDF state
    let currentPdfDoc = null;
    let currentFilename = null;
    let currentHighlightText = null;
    
    // Load PDF.js library dynamically
    const pdfjsScript = document.createElement('script');
    pdfjsScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
    document.head.appendChild(pdfjsScript);
    
    pdfjsScript.onload = () => {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    };
    
    // Add highlight styles
    const highlightStyle = document.createElement('style');
    highlightStyle.textContent = `
        .text-layer {
            position: absolute;
            left: 0;
            top: 0;
            right: 0;
            bottom: 0;
            overflow: hidden;
            opacity: 0.2;
            line-height: 1.0;
        }
        .text-layer > span {
            color: transparent;
            position: absolute;
            white-space: pre;
            cursor: text;
            transform-origin: 0% 0%;
        }
        .text-layer .highlight {
            background-color: #ffff00;
            border-radius: 3px;
            margin: -1px;
            padding: 1px;
        }
        .highlight-overlay {
            position: absolute;
            background-color: rgba(255, 255, 0, 0.4);
            border: 2px solid #ffa500;
            border-radius: 3px;
            pointer-events: none;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
    `;
    document.head.appendChild(highlightStyle);
    
    // Function to find and highlight EXACT text phrase on canvas
    async function highlightTextOnPage(page, viewport, searchText, container) {
        // Get text content from PDF page
        const textContent = await page.getTextContent();
        const textItems = textContent.items;
        
        // Build full page text with position mapping
        let fullText = '';
        const charPositions = []; // Maps character index to text item
        
        for (let i = 0; i < textItems.length; i++) {
            const item = textItems[i];
            for (let j = 0; j < item.str.length; j++) {
                charPositions.push({ itemIndex: i, charIndex: j });
            }
            fullText += item.str;
            // Add space between items
            charPositions.push({ itemIndex: i, charIndex: -1 }); // space marker
            fullText += ' ';
        }
        
        const fullTextLower = fullText.toLowerCase();
        
        // Extract the most specific phrase from the answer (first sentence or key phrase)
        const cleanSearch = searchText.replace(/\s+/g, ' ').trim();
        
        // Try to find exact phrase matches, starting with longer phrases
        const sentences = cleanSearch.split(/[.!?]+/).filter(s => s.trim().length > 10);
        let bestMatch = null;
        let bestMatchLength = 0;
        
        // First try full sentences
        for (const sentence of sentences) {
            const phrase = sentence.trim().toLowerCase();
            if (phrase.length > 5) {
                const idx = fullTextLower.indexOf(phrase);
                if (idx !== -1 && phrase.length > bestMatchLength) {
                    bestMatch = { start: idx, length: phrase.length, text: phrase };
                    bestMatchLength = phrase.length;
                }
            }
        }
        
        // If no sentence match, try key phrases (3+ consecutive words)
        if (!bestMatch) {
            const words = cleanSearch.toLowerCase().split(/\s+/).filter(w => w.length > 2);
            for (let len = Math.min(words.length, 6); len >= 3; len--) {
                for (let start = 0; start <= words.length - len; start++) {
                    const phrase = words.slice(start, start + len).join(' ');
                    const idx = fullTextLower.indexOf(phrase);
                    if (idx !== -1) {
                        bestMatch = { start: idx, length: phrase.length, text: phrase };
                        break;
                    }
                }
                if (bestMatch) break;
            }
        }
        
        // If still no match, try individual important words (nouns, verbs - longer words)
        if (!bestMatch) {
            const words = cleanSearch.toLowerCase().split(/\s+/)
                .filter(w => w.length > 4)
                .sort((a, b) => b.length - a.length); // Longer words first
            
            for (const word of words.slice(0, 3)) { // Try top 3 longest words
                const idx = fullTextLower.indexOf(word);
                if (idx !== -1) {
                    bestMatch = { start: idx, length: word.length, text: word };
                    break;
                }
            }
        }
        
        if (!bestMatch) {
            console.log('No matching text found on page for:', searchText.substring(0, 50));
            return 0;
        }
        
        console.log(`Found match: "${bestMatch.text}" at position ${bestMatch.start}`);
        
        // Find which text items contain the match
        const matchStart = bestMatch.start;
        const matchEnd = matchStart + bestMatch.length;
        
        // Collect all text items that are part of the match
        const itemsToHighlight = new Set();
        for (let i = matchStart; i < matchEnd && i < charPositions.length; i++) {
            if (charPositions[i].charIndex !== -1) { // Skip space markers
                itemsToHighlight.add(charPositions[i].itemIndex);
            }
        }
        
        // Create highlights for matched items
        let highlightsAdded = 0;
        let firstHighlight = null;
        
        for (const itemIndex of itemsToHighlight) {
            const item = textItems[itemIndex];
            if (!item.str.trim()) continue;
            
            const tx = item.transform[4];
            const ty = item.transform[5];
            const fontSize = Math.sqrt(item.transform[0] * item.transform[0] + item.transform[1] * item.transform[1]);
            
            const x = tx * viewport.scale;
            const y = (viewport.height / viewport.scale - ty) * viewport.scale - fontSize * viewport.scale;
            const width = item.width * viewport.scale;
            const height = fontSize * viewport.scale * 1.2;
            
            // Create highlight overlay
            const highlight = document.createElement('div');
            highlight.className = 'highlight-overlay';
            highlight.style.left = `${x}px`;
            highlight.style.top = `${y}px`;
            highlight.style.width = `${width}px`;
            highlight.style.height = `${height}px`;
            container.appendChild(highlight);
            highlightsAdded++;
            
            if (!firstHighlight) {
                firstHighlight = highlight;
            }
        }
        
        // Scroll to first highlight
        if (firstHighlight) {
            setTimeout(() => {
                firstHighlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
        }
        
        console.log(`Highlighted ${highlightsAdded} text items for exact phrase: "${bestMatch.text}"`);
        return highlightsAdded;
    }
    
    // Function to load PDF at specific page using PDF.js with highlighting
    async function loadPdfPage(filename, pageNum, highlightText = null) {
        previewArea.style.display = 'block';
        
        // Always rebuild the container to ensure clean state
        previewArea.innerHTML = `
            <div id="pdfContainer" style="width:100%; height:100%; overflow:auto; background:#525659; position:relative;">
                <div id="pdfPageWrapper" style="position:relative; margin:auto;">
                    <canvas id="pdfCanvas"></canvas>
                </div>
            </div>
            <div id="pdfControls" style="position:absolute; bottom:10px; left:50%; transform:translateX(-50%); background:rgba(0,0,0,0.7); padding:8px 16px; border-radius:8px; color:white; font-size:14px; z-index:10;">
                Page <span id="currentPage">1</span> / <span id="totalPages">?</span>
            </div>
        `;
        
        // Load new PDF if filename changed
        if (currentFilename !== filename) {
            currentPdfDoc = null;
            currentFilename = filename;
        }
        
        const canvas = document.getElementById('pdfCanvas');
        const ctx = canvas.getContext('2d');
        const pageWrapper = document.getElementById('pdfPageWrapper');
        
        try {
            // Load PDF document if not already loaded
            if (!currentPdfDoc) {
                currentPdfDoc = await pdfjsLib.getDocument(`/pdf_files/${filename}`).promise;
                document.getElementById('totalPages').textContent = currentPdfDoc.numPages;
            }
            
            // Ensure page number is valid
            const validPageNum = Math.max(1, Math.min(pageNum, currentPdfDoc.numPages));
            
            // Render the specific page
            const page = await currentPdfDoc.getPage(validPageNum);
            const scale = 1.5;
            const viewport = page.getViewport({ scale });
            
            canvas.height = viewport.height;
            canvas.width = viewport.width;
            pageWrapper.style.width = `${viewport.width}px`;
            pageWrapper.style.height = `${viewport.height}px`;
            
            await page.render({
                canvasContext: ctx,
                viewport: viewport
            }).promise;
            
            document.getElementById('currentPage').textContent = validPageNum;
            
            // Highlight text if provided
            if (highlightText) {
                await highlightTextOnPage(page, viewport, highlightText, pageWrapper);
            }
            
            // Scroll to top of container
            document.getElementById('pdfContainer').scrollTop = 0;
            
        } catch (error) {
            console.error('PDF load error:', error);
            previewArea.innerHTML = `<p style="color:#ef4444; padding:20px; text-align:center;">Error loading PDF. <a href="/pdf_files/${filename}#page=${pageNum}" target="_blank" style="color:#60a5fa;">Open in new tab</a></p>`;
        }
    }

    fileInput.onchange = () => {
        if (fileInput.files.length > 0) {
            status.textContent = `Target: ${fileInput.files[0].name}`;
            processBtn.disabled = false;
            processBtn.style.background = "#4f46e5";
        }
    };

    processBtn.onclick = async () => {
        processBtn.disabled = true;
        status.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/upload_pdf', { method: 'POST', body: formData });
            const data = await response.json();
            
            // Check if there was an error
            if (data.error) {
                status.textContent = "❌ Error: " + data.error;
                processBtn.disabled = false;
                return;
            }
            
            document.getElementById('stat-pages').textContent = data.stats.pages;
            document.getElementById('stat-images').textContent = data.stats.images;
            document.getElementById('stat-chars').textContent = data.stats.characters.toLocaleString();
            
            status.textContent = "✅ Knowledge Base Updated";
        } catch (e) {
            status.textContent = "❌ Error in extraction";
            processBtn.disabled = false;
        }
    };

    searchInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value;
            resultsBox.innerHTML = '<p style="text-align:center;">Searching vectors...</p>';
            
            const formData = new FormData();
            formData.append('query', query);
            
            const resp = await fetch('/search/semantic', { method: 'POST', body: formData });
            const data = await resp.json();
            render(data.results);
        }
    });

    function render(results) {
        resultsBox.innerHTML = '';
        if (!results.length) {
            resultsBox.innerHTML = '<p style="text-align:center; color:#94a3b8; padding:20px;">No matching content found. Try different keywords or a related term.</p>';
            return;
        }

        results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card';
            const score = item.similarity_score ? `• Match: ${Math.round(item.similarity_score * 100)}%` : '';
            const confidence = item.confidence ? ` • Confidence: ${Math.round(item.confidence * 100)}%` : '';
            card.innerHTML = `
                <div class="meta">PAGE ${item.section_id} • ${item.subheading} ${score}${confidence}</div>
                <div class="snippet">${item.chunk_text}</div>
            `;
            card.onclick = () => {
                // Pass the chunk text for highlighting on the PDF
                loadPdfPage(item.subheading, item.section_id, item.chunk_text);
            };
            resultsBox.appendChild(card);
        });
    }
});
