document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const status = document.getElementById('statusIndicator');
    const searchInput = document.getElementById('searchInput');
    const resultsBox = document.getElementById('searchResults');
    const previewArea = document.getElementById('previewArea');
    const pdfViewer = document.getElementById('pdfViewer');

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
            resultsBox.innerHTML = '<p>No relevance found.</p>';
            return;
        }

        results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <div class="meta">PAGE ${item.section_id} • ${item.subheading}</div>
                <div class="snippet">${item.chunk_text}</div>
            `;
            card.onclick = () => {
                previewArea.style.display = 'block';
                pdfViewer.src = `/pdf_files/${item.subheading}#page=${item.section_id}`;
            };
            resultsBox.appendChild(card);
        });
    }
});
