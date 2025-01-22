const imageUpload = document.getElementById('imageUpload');
const searchForm = document.getElementById('searchForm');
const textSearch = document.getElementById('textSearch');
const statusMessage = document.getElementById('statusMessage');
const results = document.getElementById('results');
const searchCard = document.querySelector('.search-card');
const progressBar = document.querySelector('.progress-bar');

async function handleImageUpload(file) {
    if (file.size > 10 * 1024 * 1024) {
        statusMessage.textContent = 'File size cannot exceed 10MB';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        searchCard.classList.add('loading');
        progressBar.style.width = '30%';

        const response = await fetch('/retrieve', {
            method: 'POST',
            body: formData
        });

        progressBar.style.width = '100%';
        setTimeout(() => {
            searchCard.classList.remove('loading');
            progressBar.style.width = '0';
        }, 500);

        const data = await response.json();
        handleResponse(data);
    } catch (error) {
        searchCard.classList.remove('loading');
        statusMessage.textContent = 'An error occurred during retrieval';
        console.error('Error:', error);
    }
}

async function handleTextSearch(text) {
    try {
        searchCard.classList.add('loading');
        progressBar.style.width = '30%';

        const response = await fetch('/text_retrieve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        progressBar.style.width = '100%';
        setTimeout(() => {
            searchCard.classList.remove('loading');
            progressBar.style.width = '0';
        }, 500);

        const data = await response.json();
        handleResponse(data);
    } catch (error) {
        searchCard.classList.remove('loading');
        statusMessage.textContent = 'An error occurred during retrieval';
        console.error('Error:', error);
    }
}

function handleResponse(data) {
    if (data.error) {
        statusMessage.textContent = data.error;
        results.innerHTML = '';
        return;
    }

    results.innerHTML = '';
    data.results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const img = document.createElement('img');
        img.src = result.url;
        img.alt = 'Search Result';
        img.className = 'result-image';
        img.loading = 'lazy';

        const similarity = document.createElement('div');
        similarity.className = 'result-similarity';
        similarity.textContent = `Similarity: ${result.similarity.toFixed(3)}`;

        card.appendChild(img);
        card.appendChild(similarity);
        results.appendChild(card);
    });

    statusMessage.textContent = `Found ${data.results.length} matching images`;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

const debouncedSearch = debounce((text) => {
    if (text) {
        statusMessage.textContent = 'Searching...';
        handleTextSearch(text);
    } else {
        results.innerHTML = '';
        statusMessage.textContent = '';
    }
}, 500);

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) {
        statusMessage.textContent = 'Searching...';
        handleImageUpload(file);
    }
});

searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = textSearch.value.trim();
    if (text) {
        statusMessage.textContent = 'Searching...';
        handleTextSearch(text);
    }
});

textSearch.addEventListener('input', (e) => {
    debouncedSearch(e.target.value.trim());
});