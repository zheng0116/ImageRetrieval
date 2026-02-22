const imageUpload = document.getElementById('imageUpload');
const searchForm = document.getElementById('searchForm');
const textSearch = document.getElementById('textSearch');
const statusMessage = document.getElementById('statusMessage');
const results = document.getElementById('results');
const searchCard = document.querySelector('.search-card');
const progressBar = document.querySelector('.progress-bar');
const fileName = document.getElementById('fileName');
const resultsMeta = document.getElementById('resultsMeta');
const uploadLabel = document.querySelector('.upload-label');
const tipChips = document.querySelectorAll('.tip-chip');

const MAX_FILE_SIZE = 10 * 1024 * 1024;
let progressTimer = null;
let requestSerial = 0;

function setStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.classList.remove('success', 'error', 'warning', 'info');

    if (message) {
        statusMessage.classList.add(type);
    }
}

function setResultsMeta(message) {
    resultsMeta.textContent = message;
}

function setEmptyState(message) {
    results.innerHTML = '';

    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = message;

    results.appendChild(empty);
}

function startLoading(metaMessage) {
    searchCard.classList.add('loading');
    setResultsMeta(metaMessage);

    clearInterval(progressTimer);

    let progress = 14;
    progressBar.style.width = `${progress}%`;

    progressTimer = setInterval(() => {
        progress = Math.min(progress + Math.random() * 13, 92);
        progressBar.style.width = `${progress.toFixed(1)}%`;
    }, 220);
}

function stopLoading() {
    clearInterval(progressTimer);
    progressBar.style.width = '100%';

    setTimeout(() => {
        searchCard.classList.remove('loading');
        progressBar.style.width = '0';
    }, 280);
}

async function postFormData(url, formData) {
    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.error || `请求失败 (${response.status})`);
    }

    return data;
}

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.error || `请求失败 (${response.status})`);
    }

    return data;
}

function handleResponse(data) {
    if (data.error) {
        setStatus(data.error, 'error');
        setResultsMeta('请求失败');
        setEmptyState('检索未完成，请调整关键词或稍后再试。');
        return;
    }

    const list = Array.isArray(data.results) ? data.results : [];

    results.innerHTML = '';

    if (!list.length) {
        setStatus('没有找到匹配图片', 'warning');
        setResultsMeta('返回 0 条结果');
        setEmptyState('暂时没有匹配项，建议尝试更宽泛的描述或更清晰的参考图。');
        return;
    }

    list.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.animationDelay = `${Math.min(index * 0.06, 0.45)}s`;

        const rank = document.createElement('span');
        rank.className = 'result-rank';
        rank.textContent = `#${index + 1}`;

        const img = document.createElement('img');
        img.src = result.url;
        img.alt = `检索结果 ${index + 1}`;
        img.className = 'result-image';
        img.loading = 'lazy';

        const similarity = document.createElement('div');
        similarity.className = 'result-similarity';

        const value = Number(result.similarity);
        const score = Number.isFinite(value) ? value : 0;
        similarity.textContent = `相似度 ${score.toFixed(3)} (${(score * 100).toFixed(1)}%)`;

        card.appendChild(rank);
        card.appendChild(img);
        card.appendChild(similarity);
        results.appendChild(card);
    });

    setStatus(`已找到 ${list.length} 张匹配图片`, 'success');
    setResultsMeta(`共 ${list.length} 条结果，按相似度排序`);
}

async function handleImageUpload(file) {
    if (!file) {
        return;
    }

    if (!file.type.startsWith('image/')) {
        setStatus('请上传有效的图片文件', 'warning');
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        setStatus('图片大小不能超过 10MB', 'warning');
        return;
    }

    fileName.textContent = `已选择：${file.name}`;
    setStatus('正在进行图搜图检索...', 'info');

    const currentRequest = ++requestSerial;
    const formData = new FormData();
    formData.append('file', file);

    try {
        startLoading('图像检索中...');
        const data = await postFormData('/retrieve', formData);

        if (currentRequest !== requestSerial) {
            return;
        }

        handleResponse(data);
    } catch (error) {
        if (currentRequest !== requestSerial) {
            return;
        }

        setStatus(error.message || '检索过程发生错误', 'error');
        setResultsMeta('请求失败');
        setEmptyState('检索请求失败，请稍后重试。');
        console.error('Error:', error);
    } finally {
        if (currentRequest === requestSerial) {
            stopLoading();
        }
    }
}

async function handleTextSearch(text) {
    const query = text.trim();

    if (!query) {
        setStatus('', 'info');
        setResultsMeta('等待查询...');
        setEmptyState('输入关键词，或拖拽上传一张图片开始检索。');
        return;
    }

    setStatus('正在进行文本检索...', 'info');
    const currentRequest = ++requestSerial;

    try {
        startLoading('文本检索中...');
        const data = await postJSON('/text_retrieve', { text: query });

        if (currentRequest !== requestSerial) {
            return;
        }

        handleResponse(data);
    } catch (error) {
        if (currentRequest !== requestSerial) {
            return;
        }

        setStatus(error.message || '检索过程发生错误', 'error');
        setResultsMeta('请求失败');
        setEmptyState('检索请求失败，请稍后重试。');
        console.error('Error:', error);
    } finally {
        if (currentRequest === requestSerial) {
            stopLoading();
        }
    }
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
    handleTextSearch(text);
}, 450);

imageUpload.addEventListener('change', (event) => {
    const file = event.target.files?.[0];

    if (file) {
        handleImageUpload(file);
    }
});

searchForm.addEventListener('submit', (event) => {
    event.preventDefault();
    handleTextSearch(textSearch.value);
});

textSearch.addEventListener('input', (event) => {
    debouncedSearch(event.target.value);
});

tipChips.forEach((chip) => {
    chip.addEventListener('click', () => {
        const query = chip.textContent.trim();

        if (!query) {
            return;
        }

        textSearch.value = query;
        handleTextSearch(query);
    });
});

['dragenter', 'dragover'].forEach((eventName) => {
    searchCard.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        searchCard.classList.add('drag-active');
        uploadLabel.classList.add('drag-over');
    });
});

['dragleave', 'drop'].forEach((eventName) => {
    searchCard.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        searchCard.classList.remove('drag-active');
        uploadLabel.classList.remove('drag-over');
    });
});

searchCard.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0];

    if (file) {
        handleImageUpload(file);
    }
});

setEmptyState('输入关键词，或拖拽上传一张图片开始检索。');
