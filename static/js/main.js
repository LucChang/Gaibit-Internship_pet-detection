// 全域 JavaScript 狀態與控制器
let activeRois = [];
let eventLogs = [];
let isDrawing = false;
let startX = 0, startY = 0;
let currentX = 0, currentY = 0;

// HTML 元素 DOM 參考
const canvas = document.getElementById('roiCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const videoStream = document.getElementById('videoStream');
const videoWrapper = document.getElementById('videoWrapper');
const folderSelect = document.getElementById('folderSelect');

const inputRoiCategory = document.getElementById('inputRoiCategory');
const customCategoryGroup = document.getElementById('customCategoryGroup');
const inputCustomCategory = document.getElementById('inputCustomCategory');
const inputRoiColor = document.getElementById('inputRoiColor');
const roisContainer = document.getElementById('roisContainer');

const fpsBadge = document.getElementById('fpsBadge');
const statTotalEvents = document.getElementById('statTotalEvents');
const statActiveContacts = document.getElementById('statActiveContacts');
const statActiveObjects = document.getElementById('statActiveObjects');
const statRoiCount = document.getElementById('statRoiCount');

const eventsTableBody = document.getElementById('eventsTableBody');
const inputLogSearch = document.getElementById('inputLogSearch');
const selectEventFilter = document.getElementById('selectEventFilter');
const btnClearLogs = document.getElementById('btnClearLogs');

// 頁面載入完成初始化程序
document.addEventListener('DOMContentLoaded', () => {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // 載入監視器選單與既有 ROI 設定
    fetchFolders();
    fetchRois();

    // 啟動定時監控與事件輪詢 (1 秒更新一次)
    setInterval(fetchStatus, 1000);
    setInterval(fetchEvents, 1000);

    // 切換自訂類別輸入框
    if (inputRoiCategory) {
        inputRoiCategory.addEventListener('change', (e) => {
            if (e.target.value === 'CUSTOM') {
                customCategoryGroup.style.display = 'flex';
                inputCustomCategory.focus();
            } else {
                customCategoryGroup.style.display = 'none';
            }
        });
    }

    // Canvas 滑鼠事件掛載 (畫面上拖曳拉框標記)
    if (canvas) {
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseup', handleMouseUp);
    }

    // 表格搜尋與過濾事件
    if (inputLogSearch) inputLogSearch.addEventListener('input', renderEventsTable);
    if (selectEventFilter) selectEventFilter.addEventListener('change', renderEventsTable);

    // 監視器切換處理
    if (folderSelect) {
        folderSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                selectCameraFolder(e.target.value);
            }
        });
    }

    // 清除日誌按鈕掛載
    if (btnClearLogs) btnClearLogs.addEventListener('click', clearLogs);
});



// 調整 Canvas 解析度與顯示比例
function resizeCanvas() {
    if (videoWrapper && canvas) {
        canvas.width = videoWrapper.clientWidth;
        canvas.height = videoWrapper.clientHeight;
        redrawCanvas();
    }
}

// 取得目前設定之 ROI 類別名稱
function getSelectedCategoryName() {
    if (inputRoiCategory.value === 'CUSTOM') {
        const val = inputCustomCategory.value.trim();
        return val !== '' ? val : '自訂類別 Custom';
    }
    return inputRoiCategory.value;
}

// 繪製與重繪 Canvas 上的 ROI
function redrawCanvas() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. 繪製既有 ROI 框
    activeRois.forEach(roi => {
        const rx = roi.x * canvas.width;
        const ry = roi.y * canvas.height;
        const rw = roi.w * canvas.width;
        const rh = roi.h * canvas.height;

        ctx.strokeStyle = roi.color || '#00E676';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([]);
        ctx.strokeRect(rx, ry, rw, rh);

        // 標籤背景與文字
        ctx.fillStyle = roi.color || '#00E676';
        ctx.font = '600 12px Outfit, sans-serif';
        const label = `ROI: ${roi.category}`;
        const textWidth = ctx.measureText(label).width;

        ctx.fillRect(rx, Math.max(0, ry - 22), textWidth + 12, 22);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, rx + 6, Math.max(15, ry - 6));
    });

    // 2. 正在拖曳拉框中的預覽矩形
    if (isDrawing) {
        const rx = Math.min(startX, currentX);
        const ry = Math.min(startY, currentY);
        const rw = Math.abs(currentX - startX);
        const rh = Math.abs(currentY - startY);

        ctx.strokeStyle = inputRoiColor.value || '#00E676';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 6]);
        ctx.strokeRect(rx, ry, rw, rh);
    }
}

// Canvas 滑鼠按下
function handleMouseDown(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    currentX = startX;
    currentY = startY;
}

// Canvas 滑鼠移動
function handleMouseMove(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    currentX = e.clientX - rect.left;
    currentY = e.clientY - rect.top;
    redrawCanvas();
}

// Canvas 滑鼠放開 (建立新 ROI)
function handleMouseUp(e) {
    if (!isDrawing) return;
    isDrawing = false;

    const rect = canvas.getBoundingClientRect();
    currentX = e.clientX - rect.left;
    currentY = e.clientY - rect.top;

    const rx = Math.min(startX, currentX);
    const ry = Math.min(startY, currentY);
    const rw = Math.abs(currentX - startX);
    const rh = Math.abs(currentY - startY);

    if (rw > 15 && rh > 15) {
        const normX = rx / canvas.width;
        const normY = ry / canvas.height;
        const normW = rw / canvas.width;
        const normH = rh / canvas.height;

        const categoryName = getSelectedCategoryName();

        const newRoi = {
            id: 'roi_' + Date.now(),
            category: categoryName,
            color: inputRoiColor.value,
            x: parseFloat(normX.toFixed(4)),
            y: parseFloat(normY.toFixed(4)),
            w: parseFloat(normW.toFixed(4)),
            h: parseFloat(normH.toFixed(4))
        };

        activeRois.push(newRoi);
        saveRoisBackend();
    }

    redrawCanvas();
}

// REST API: 取得 ROI 列表
async function fetchRois() {
    try {
        const res = await fetch('/api/rois');
        const data = await res.json();
        if (data.rois) {
            activeRois = data.rois;
            renderRoiPills();
            redrawCanvas();
            statRoiCount.textContent = activeRois.length;
        }
    } catch (err) {
        console.error('Fetch ROIs Error:', err);
    }
}

// 儲存 ROIs 到 Flask 後端
async function saveRoisBackend() {
    try {
        await fetch('/api/rois', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rois: activeRois })
        });
        renderRoiPills();
        statRoiCount.textContent = activeRois.length;
    } catch (err) {
        console.error('Save ROIs Error:', err);
    }
}

// 刪除指定 ROI
async function deleteRoi(roiId) {
    activeRois = activeRois.filter(r => r.id !== roiId);
    try {
        await fetch(`/api/rois/${roiId}`, { method: 'DELETE' });
    } catch (err) {
        console.error('Delete ROI Error:', err);
    }
    renderRoiPills();
    redrawCanvas();
    statRoiCount.textContent = activeRois.length;
}

// 渲染已劃定 ROI 標籤清單與 Bounding Box 尺寸編輯面板
function renderRoiPills() {
    if (!roisContainer) return;
    roisContainer.innerHTML = '';
    if (activeRois.length === 0) {
        roisContainer.innerHTML = '<span class="drawing-hint"><i class="fa-solid fa-circle-exclamation"></i> 尚未劃定任何 ROI 類別區域</span>';
        return;
    }

    activeRois.forEach(roi => {
        const card = document.createElement('div');
        card.className = 'roi-edit-card';
        card.innerHTML = `
            <div class="roi-card-header">
                <div class="roi-card-title">
                    <span class="roi-pill-color" style="background-color: ${roi.color}; color: ${roi.color};"></span>
                    <strong>${roi.category}</strong>
                </div>
                <button class="roi-pill-delete" aria-label="刪除 ${roi.category} ROI" title="刪除 ROI">
                    <i class="fa-solid fa-trash-can" aria-hidden="true"></i>
                </button>
            </div>
            <div class="roi-dim-controls">
                <div class="dim-field" title="縮放矩形框寬度 (Width)">
                    <span class="dim-label">寬 (W):</span>
                    <input type="number" step="0.01" min="0.01" max="1" class="dim-input input-w" value="${roi.w}">
                </div>
                <div class="dim-field" title="縮放矩形框高度 (Height)">
                    <span class="dim-label">高 (H):</span>
                    <input type="number" step="0.01" min="0.01" max="1" class="dim-input input-h" value="${roi.h}">
                </div>
                <div class="dim-field" title="移動矩形框 X 軸座標">
                    <span class="dim-label">X 軸:</span>
                    <input type="number" step="0.01" min="0" max="1" class="dim-input input-x" value="${roi.x}">
                </div>
                <div class="dim-field" title="移動矩形框 Y 軸座標">
                    <span class="dim-label">Y 軸:</span>
                    <input type="number" step="0.01" min="0" max="1" class="dim-input input-y" value="${roi.y}">
                </div>
            </div>
        `;

        const inputW = card.querySelector('.input-w');
        const inputH = card.querySelector('.input-h');
        const inputX = card.querySelector('.input-x');
        const inputY = card.querySelector('.input-y');

        const handleDimChange = () => {
            roi.w = Math.max(0.01, Math.min(1, parseFloat(inputW.value) || 0.1));
            roi.h = Math.max(0.01, Math.min(1, parseFloat(inputH.value) || 0.1));
            roi.x = Math.max(0, Math.min(1, parseFloat(inputX.value) || 0));
            roi.y = Math.max(0, Math.min(1, parseFloat(inputY.value) || 0));

            saveRoisBackend();
            redrawCanvas();
        };

        inputW.addEventListener('input', handleDimChange);
        inputH.addEventListener('input', handleDimChange);
        inputX.addEventListener('input', handleDimChange);
        inputY.addEventListener('input', handleDimChange);

        card.querySelector('.roi-pill-delete').addEventListener('click', () => deleteRoi(roi.id));
        roisContainer.appendChild(card);
    });
}

// 取得監視器列表 API
async function fetchFolders() {
    if (!folderSelect) return;
    try {
        const res = await fetch('/api/folders');
        const data = await res.json();
        folderSelect.innerHTML = '';
        if (data.folders && data.folders.length > 0) {
            data.folders.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f;
                opt.textContent = `📹 監視器頻道: ${f}`;
                if (f === data.current_folder) opt.selected = true;
                folderSelect.appendChild(opt);
            });
        } else {
            folderSelect.innerHTML = '<option value="video_test">📹 監視器頻道: video_test</option>';
        }
    } catch (err) {
        console.error('Fetch Folders Error:', err);
    }
}

// 切換監視器頻道 API
async function selectCameraFolder(folder) {
    try {
        const res = await fetch('/api/select_folder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder: folder })
        });
        const data = await res.json();
        if (data.status === 'success') {
            videoStream.src = '/video_feed?' + Date.now();
            fetchRois();
        }
    } catch (err) {
        console.error('Select Camera Error:', err);
    }
}

// 輪詢取得系統狀態數據 API
async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        const status = await res.json();
        const fpsVal = typeof status.fps === 'number' ? status.fps.toFixed(1) : status.fps;
        fpsBadge.textContent = `FPS: ${fpsVal}`;
        statActiveContacts.textContent = status.active_contacts;
        statActiveObjects.textContent = status.active_objects;
        statTotalEvents.textContent = status.total_events;
    } catch (err) {
        console.error('Fetch Status Error:', err);
    }
}

let lastEventsHash = '';

// 輪詢取得接觸事件日誌 API
async function fetchEvents() {
    try {
        const res = await fetch('/api/events');
        const data = await res.json();
        if (data.events) {
            const currentHash = JSON.stringify(data.events);
            // 只有當事件日誌發生變更時，才重新渲染 DOM
            if (currentHash !== lastEventsHash) {
                lastEventsHash = currentHash;
                eventLogs = data.events;
                renderEventsTable();
            }
        }
    } catch (err) {
        console.error('Fetch Events Error:', err);
    }
}

// 清除事件紀錄 API
async function clearLogs() {
    try {
        await fetch('/api/clear_events', { method: 'POST' });
        eventLogs = [];
        lastEventsHash = '';
        renderEventsTable();
    } catch (err) {
        console.error('Clear Logs Error:', err);
    }
}

// 渲染事件歷史表格
function renderEventsTable() {
    if (!eventsTableBody) return;
    const searchText = inputLogSearch ? inputLogSearch.value.toLowerCase().trim() : '';
    const filterType = selectEventFilter ? selectEventFilter.value : 'ALL';

    const filtered = eventLogs.filter(evt => {
        if (filterType !== 'ALL' && evt.event_type !== filterType) return false;

        if (searchText) {
            const matchObj = evt.object_id.toLowerCase().includes(searchText);
            const matchRoi = evt.roi_category.toLowerCase().includes(searchText);
            return matchObj || matchRoi;
        }

        return true;
    });

    eventsTableBody.innerHTML = '';

    if (filtered.length === 0) {
        eventsTableBody.innerHTML = '<tr><td colspan="6" class="empty-msg"><i class="fa-solid fa-folder-open" aria-hidden="true"></i> 尚無符合條件的觸發事件紀錄</td></tr>';
        return;
    }

    filtered.forEach(evt => {
        const tr = document.createElement('tr');

        const badgeClass = 'badge-contact';
        const badgeLabel = 'CONTACT 接觸';

        let clipHtml = '<span class="text-muted" style="font-size: 0.82rem;">-</span>';
        if (evt.clip_url) {
            clipHtml = `<button class="btn-clip" data-url="${evt.clip_url}" title="播放/下載接觸前後 10 秒影片片段">
                <i class="fa-solid fa-film"></i> 觀看 20s 片段
            </button>`;
        } else if (evt.duration_sec >= 1.0) {
            clipHtml = `<span class="text-muted" style="font-size: 0.82rem;"><i class="fa-solid fa-spinner fa-spin"></i> 剪輯中…</span>`;
        }

        tr.innerHTML = `
            <td><strong class="tabular-num">${evt.time_str}</strong></td>
            <td><span class="text-highlight tabular-num">${evt.object_id}</span></td>
            <td><strong>${evt.roi_category}</strong></td>
            <td><span class="badge ${badgeClass}">${badgeLabel}</span></td>
            <td class="tabular-num">${evt.duration_sec > 0 ? evt.duration_sec + ' 秒' : '-'}</td>
            <td>${clipHtml}</td>
        `;

        const clipBtn = tr.querySelector('.btn-clip');
        if (clipBtn) {
            clipBtn.addEventListener('click', () => {
                const url = clipBtn.getAttribute('data-url');
                openVideoModal(url);
            });
        }

        eventsTableBody.appendChild(tr);
    });
}

// 影片 Modal 彈窗操控功能
const videoModal = document.getElementById('videoModal');
const modalVideoPlayer = document.getElementById('modalVideoPlayer');
const modalDownloadBtn = document.getElementById('modalDownloadBtn');
const btnCloseModal = document.getElementById('btnCloseModal');

function openVideoModal(url) {
    if (!videoModal || !modalVideoPlayer) return;
    modalVideoPlayer.src = url;
    if (modalDownloadBtn) modalDownloadBtn.href = url;
    videoModal.style.display = 'flex';
    videoModal.setAttribute('aria-hidden', 'false');
}

function closeVideoModal() {
    if (!videoModal || !modalVideoPlayer) return;
    modalVideoPlayer.pause();
    modalVideoPlayer.src = '';
    videoModal.style.display = 'none';
    videoModal.setAttribute('aria-hidden', 'true');
}

if (btnCloseModal) {
    btnCloseModal.addEventListener('click', closeVideoModal);
}

if (videoModal) {
    videoModal.addEventListener('click', (e) => {
        if (e.target === videoModal) closeVideoModal();
    });
}

