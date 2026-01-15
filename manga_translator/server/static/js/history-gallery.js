/**
 * 翻译历史相册功能
 * 独立的历史记录管理模块
 */

// ============================================================================
// 状态管理
// ============================================================================

let historyData = []; // 存储历史数据
let selectedHistoryItems = new Set(); // 选中的历史项

// ============================================================================
// 工具函数
// ============================================================================

/**
 * 获取认证 token
 */
function getSessionToken() {
    return localStorage.getItem('session_token');
}

/**
 * 加载需要认证的图片
 */
async function loadAuthenticatedImage(url, container) {
    try {
        const sessionToken = getSessionToken();
        const resp = await fetch(url, {
            headers: { 'X-Session-Token': sessionToken }
        });
        
        if (!resp.ok) {
            container.innerHTML = '<span style="color: #999;">加载失败</span>';
            return;
        }
        
        const blob = await resp.blob();
        const blobUrl = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = blobUrl;
        img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
        img.onload = () => URL.revokeObjectURL(blobUrl);
        img.onerror = () => {
            URL.revokeObjectURL(blobUrl);
            container.innerHTML = '<span style="color: #999;">加载失败</span>';
        };
        container.innerHTML = '';
        container.appendChild(img);
    } catch (e) {
        console.error('Failed to load image:', e);
        container.innerHTML = '<span style="color: #999;">加载失败</span>';
    }
}

// ============================================================================
// 历史数据加载
// ============================================================================

/**
 * 加载翻译历史
 */
async function loadTranslationHistory() {
    try {
        const sessionToken = getSessionToken();
        const resp = await fetch('/api/history', {
            headers: { 'X-Session-Token': sessionToken }
        });
        
        if (!resp.ok) {
            console.log('Failed to load history:', resp.status);
            return;
        }
        
        const data = await resp.json();
        historyData = data.history || [];
        renderHistoryList(historyData);
    } catch (e) {
        console.error('Failed to load translation history:', e);
    }
}

/**
 * 渲染侧边栏历史列表
 */
function renderHistoryList(historyItems) {
    const historyList = document.getElementById('history-list');
    const historyEmpty = document.getElementById('history-empty');
    
    if (!historyList) return;
    
    historyList.innerHTML = '';
    
    if (!historyItems || historyItems.length === 0) {
        if (historyEmpty) historyEmpty.style.display = 'block';
        return;
    }
    
    if (historyEmpty) historyEmpty.style.display = 'none';
    
    // 只显示最近5条
    const displayItems = historyItems.slice(0, 5);
    
    displayItems.forEach(item => {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.style.cssText = 'padding: 6px 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; cursor: pointer; font-size: 12px;';
        
        const timestamp = new Date(item.timestamp).toLocaleString();
        const fileCount = item.file_count || 1;
        
        li.innerHTML = `
            <div style="flex: 1; overflow: hidden;">
                <div style="color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${timestamp}</div>
                <div style="font-size: 10px; color: #888;">${fileCount} 个文件</div>
            </div>
        `;
        
        li.addEventListener('click', () => viewHistoryDetail(item.session_token));
        historyList.appendChild(li);
    });
    
    // 显示"打开相册"按钮
    if (historyItems.length > 0) {
        const galleryBtn = document.createElement('li');
        galleryBtn.style.cssText = 'padding: 8px; text-align: center; cursor: pointer; color: #2196F3; font-size: 12px;';
        galleryBtn.textContent = historyItems.length > 5 ? `📷 查看全部 (${historyItems.length})` : '📷 打开相册';
        galleryBtn.addEventListener('click', openHistoryGallery);
        historyList.appendChild(galleryBtn);
    }
}

// ============================================================================
// 相册弹窗
// ============================================================================

/**
 * 打开历史相册弹窗
 */
function openHistoryGallery() {
    const modal = document.createElement('div');
    modal.id = 'history-gallery-modal';
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.8); z-index: 10000;
        display: flex; flex-direction: column;
    `;
    
    modal.innerHTML = `
        <div style="background: #fff; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #ddd;">
            <h3 style="margin: 0; font-size: 18px;">📷 翻译历史相册</h3>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span id="gallery-selection-info" style="font-size: 13px; color: #666;"></span>
                <button id="gallery-download-selected" class="secondary-btn" style="padding: 6px 12px; display: none;">下载选中</button>
                <button id="gallery-download-all" class="secondary-btn" style="padding: 6px 12px;">下载全部</button>
                <button id="gallery-close" style="background: none; border: none; font-size: 24px; cursor: pointer; padding: 0 5px;">×</button>
            </div>
        </div>
        <div id="gallery-content" style="flex: 1; overflow-y: auto; padding: 20px; background: #f5f5f5;">
            <div id="gallery-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;"></div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 绑定事件
    document.getElementById('gallery-close').addEventListener('click', closeHistoryGallery);
    modal.addEventListener('click', (e) => { if (e.target === modal) closeHistoryGallery(); });
    document.getElementById('gallery-download-all').addEventListener('click', downloadAllHistory);
    document.getElementById('gallery-download-selected').addEventListener('click', downloadSelectedHistory);
    
    // 渲染历史项
    renderGalleryItems();
    selectedHistoryItems.clear();
    updateSelectionInfo();
}

/**
 * 关闭相册弹窗
 */
function closeHistoryGallery() {
    const modal = document.getElementById('history-gallery-modal');
    if (modal) modal.remove();
    selectedHistoryItems.clear();
}

/**
 * 渲染相册内容
 */
async function renderGalleryItems() {
    const grid = document.getElementById('gallery-grid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    if (historyData.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #888; padding: 40px;">暂无翻译历史</div>';
        return;
    }
    
    // 按日期分组
    const groupedByDate = {};
    historyData.forEach(item => {
        const date = new Date(item.timestamp).toLocaleDateString();
        if (!groupedByDate[date]) groupedByDate[date] = [];
        groupedByDate[date].push(item);
    });
    
    // 渲染每个日期组
    for (const [date, items] of Object.entries(groupedByDate)) {
        const dateHeader = document.createElement('div');
        dateHeader.style.cssText = 'grid-column: 1/-1; font-size: 14px; font-weight: bold; color: #333; padding: 10px 0 5px; border-bottom: 1px solid #ddd; margin-bottom: 10px;';
        dateHeader.textContent = date;
        grid.appendChild(dateHeader);
        
        for (const item of items) {
            const card = await createGalleryCard(item);
            grid.appendChild(card);
        }
    }
}

/**
 * 创建相册卡片
 */
async function createGalleryCard(item) {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.dataset.token = item.session_token;
    card.style.cssText = `
        background: #fff; border-radius: 8px; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
    `;
    
    const timestamp = new Date(item.timestamp).toLocaleTimeString();
    const fileCount = item.file_count || 1;
    const files = item.files || (item.metadata && item.metadata.files) || [];
    
    card.innerHTML = `
        <div class="gallery-checkbox" style="position: absolute; top: 8px; left: 8px; z-index: 1;">
            <input type="checkbox" data-token="${item.session_token}" style="width: 18px; height: 18px; cursor: pointer;">
        </div>
        <div class="thumbnail-container" style="height: 150px; background: #eee; display: flex; align-items: center; justify-content: center; overflow: hidden;">
            <span style="color: #999;">加载中...</span>
        </div>
        <div style="padding: 10px;">
            <div style="font-size: 12px; color: #333;">${timestamp}</div>
            <div style="font-size: 11px; color: #888; margin-top: 3px;">${fileCount} 个文件</div>
            <div style="display: flex; gap: 5px; margin-top: 8px;">
                <button class="gallery-view-btn secondary-btn" style="flex: 1; padding: 4px 8px; font-size: 11px;">查看</button>
                <button class="gallery-download-btn secondary-btn" style="flex: 1; padding: 4px 8px; font-size: 11px;">下载</button>
                <button class="gallery-delete-btn secondary-btn" style="padding: 4px 8px; font-size: 11px; color: #e74c3c;">🗑</button>
            </div>
        </div>
    `;
    
    // 悬停效果
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-3px)';
        card.style.boxShadow = '0 4px 15px rgba(0,0,0,0.15)';
    });
    card.addEventListener('mouseleave', () => {
        card.style.transform = '';
        card.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
    });
    
    // 复选框事件
    const checkbox = card.querySelector('input[type="checkbox"]');
    checkbox.addEventListener('change', (e) => {
        e.stopPropagation();
        if (checkbox.checked) {
            selectedHistoryItems.add(item.session_token);
        } else {
            selectedHistoryItems.delete(item.session_token);
        }
        updateSelectionInfo();
    });
    
    // 按钮事件
    card.querySelector('.gallery-view-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        viewHistoryInModal(item.session_token);
    });
    
    card.querySelector('.gallery-download-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        downloadHistoryItem(item.session_token);
    });
    
    card.querySelector('.gallery-delete-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm('确定要删除这条翻译历史吗？')) {
            await deleteHistoryItem(item.session_token);
            card.remove();
        }
    });
    
    // 异步加载缩略图
    if (files.length > 0) {
        const filename = files[0].split('/').pop().split('\\').pop();
        loadAuthenticatedImage(
            `/api/history/${item.session_token}/file/${filename}`,
            card.querySelector('.thumbnail-container')
        );
    } else {
        card.querySelector('.thumbnail-container').innerHTML = '<span style="color: #999;">无预览</span>';
    }
    
    return card;
}

/**
 * 更新选择信息
 */
function updateSelectionInfo() {
    const info = document.getElementById('gallery-selection-info');
    const downloadBtn = document.getElementById('gallery-download-selected');
    
    if (info && downloadBtn) {
        if (selectedHistoryItems.size > 0) {
            info.textContent = `已选择 ${selectedHistoryItems.size} 项`;
            downloadBtn.style.display = 'inline-block';
        } else {
            info.textContent = '';
            downloadBtn.style.display = 'none';
        }
    }
}

// ============================================================================
// 图片查看器
// ============================================================================

/**
 * 在弹窗中查看历史图片
 */
async function viewHistoryInModal(historySessionToken) {
    try {
        const sessionToken = getSessionToken();
        const resp = await fetch(`/api/history/${historySessionToken}`, {
            headers: { 'X-Session-Token': sessionToken }
        });
        
        if (!resp.ok) {
            alert('获取历史详情失败');
            return;
        }
        
        const data = await resp.json();
        const session = data.session;
        
        if (!session || !session.files || session.files.length === 0) {
            alert('没有可显示的图片');
            return;
        }
        
        // 创建图片查看弹窗
        const viewer = document.createElement('div');
        viewer.id = 'history-image-viewer';
        viewer.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.95); z-index: 10001;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        `;
        
        const files = session.files;
        let currentIndex = 0;
        
        viewer.innerHTML = `
            <button id="viewer-close" style="position: absolute; top: 15px; right: 20px; background: none; border: none; color: #fff; font-size: 30px; cursor: pointer;">×</button>
            <button id="viewer-prev" style="position: absolute; left: 20px; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.2); border: none; color: #fff; font-size: 30px; cursor: pointer; padding: 10px 15px; border-radius: 5px;">‹</button>
            <button id="viewer-next" style="position: absolute; right: 20px; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.2); border: none; color: #fff; font-size: 30px; cursor: pointer; padding: 10px 15px; border-radius: 5px;">›</button>
            <img id="viewer-image" style="max-width: 90%; max-height: 80%; object-fit: contain;">
            <div id="viewer-info" style="color: #fff; margin-top: 15px; font-size: 14px;"></div>
            <button id="viewer-download" class="secondary-btn" style="margin-top: 10px; padding: 8px 20px;">下载此图片</button>
        `;
        
        document.body.appendChild(viewer);
        
        async function showImage(index) {
            const filePath = files[index];
            const filename = filePath.split('/').pop().split('\\').pop();
            const imageUrl = `/api/history/${historySessionToken}/file/${filename}`;
            document.getElementById('viewer-info').textContent = `${index + 1} / ${files.length} - ${filename}`;
            
            try {
                const sessionToken = getSessionToken();
                const resp = await fetch(imageUrl, {
                    headers: { 'X-Session-Token': sessionToken }
                });
                if (resp.ok) {
                    const blob = await resp.blob();
                    const blobUrl = URL.createObjectURL(blob);
                    const img = document.getElementById('viewer-image');
                    if (img.dataset.blobUrl) URL.revokeObjectURL(img.dataset.blobUrl);
                    img.dataset.blobUrl = blobUrl;
                    img.src = blobUrl;
                }
            } catch (e) {
                console.error('Failed to load image:', e);
            }
        }
        
        showImage(currentIndex);
        
        document.getElementById('viewer-close').addEventListener('click', () => viewer.remove());
        viewer.addEventListener('click', (e) => { if (e.target === viewer) viewer.remove(); });
        
        document.getElementById('viewer-prev').addEventListener('click', () => {
            currentIndex = (currentIndex - 1 + files.length) % files.length;
            showImage(currentIndex);
        });
        
        document.getElementById('viewer-next').addEventListener('click', () => {
            currentIndex = (currentIndex + 1) % files.length;
            showImage(currentIndex);
        });
        
        document.getElementById('viewer-download').addEventListener('click', async () => {
            const filePath = files[currentIndex];
            const filename = filePath.split('/').pop().split('\\').pop();
            const imageUrl = `/api/history/${historySessionToken}/file/${filename}`;
            
            try {
                const sessionToken = getSessionToken();
                const resp = await fetch(imageUrl, {
                    headers: { 'X-Session-Token': sessionToken }
                });
                if (resp.ok) {
                    const blob = await resp.blob();
                    const blobUrl = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = blobUrl;
                    a.download = filename;
                    a.click();
                    URL.revokeObjectURL(blobUrl);
                }
            } catch (e) {
                console.error('Failed to download:', e);
            }
        });
        
        // 键盘导航
        const keyHandler = (e) => {
            if (e.key === 'Escape') viewer.remove();
            if (e.key === 'ArrowLeft') { currentIndex = (currentIndex - 1 + files.length) % files.length; showImage(currentIndex); }
            if (e.key === 'ArrowRight') { currentIndex = (currentIndex + 1) % files.length; showImage(currentIndex); }
        };
        document.addEventListener('keydown', keyHandler);
        
    } catch (e) {
        console.error('Failed to view history:', e);
        alert('获取历史详情失败');
    }
}

/**
 * 查看历史详情（快捷方式）
 */
async function viewHistoryDetail(sessionToken) {
    await viewHistoryInModal(sessionToken);
}

// ============================================================================
// 下载功能
// ============================================================================

/**
 * 下载单个历史项
 * 如果只有一个文件直接下载图片，多个文件则在浏览器端打包
 */
async function downloadHistoryItem(historyToken) {
    const sessionToken = getSessionToken();
    
    try {
        // 获取会话详情
        const detailResp = await fetch(`/api/history/${historyToken}`, {
            headers: { 'X-Session-Token': sessionToken }
        });
        
        if (!detailResp.ok) {
            alert('获取历史详情失败');
            return;
        }
        
        const detailData = await detailResp.json();
        const session = detailData.session;
        
        if (!session || !session.files || session.files.length === 0) {
            alert('没有可下载的文件');
            return;
        }
        
        // 如果只有一个文件，直接下载图片（使用 URL 参数认证，支持 IDM）
        if (session.files.length === 1) {
            const filename = session.files[0].split('/').pop().split('\\').pop();
            // 使用 URL 参数传递 token，支持 IDM 等下载器
            const fileUrl = `/api/history/${historyToken}/file/${filename}?token=${encodeURIComponent(sessionToken)}`;
            
            // 创建下载链接
            const a = document.createElement('a');
            a.href = fileUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } else {
            // 多个文件，使用 JSZip 打包
            await downloadMultipleAsZip([historyToken], `history_${historyToken.substring(0, 8)}`);
        }
    } catch (e) {
        console.error('Failed to download:', e);
        alert('下载失败');
    }
}

/**
 * 下载选中的历史项
 * 在浏览器端使用 JSZip 打包成一个压缩包
 */
async function downloadSelectedHistory() {
    if (selectedHistoryItems.size === 0) {
        alert('请先选择要下载的历史记录');
        return;
    }
    
    const tokens = Array.from(selectedHistoryItems);
    await downloadMultipleAsZip(tokens, 'history_selected');
}

/**
 * 下载全部历史
 */
/**
 * 下载全部历史
 * 在浏览器端使用 JSZip 打包成一个压缩包
 */
async function downloadAllHistory() {
    if (historyData.length === 0) {
        alert('没有可下载的历史记录');
        return;
    }
    
    const tokens = historyData.map(item => item.session_token);
    await downloadMultipleAsZip(tokens, 'history_all');
}

/**
 * 将多个历史会话打包成一个 ZIP 下载
 * @param {string[]} tokens - 会话 token 列表
 * @param {string} zipName - ZIP 文件名前缀
 */
async function downloadMultipleAsZip(tokens, zipName) {
    // 检查 JSZip 是否可用
    if (typeof JSZip === 'undefined') {
        alert('JSZip 未加载，无法打包下载');
        return;
    }
    
    const sessionToken = getSessionToken();
    const zip = new JSZip();
    let successCount = 0;
    
    // 显示进度
    const progressInfo = document.getElementById('gallery-selection-info');
    if (progressInfo) {
        progressInfo.textContent = `正在打包 0/${tokens.length}...`;
    }
    
    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        
        try {
            // 获取会话详情
            const detailResp = await fetch(`/api/history/${token}`, {
                headers: { 'X-Session-Token': sessionToken }
            });
            
            if (!detailResp.ok) continue;
            
            const detailData = await detailResp.json();
            const session = detailData.session;
            
            if (!session || !session.files || session.files.length === 0) continue;
            
            // 下载每个文件
            for (const filePath of session.files) {
                const filename = filePath.split('/').pop().split('\\').pop();
                const fileUrl = `/api/history/${token}/file/${filename}`;
                
                const fileResp = await fetch(fileUrl, {
                    headers: { 'X-Session-Token': sessionToken }
                });
                
                if (fileResp.ok) {
                    const blob = await fileResp.blob();
                    // 使用 token 前8位作为文件夹名
                    zip.file(`${token.substring(0, 8)}/${filename}`, blob);
                    successCount++;
                }
            }
            
            // 更新进度
            if (progressInfo) {
                progressInfo.textContent = `正在打包 ${i + 1}/${tokens.length}...`;
            }
            
        } catch (e) {
            console.error(`Failed to process session ${token}:`, e);
        }
    }
    
    if (successCount === 0) {
        alert('没有可下载的文件');
        if (progressInfo) progressInfo.textContent = '';
        return;
    }
    
    // 生成并下载 ZIP
    if (progressInfo) {
        progressInfo.textContent = '正在生成压缩包...';
    }
    
    try {
        const content = await zip.generateAsync({ type: 'blob' });
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const url = URL.createObjectURL(content);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${zipName}_${timestamp}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        if (progressInfo) {
            progressInfo.textContent = `已下载 ${successCount} 个文件`;
            setTimeout(() => { progressInfo.textContent = ''; }, 3000);
        }
    } catch (e) {
        console.error('Failed to generate ZIP:', e);
        alert('生成压缩包失败');
        if (progressInfo) progressInfo.textContent = '';
    }
}

// ============================================================================
// 删除功能
// ============================================================================

/**
 * 删除历史项
 */
async function deleteHistoryItem(token) {
    try {
        const sessionToken = getSessionToken();
        const resp = await fetch(`/api/history/${token}`, {
            method: 'DELETE',
            headers: { 'X-Session-Token': sessionToken }
        });
        
        if (!resp.ok) {
            alert('删除失败');
            return false;
        }
        
        // 从本地数据中移除
        historyData = historyData.filter(item => item.session_token !== token);
        selectedHistoryItems.delete(token);
        
        // 刷新侧边栏历史列表
        renderHistoryList(historyData);
        
        return true;
    } catch (e) {
        console.error('Failed to delete history:', e);
        alert('删除失败');
        return false;
    }
}

// ============================================================================
// 初始化
// ============================================================================

// 页面加载完成后绑定刷新按钮
document.addEventListener('DOMContentLoaded', () => {
    const refreshBtn = document.getElementById('refresh-history-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadTranslationHistory);
    }
});
