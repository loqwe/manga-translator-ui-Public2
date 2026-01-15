// API密钥管理模块 - 支持预设管理
class EnvVarsModule {
    constructor(app) {
        this.app = app;
        this.presets = [];
        this.groups = [];
        // 完整的API密钥配置列表
        this.envKeyGroups = [
            {
                name: 'OpenAI / ChatGPT',
                i18nKey: 'translator_openai',
                keys: [
                    { key: 'OPENAI_API_KEY', i18n: 'label_OPENAI_API_KEY', type: 'password', placeholder: 'sk-...' },
                    { key: 'OPENAI_API_BASE', i18n: 'label_OPENAI_API_BASE', type: 'text', placeholder: 'https://api.openai.com/v1' },
                    { key: 'OPENAI_MODEL', i18n: 'label_OPENAI_MODEL', type: 'text', placeholder: 'gpt-4o' }
                ]
            },
            {
                name: 'Google Gemini',
                i18nKey: 'translator_gemini',
                keys: [
                    { key: 'GEMINI_API_KEY', i18n: 'label_GEMINI_API_KEY', type: 'password', placeholder: 'AIza...' },
                    { key: 'GEMINI_MODEL', i18n: 'label_GEMINI_MODEL', type: 'text', placeholder: 'gemini-1.5-flash' },
                    { key: 'GEMINI_API_BASE', i18n: 'label_GEMINI_API_BASE', type: 'text', placeholder: '' }
                ]
            },
            {
                name: 'DeepSeek',
                keys: [
                    { key: 'DEEPSEEK_API_KEY', i18n: 'label_DEEPSEEK_API_KEY', type: 'password', placeholder: 'sk-...' },
                    { key: 'DEEPSEEK_API_BASE', i18n: 'label_DEEPSEEK_API_BASE', type: 'text', placeholder: 'https://api.deepseek.com' },
                    { key: 'DEEPSEEK_MODEL', i18n: 'label_DEEPSEEK_MODEL', type: 'text', placeholder: 'deepseek-chat' }
                ]
            },

        ];
        
        // 扁平化的key列表用于加载
        this.envKeys = this.envKeyGroups.flatMap(g => g.keys.map(k => k.key));
    }
    
    t(key, fallback) {
        return window.i18n?.t(key) || fallback;
    }
    
    async load() {
        await Promise.all([
            this.loadEnvVars(),
            this.loadPresets(),
            this.loadGroups()
        ]);
    }
    
    async loadEnvVars() {
        try {
            const resp = await fetch('/admin/env-vars?show_values=true', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (resp.ok) {
                const data = await resp.json();
                const vars = data.vars || data;
                
                this.envKeys.forEach(key => {
                    const input = document.getElementById(`env-${key}`);
                    if (input && vars[key]) {
                        input.value = vars[key];
                    }
                });
            }
        } catch (e) {
            console.error('Failed to load env vars:', e);
        }
    }
    
    async loadPresets() {
        try {
            const resp = await fetch('/api/admin/presets?include_config=true', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (resp.ok) {
                const data = await resp.json();
                this.presets = data.presets || [];
                this.renderPresetsList();
            }
        } catch (e) {
            console.error('Failed to load presets:', e);
        }
    }
    
    async loadGroups() {
        try {
            // 正确的API路径
            const resp = await fetch('/api/admin/groups', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (resp.ok) {
                const data = await resp.json();
                this.groups = data.groups || [];
            }
        } catch (e) {
            console.error('Failed to load groups:', e);
        }
    }
    
    renderPresetsList() {
        const container = document.getElementById('presets-list');
        if (!container) return;
        
        if (this.presets.length === 0) {
            container.innerHTML = '<div style="color:#6b7280;text-align:center;padding:20px;">暂无预设，点击上方按钮创建</div>';
            return;
        }
        
        container.innerHTML = this.presets.map(preset => {
            // 显示配置了哪些API
            const configuredApis = Object.keys(preset.config || {}).filter(k => k.includes('API_KEY') || k.includes('AUTH_KEY') || k.includes('TOKEN'));
            const apiTags = configuredApis.map(k => {
                const name = k.replace('_API_KEY', '').replace('_AUTH_KEY', '').replace('_TOKEN', '');
                return `<span class="badge badge-success" style="font-size:10px;margin-right:4px;">${name}</span>`;
            }).join('');
            
            return `
            <div class="preset-card" style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:start;">
                    <div style="flex:1;">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                            <h4 style="margin:0;font-size:15px;font-weight:600;">${this.escapeHtml(preset.name)}</h4>
                            ${apiTags}
                        </div>
                        <p style="margin:0 0 8px 0;color:#6b7280;font-size:13px;">${this.escapeHtml(preset.description || '无描述')}</p>
                        <div style="font-size:12px;color:#9ca3af;">
                            创建者: ${preset.created_by || '未知'} | 
                            创建时间: ${preset.created_at ? new Date(preset.created_at).toLocaleString() : '未知'}
                        </div>
                        ${preset.visible_to_groups && preset.visible_to_groups.length > 0 ? `
                            <div style="margin-top:8px;">
                                <span style="font-size:12px;color:#6b7280;">可见用户组: </span>
                                ${preset.visible_to_groups.map(g => `<span class="badge badge-info" style="font-size:11px;">${g}</span>`).join(' ')}
                            </div>
                        ` : '<div style="margin-top:8px;font-size:12px;color:#9ca3af;">所有用户组可见</div>'}
                    </div>
                    <div style="display:flex;gap:8px;margin-left:16px;">
                        <button class="btn btn-secondary btn-sm" onclick="envVarsModule.editPreset('${preset.id}')">✏️ 编辑</button>
                        <button class="btn btn-danger btn-sm" onclick="envVarsModule.deletePreset('${preset.id}')">🗑️ 删除</button>
                    </div>
                </div>
            </div>
        `}).join('');
    }
    
    // 生成API密钥输入表单HTML
    generateApiKeyFormHtml(prefix = 'preset', existingConfig = {}) {
        return this.envKeyGroups.map(group => `
            <div class="env-group" style="margin-bottom:16px;padding:16px;background:linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);border-radius:12px;border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <h5 style="margin:0 0 12px 0;font-size:14px;font-weight:600;color:#1e293b;display:flex;align-items:center;gap:8px;">
                    <span style="width:8px;height:8px;background:#3b82f6;border-radius:50%;"></span>
                    ${this.t(group.i18nKey, group.name)}
                </h5>
                <div class="form-grid" style="gap:12px;">
                    ${group.keys.map(k => `
                        <div class="form-group" style="margin-bottom:0;">
                            <label class="form-label" style="font-size:12px;color:#64748b;">${this.t(k.i18n, k.key)}</label>
                            <input type="${k.type}" class="form-input" id="${prefix}-${k.key}" 
                                   value="${this.escapeHtml(existingConfig[k.key] || '')}"
                                   placeholder="${k.placeholder}" 
                                   style="font-size:13px;background:#fff;border:1px solid #cbd5e1;">
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }
    
    // 生成用户组选择器HTML
    generateGroupSelectorHtml(prefix = 'preset', selectedGroups = []) {
        const groupItems = this.groups.map(g => {
            const isSelected = selectedGroups.includes(g.id);
            return `
                <div class="group-select-item" data-group-id="${g.id}" 
                     style="display:flex;align-items:center;padding:10px 12px;background:${isSelected ? '#eff6ff' : '#fff'};border:1px solid ${isSelected ? '#3b82f6' : '#e2e8f0'};border-radius:8px;cursor:pointer;transition:all 0.2s;"
                     onclick="envVarsModule.toggleGroupSelection(this, '${prefix}')">
                    <div style="width:20px;height:20px;border:2px solid ${isSelected ? '#3b82f6' : '#cbd5e1'};border-radius:4px;margin-right:10px;display:flex;align-items:center;justify-content:center;background:${isSelected ? '#3b82f6' : '#fff'};">
                        ${isSelected ? '<span style="color:#fff;font-size:12px;">✓</span>' : ''}
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:13px;font-weight:500;color:#1e293b;">${this.escapeHtml(g.name || g.id)}</div>
                        ${g.description ? `<div style="font-size:11px;color:#94a3b8;margin-top:2px;">${this.escapeHtml(g.description)}</div>` : ''}
                    </div>
                </div>
            `;
        }).join('');
        
        return `
            <div id="${prefix}-groups-container" style="display:grid;gap:8px;max-height:200px;overflow-y:auto;padding:4px;">
                ${groupItems || '<div style="color:#94a3b8;text-align:center;padding:20px;">暂无用户组</div>'}
            </div>
            <input type="hidden" id="${prefix}-selected-groups" value="${selectedGroups.join(',')}">
        `;
    }
    
    // 切换用户组选择
    toggleGroupSelection(element, prefix) {
        const groupId = element.dataset.groupId;
        const hiddenInput = document.getElementById(`${prefix}-selected-groups`);
        let selectedGroups = hiddenInput.value ? hiddenInput.value.split(',').filter(Boolean) : [];
        
        const isSelected = selectedGroups.includes(groupId);
        
        if (isSelected) {
            selectedGroups = selectedGroups.filter(id => id !== groupId);
            element.style.background = '#fff';
            element.style.borderColor = '#e2e8f0';
            element.querySelector('div > div:first-child').style.borderColor = '#cbd5e1';
            element.querySelector('div > div:first-child').style.background = '#fff';
            element.querySelector('div > div:first-child').innerHTML = '';
        } else {
            selectedGroups.push(groupId);
            element.style.background = '#eff6ff';
            element.style.borderColor = '#3b82f6';
            element.querySelector('div > div:first-child').style.borderColor = '#3b82f6';
            element.querySelector('div > div:first-child').style.background = '#3b82f6';
            element.querySelector('div > div:first-child').innerHTML = '<span style="color:#fff;font-size:12px;">✓</span>';
        }
        
        hiddenInput.value = selectedGroups.join(',');
    }
    
    showCreatePresetModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal" style="max-width:720px;max-height:90vh;display:flex;flex-direction:column;background:#fff;border-radius:16px;box-shadow:0 25px 50px -12px rgba(0,0,0,0.25);">
                <div class="modal-header" style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;border-radius:16px 16px 0 0;padding:20px 24px;">
                    <h3 style="margin:0;font-size:18px;font-weight:600;">📦 创建API密钥预设</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()" style="color:#fff;opacity:0.8;">×</button>
                </div>
                <div class="modal-body" style="overflow-y:auto;flex:1;padding:24px;background:#f8fafc;">
                    <!-- 基本信息 -->
                    <div style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin:0 0 16px 0;font-size:14px;color:#374151;display:flex;align-items:center;gap:8px;">
                            <span style="font-size:16px;">📝</span> 基本信息
                        </h4>
                        <div class="form-group" style="margin-bottom:16px;">
                            <label class="form-label" style="font-weight:500;">预设名称 <span style="color:#ef4444;">*</span></label>
                            <input type="text" class="form-input" id="preset-name" placeholder="例如: 生产环境API、测试环境API" style="border-radius:8px;">
                        </div>
                        <div class="form-group" style="margin-bottom:0;">
                            <label class="form-label" style="font-weight:500;">描述</label>
                            <textarea class="form-input" id="preset-description" rows="2" placeholder="预设的用途说明，方便管理" style="border-radius:8px;resize:none;"></textarea>
                        </div>
                    </div>
                    
                    <!-- 可见用户组 -->
                    <div style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin:0 0 8px 0;font-size:14px;color:#374151;display:flex;align-items:center;gap:8px;">
                            <span style="font-size:16px;">👥</span> 可见用户组
                        </h4>
                        <p style="margin:0 0 12px 0;font-size:12px;color:#6b7280;">选择哪些用户组可以使用此预设，不选则所有组可见</p>
                        ${this.generateGroupSelectorHtml('preset', [])}
                    </div>
                    
                    <!-- API密钥配置 -->
                    <div style="background:#fff;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin:0 0 8px 0;font-size:14px;color:#374151;display:flex;align-items:center;gap:8px;">
                            <span style="font-size:16px;">🔑</span> API密钥配置
                        </h4>
                        <p style="margin:0 0 16px 0;font-size:12px;color:#6b7280;">只需填写需要的API密钥，留空的项目将使用服务器默认配置</p>
                        ${this.generateApiKeyFormHtml('preset')}
                    </div>
                </div>
                <div class="modal-footer" style="background:#fff;border-top:1px solid #e5e7eb;padding:16px 24px;border-radius:0 0 16px 16px;">
                    <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()" style="border-radius:8px;">取消</button>
                    <button class="btn btn-primary" onclick="envVarsModule.createPreset()" style="border-radius:8px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);border:none;">✅ 创建预设</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
    
    async createPreset() {
        const name = document.getElementById('preset-name')?.value?.trim();
        const description = document.getElementById('preset-description')?.value?.trim();
        
        if (!name) {
            alert('请输入预设名称');
            return;
        }
        
        // 从隐藏字段获取选中的用户组
        const selectedGroupsStr = document.getElementById('preset-selected-groups')?.value || '';
        const selectedGroups = selectedGroupsStr ? selectedGroupsStr.split(',').filter(Boolean) : [];
        
        // 收集所有API密钥配置
        const config = {};
        this.envKeys.forEach(key => {
            const value = document.getElementById(`preset-${key}`)?.value?.trim();
            if (value) config[key] = value;
        });
        
        if (Object.keys(config).length === 0) {
            alert('请至少配置一个API密钥');
            return;
        }
        
        try {
            const resp = await fetch('/api/admin/presets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-Token': this.app.sessionToken
                },
                body: JSON.stringify({
                    name,
                    description,
                    config,
                    visible_to_groups: selectedGroups.length > 0 ? selectedGroups : null
                })
            });
            
            if (resp.ok) {
                document.querySelector('.modal-overlay')?.remove();
                alert('预设创建成功！');
                await this.loadPresets();
            } else {
                const err = await resp.json();
                alert('创建失败: ' + (err.detail || '未知错误'));
            }
        } catch (e) {
            alert('创建失败: ' + e.message);
        }
    }
    
    async editPreset(presetId) {
        try {
            const resp = await fetch(`/api/admin/presets/${presetId}?decrypt=true`, {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (!resp.ok) {
                alert('获取预设详情失败');
                return;
            }
            
            const data = await resp.json();
            const preset = data.preset;
            const selectedGroups = preset.visible_to_groups || [];
            
            const modal = document.createElement('div');
            modal.className = 'modal-overlay';
            modal.innerHTML = `
                <div class="modal" style="max-width:720px;max-height:90vh;display:flex;flex-direction:column;background:#fff;border-radius:16px;box-shadow:0 25px 50px -12px rgba(0,0,0,0.25);">
                    <div class="modal-header" style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;border-radius:16px 16px 0 0;padding:20px 24px;">
                        <h3 style="margin:0;font-size:18px;font-weight:600;">✏️ 编辑预设: ${this.escapeHtml(preset.name)}</h3>
                        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()" style="color:#fff;opacity:0.8;">×</button>
                    </div>
                    <div class="modal-body" style="overflow-y:auto;flex:1;padding:24px;background:#f8fafc;">
                        <!-- 基本信息 -->
                        <div style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                            <h4 style="margin:0 0 16px 0;font-size:14px;color:#374151;display:flex;align-items:center;gap:8px;">
                                <span style="font-size:16px;">📝</span> 基本信息
                            </h4>
                            <div class="form-group" style="margin-bottom:16px;">
                                <label class="form-label" style="font-weight:500;">预设名称 <span style="color:#ef4444;">*</span></label>
                                <input type="text" class="form-input" id="edit-preset-name" value="${this.escapeHtml(preset.name)}" style="border-radius:8px;">
                            </div>
                            <div class="form-group" style="margin-bottom:0;">
                                <label class="form-label" style="font-weight:500;">描述</label>
                                <textarea class="form-input" id="edit-preset-description" rows="2" style="border-radius:8px;resize:none;">${this.escapeHtml(preset.description || '')}</textarea>
                            </div>
                        </div>
                        
                        <!-- 可见用户组 -->
                        <div style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                            <h4 style="margin:0 0 8px 0;font-size:14px;color:#374151;display:flex;align-items:center;gap:8px;">
                                <span style="font-size:16px;">👥</span> 可见用户组
                            </h4>
                            <p style="margin:0 0 12px 0;font-size:12px;color:#6b7280;">选择哪些用户组可以使用此预设，不选则所有组可见</p>
                            ${this.generateGroupSelectorHtml('edit-preset', selectedGroups)}
                        </div>
                        
                        <!-- API密钥配置 -->
                        <div style="background:#fff;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                            <h4 style="margin:0 0 8px 0;font-size:14px;color:#374151;display:flex;align-items:center;gap:8px;">
                                <span style="font-size:16px;">🔑</span> API密钥配置
                            </h4>
                            <p style="margin:0 0 16px 0;font-size:12px;color:#6b7280;">只需填写需要的API密钥，留空的项目将使用服务器默认配置</p>
                            ${this.generateApiKeyFormHtml('edit-preset', preset.config || {})}
                        </div>
                    </div>
                    <div class="modal-footer" style="background:#fff;border-top:1px solid #e5e7eb;padding:16px 24px;border-radius:0 0 16px 16px;">
                        <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()" style="border-radius:8px;">取消</button>
                        <button class="btn btn-primary" onclick="envVarsModule.updatePreset('${presetId}')" style="border-radius:8px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);border:none;">💾 保存修改</button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        } catch (e) {
            alert('获取预设失败: ' + e.message);
        }
    }
    
    async updatePreset(presetId) {
        const name = document.getElementById('edit-preset-name')?.value?.trim();
        const description = document.getElementById('edit-preset-description')?.value?.trim();
        
        if (!name) {
            alert('请输入预设名称');
            return;
        }
        
        // 从隐藏字段获取选中的用户组
        const selectedGroupsStr = document.getElementById('edit-preset-selected-groups')?.value || '';
        const selectedGroups = selectedGroupsStr ? selectedGroupsStr.split(',').filter(Boolean) : [];
        
        const config = {};
        this.envKeys.forEach(key => {
            const value = document.getElementById(`edit-preset-${key}`)?.value?.trim();
            if (value) config[key] = value;
        });
        
        try {
            const resp = await fetch(`/api/admin/presets/${presetId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-Token': this.app.sessionToken
                },
                body: JSON.stringify({
                    name,
                    description,
                    config,
                    visible_to_groups: selectedGroups.length > 0 ? selectedGroups : null
                })
            });
            
            if (resp.ok) {
                document.querySelector('.modal-overlay')?.remove();
                alert('预设更新成功！');
                await this.loadPresets();
            } else {
                const err = await resp.json();
                alert('更新失败: ' + (err.detail || '未知错误'));
            }
        } catch (e) {
            alert('更新失败: ' + e.message);
        }
    }
    
    async deletePreset(presetId) {
        if (!confirm('确定要删除这个预设吗？此操作不可恢复。')) return;
        
        try {
            const resp = await fetch(`/api/admin/presets/${presetId}`, {
                method: 'DELETE',
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (resp.ok) {
                alert('预设已删除');
                await this.loadPresets();
            } else {
                alert('删除失败');
            }
        } catch (e) {
            alert('删除失败: ' + e.message);
        }
    }
    
    async saveEnvVars() {
        const envVars = {};
        
        this.envKeys.forEach(key => {
            const input = document.getElementById(`env-${key}`);
            if (input && input.value.trim()) {
                envVars[key] = input.value.trim();
            }
        });
        
        try {
            const resp = await fetch('/admin/env-vars', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-Token': this.app.sessionToken
                },
                body: JSON.stringify(envVars)
            });
            
            if (resp.ok) {
                alert('API密钥已保存并立即生效！');
            } else {
                throw new Error('保存失败');
            }
        } catch (e) {
            alert('保存失败: ' + e.message);
        }
    }
    
    escapeHtml(str) {
        if (!str) return '';
        return String(str).replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;');
    }
}

window.EnvVarsModule = EnvVarsModule;
