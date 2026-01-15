// 通用权限编辑器组件
// 编辑用户组时：加载服务器默认配置作为初始值
// 编辑用户时：加载用户组配置作为初始值
// 支持参数锁定：visible=false(隐藏), readonly=true(只读)
class PermissionEditor {
    constructor(options = {}) {
        this.configOptions = null;      // 下拉框选项
        this.baseConfig = null;         // 基础配置（服务器配置或用户组配置）
        this.currentValues = {};        // 当前编辑的值
        this.parameterConfig = {};      // 参数控制配置 {fullKey: {visible, readonly, default_value}}
        this.parentParameterConfig = {};// 上级参数配置（用于继承判断）
        this.onSave = options.onSave || (() => {});
        this.onCancel = options.onCancel || (() => {});
        this.title = options.title || '编辑权限';
        this.mode = options.mode || 'group';  // 'group' 或 'user'
        this.sessionToken = options.sessionToken || '';
        this.modalId = 'permission-editor-modal';
    }
    
    t(key, fallback) {
        return window.i18n ? window.i18n.t(key, fallback) : (fallback || key);
    }
    
    async loadData() {
        try {
            const [optionsResp, defaultsResp, presetsResp] = await Promise.all([
                fetch('/config/options', { headers: { 'X-Session-Token': this.sessionToken } }),
                fetch('/config/defaults', { headers: { 'X-Session-Token': this.sessionToken } }),
                fetch('/api/admin/presets', { headers: { 'X-Session-Token': this.sessionToken } })
            ]);
            
            if (optionsResp.ok) {
                this.configOptions = await optionsResp.json();
                console.log('[PermissionEditor] Loaded config options:', Object.keys(this.configOptions));
            }
            if (defaultsResp.ok) {
                this.baseConfig = await defaultsResp.json();
                console.log('[PermissionEditor] Loaded server defaults:', this.baseConfig);
            }
            if (presetsResp.ok) {
                const data = await presetsResp.json();
                this.presets = data.presets || [];
                console.log('[PermissionEditor] Loaded presets:', this.presets.length);
            }
            
            if (!this.configOptions) this.configOptions = {};
            if (!this.baseConfig) this.baseConfig = {};
            if (!this.presets) this.presets = [];
        } catch (e) {
            console.error('Failed to load config data:', e);
            this.configOptions = {};
            this.baseConfig = {};
            this.presets = [];
        }
    }
    
    // 获取配置值：优先用当前编辑值，否则用基础配置
    getValue(section, key) {
        const fullKey = `${section}.${key}`;
        if (this.currentValues[fullKey] !== undefined) {
            return this.currentValues[fullKey];
        }
        if (this.baseConfig[section] && this.baseConfig[section][key] !== undefined) {
            return this.baseConfig[section][key];
        }
        return null;
    }
    
    async show(existingConfig = {}, groupConfig = null, parentParamConfig = null, parentTranslatorConfig = null, groupName = null) {
        await this.loadData();
        
        // 保存用户组名称（用于用户模式显示）
        this.currentGroupName = groupName || '默认';
        
        // 保存上级参数配置（用于继承判断）
        this.parentParameterConfig = parentParamConfig || {};
        // 保存上级翻译器配置（用于继承判断）
        this.parentTranslatorConfig = parentTranslatorConfig || {};
        
        // 如果是编辑用户，用用户组配置作为基础
        if (this.mode === 'user' && groupConfig) {
            // 保存用户组原始配置（用于比较，只保存与用户组不同的值）
            this.groupOriginalConfig = JSON.parse(JSON.stringify(groupConfig));
            this.baseConfig = this.mergeConfig(this.baseConfig, groupConfig);
        } else {
            this.groupOriginalConfig = null;
        }
        
        // 合并已有配置
        if (existingConfig && Object.keys(existingConfig).length > 0) {
            // 提取参数控制配置
            if (existingConfig.parameter_config) {
                this.parameterConfig = existingConfig.parameter_config;
            }
            this.baseConfig = this.mergeConfig(this.baseConfig, existingConfig);
        }
        
        this.render();
    }
    
    mergeConfig(base, override) {
        const result = JSON.parse(JSON.stringify(base));
        for (const section in override) {
            if (typeof override[section] === 'object' && !Array.isArray(override[section])) {
                if (!result[section]) result[section] = {};
                Object.assign(result[section], override[section]);
            } else {
                result[section] = override[section];
            }
        }
        return result;
    }

    render() {
        // 移除已存在的模态框
        const existing = document.getElementById(this.modalId);
        if (existing) existing.remove();
        
        const modal = document.createElement('div');
        modal.id = this.modalId;
        modal.className = 'modal-overlay';
        
        // 用户模式：简化界面，权限完全依赖用户组
        if (this.mode === 'user') {
            modal.innerHTML = `
                <div class="modal-container" style="max-width: 500px;">
                    <div class="modal-header">
                        <h2>${this.title}</h2>
                        <button class="modal-close" onclick="window._permEditor.close()">×</button>
                    </div>
                    <div class="modal-body">
                        <div class="form-section">
                            <p style="color:#6b7280;margin-bottom:16px;">
                                用户的权限和配额完全由所属用户组决定。如需修改权限，请编辑对应的用户组配置。
                            </p>
                            <div style="background:#f3f4f6;padding:12px;border-radius:8px;">
                                <p style="margin:0;font-size:14px;"><strong>当前用户组：</strong> ${this.currentGroupName || '默认'}</p>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="window._permEditor.close()">${this.t('web_cancel', '关闭')}</button>
                    </div>
                </div>
            `;
        } else {
            // 用户组模式：完整编辑界面
            modal.innerHTML = `
                <div class="modal-container modal-large">
                    <div class="modal-header">
                        <h2>${this.title}</h2>
                        <button class="modal-close" onclick="window._permEditor.close()">×</button>
                    </div>
                    <div class="modal-body">
                        <div class="modal-tabs">
                            <button class="modal-tab active" data-tab="basic">${this.t('Basic Settings', '基础设置')}</button>
                            <button class="modal-tab" data-tab="cli">${this.t('CLI Options', '输出选项')}</button>
                            <button class="modal-tab" data-tab="advanced">${this.t('Advanced Settings', '高级设置')}</button>
                            <button class="modal-tab" data-tab="render">${this.t('label_renderer', '渲染设置')}</button>
                            <button class="modal-tab" data-tab="quota">${this.t('web_quota_management', '配额限制')}</button>
                            <button class="modal-tab" data-tab="features">${this.t('web_group_permissions', '功能权限')}</button>
                        </div>
                        <div class="modal-tab-content active" id="tab-basic">${this.renderBasicTab()}</div>
                        <div class="modal-tab-content" id="tab-cli">${this.renderCliTab()}</div>
                        <div class="modal-tab-content" id="tab-advanced">${this.renderAdvancedTab()}</div>
                        <div class="modal-tab-content" id="tab-render">${this.renderRenderTab()}</div>
                        <div class="modal-tab-content" id="tab-quota">${this.renderQuotaTab()}</div>
                        <div class="modal-tab-content" id="tab-features">${this.renderFeaturesTab()}</div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="window._permEditor.close()">${this.t('web_cancel', '取消')}</button>
                        <button class="btn btn-primary" onclick="window._permEditor.save()">${this.t('web_save', '保存')}</button>
                    </div>
                </div>
            `;
        }
        document.body.appendChild(modal);
        
        // 绑定标签页切换
        modal.querySelectorAll('.modal-tab').forEach(tab => {
            tab.onclick = (e) => {
                e.preventDefault();
                this.switchTab(tab.dataset.tab);
            };
        });
        
        // 绑定禁用开关事件
        modal.querySelectorAll('.param-disabled-cb').forEach(cb => {
            cb.onchange = (e) => {
                const icon = e.target.nextElementSibling;
                if (icon) {
                    icon.textContent = e.target.checked ? '🚫' : '✓';
                }
            };
        });
        
        // 设置超分模型联动
        this.setupUpscalerDependency();
        
        window._permEditor = this;
    }
    
    switchTab(tabId) {
        document.querySelectorAll(`#${this.modalId} .modal-tab`).forEach(t => t.classList.remove('active'));
        document.querySelectorAll(`#${this.modalId} .modal-tab-content`).forEach(c => c.classList.remove('active'));
        document.querySelector(`#${this.modalId} .modal-tab[data-tab="${tabId}"]`)?.classList.add('active');
        document.getElementById(`tab-${tabId}`)?.classList.add('active');
    }
    
    // 获取选项的翻译文本
    getOptionLabel(key, value) {
        // 翻译器选项
        if (key === 'translator') {
            return this.t(`translator_${value}`, value);
        }
        // 语言选项
        if (key === 'target_lang') {
            return this.t(`lang_${value}`, value);
        }
        // 对齐方式
        if (key === 'alignment') {
            return this.t(`alignment_${value}`, value);
        }
        // 文本方向
        if (key === 'direction') {
            return this.t(`direction_${value}`, value);
        }
        // 排版模式
        if (key === 'layout_mode') {
            return this.t(`layout_mode_${value}`, value);
        }
        // Real-CUGAN 模型
        if (key === 'realcugan_model') {
            const modelKey = value.replace(/-/g, '_').replace(/\./g, '_');
            return this.t(`realcugan_${modelKey}`, value);
        }
        // 超分倍数
        if (key === 'upscale_ratio') {
            if (value === '不使用' || value === 'none') return this.t('upscale_ratio_not_use', '不使用');
            return value;
        }
        // 输出格式
        if (key === 'format') {
            if (value === '不指定') return this.t('format_not_specified', '不指定');
            return value;
        }
        // 字体路径 - 区分用户字体和服务器字体
        if (key === 'font_path') {
            if (value && value.startsWith('user:')) {
                // 用户字体: user:{username}/{filename} -> [我的] filename
                const parts = value.split('/');
                const filename = parts[parts.length - 1];
                return `[我的] ${filename}`;
            }
            return value;
        }
        return value;
    }
    
    // 创建下拉框
    createSelect(section, key, options) {
        const value = this.getValue(section, key);
        const id = `perm-${section}-${key}`;
        console.log(`[PermissionEditor] createSelect: ${section}.${key} = "${value}"`);
        const optionsHtml = (options || []).map(opt => {
            const label = this.getOptionLabel(key, opt);
            const selected = (value === opt || String(value) === String(opt)) ? 'selected' : '';
            return `<option value="${opt}" ${selected}>${label}</option>`;
        }).join('');
        return `<select id="${id}" class="form-control" data-section="${section}" data-key="${key}">${optionsHtml}</select>`;
    }
    
    // 创建输入框
    createInput(section, key, type = 'text') {
        const value = this.getValue(section, key);
        const id = `perm-${section}-${key}`;
        const inputType = type === 'number' ? 'number' : 'text';
        const displayValue = value === null ? '' : value;
        return `<input type="${inputType}" id="${id}" class="form-control" value="${displayValue}" data-section="${section}" data-key="${key}">`;
    }
    
    // 创建复选框
    createCheckbox(section, key) {
        const value = this.getValue(section, key);
        const id = `perm-${section}-${key}`;
        const checked = value === true ? 'checked' : '';
        return `<input type="checkbox" id="${id}" ${checked} data-section="${section}" data-key="${key}">`;
    }
    
    // 获取参数禁用状态
    // 用户模式下：继承用户组的禁用状态，但用户可以覆盖（白名单/黑名单）
    isParamDisabled(section, key) {
        const fullKey = `${section}.${key}`;
        const config = this.parameterConfig[fullKey] || {};
        const parentConfig = this.parentParameterConfig[fullKey] || {};
        
        const parentDisabled = parentConfig.disabled === true;
        
        // 用户模式下，检查是否有用户级别的覆盖
        if (this.mode === 'user') {
            // 如果用户有明确设置，使用用户设置
            if (config.disabled !== undefined) {
                return {
                    disabled: config.disabled === true,
                    parentDisabled: parentDisabled,
                    isOverride: config.disabled !== parentDisabled  // 是否覆盖了上级设置
                };
            }
            // 否则继承上级
            return {
                disabled: parentDisabled,
                parentDisabled: parentDisabled,
                isOverride: false
            };
        }
        
        // 用户组模式
        return {
            disabled: config.disabled === true,
            parentDisabled: false,
            isOverride: false
        };
    }
    
    // 创建禁用开关
    createDisableToggle(section, key) {
        const fullKey = `${section}.${key}`;
        const status = this.isParamDisabled(section, key);
        
        const disabledChecked = status.disabled ? 'checked' : '';
        
        // 用户模式下，如果继承了用户组的禁用，显示特殊样式但允许修改（白名单）
        let extraClass = '';
        let title = '禁用后用户看不到此参数';
        if (this.mode === 'user' && status.parentDisabled) {
            if (status.disabled) {
                title = '用户组已禁用，取消勾选可为此用户开启（白名单）';
                extraClass = ' inherited-disabled';
            } else {
                title = '已为此用户开启（白名单覆盖用户组禁用）';
                extraClass = ' whitelist-override';
            }
        }
        
        return `
            <label class="param-disable-toggle${extraClass}" title="${title}">
                <input type="checkbox" class="param-disabled-cb" data-fullkey="${fullKey}" data-parent-disabled="${status.parentDisabled}" ${disabledChecked}>
                <span class="disable-icon">${status.disabled ? '🚫' : '✓'}</span>
            </label>
        `;
    }
    
    // 创建表单行（带禁用开关）
    createFormRow(label, inputHtml, description = '', section = null, key = null) {
        const disableToggle = (section && key) ? this.createDisableToggle(section, key) : '';
        return `
            <div class="form-row">
                <label class="form-label">${label}</label>
                <div class="form-input-wrapper">
                    <div class="form-input">${inputHtml}</div>
                    ${disableToggle}
                </div>
                ${description ? `<div class="form-desc">${description}</div>` : ''}
            </div>
        `;
    }
    
    // 创建表单行（无禁用开关，用于配额和功能权限）
    createFormRowSimple(label, inputHtml, description = '') {
        return `
            <div class="form-row">
                <label class="form-label">${label}</label>
                <div class="form-input">${inputHtml}</div>
                ${description ? `<div class="form-desc">${description}</div>` : ''}
            </div>
        `;
    }

    // 创建预设选择下拉框
    createPresetSelect() {
        const currentPresetId = this.getValue('_meta', 'default_preset_id') || '';
        const presetOptions = this.presets.map(p => 
            `<option value="${p.id}" ${currentPresetId === p.id ? 'selected' : ''}>${this.escapeHtml(p.name)}</option>`
        ).join('');
        
        return `
            <select id="perm-_meta-default_preset_id" class="form-control" data-section="_meta" data-key="default_preset_id">
                <option value="" ${!currentPresetId ? 'selected' : ''}>${this.mode === 'user' ? '继承用户组设置' : '使用服务器默认'}</option>
                ${presetOptions}
            </select>
        `;
    }
    
    escapeHtml(str) {
        if (!str) return '';
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // 基础设置标签页：API预设、翻译器、OCR、检测器、目标语言
    renderBasicTab() {
        const opts = this.configOptions;
        
        // API密钥预设选择器只在用户组模式下显示，用户模式下预设由用户组管理
        const presetSection = this.mode === 'group' ? `
            <div class="form-section" style="background:linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);border-radius:8px;padding:16px;margin-bottom:16px;">
                <h3 style="color:#92400e;">🔑 ${this.t('API Keys (.env)', 'API密钥预设')}</h3>
                <p style="font-size:12px;color:#a16207;margin-bottom:12px;">选择此用户组默认使用的API密钥预设</p>
                ${this.createFormRowSimple(this.t('web_default_preset', '默认预设'), this.createPresetSelect())}
            </div>
        ` : '';
        
        return `
            ${presetSection}
            <p class="param-ctrl-hint">✓ 启用 | 🚫 禁用（用户不可见）</p>
            <div class="form-section">
                <h3>${this.t('label_translator', '翻译器')}</h3>
                ${this.createFormRow(this.t('label_translator', '翻译器'), this.createSelect('translator', 'translator', opts.translator), '', 'translator', 'translator')}
                ${this.createFormRow(this.t('label_target_lang', '目标语言'), this.createSelect('translator', 'target_lang', opts.target_lang), '', 'translator', 'target_lang')}
                ${this.createFormRow(this.t('label_high_quality_prompt_path', '自定义提示词'), this.createSelect('translator', 'high_quality_prompt_path', opts.high_quality_prompt_path), '', 'translator', 'high_quality_prompt_path')}
                ${this.createFormRow(this.t('label_extract_glossary', '自动提取新术语'), this.createCheckbox('translator', 'extract_glossary'), '自动从翻译中提取新术语并保存到提示词文件', 'translator', 'extract_glossary')}
                ${this.createFormRow(this.t('label_no_text_lang_skip', '不跳过目标语言文本'), this.createCheckbox('translator', 'no_text_lang_skip'), '即使文本已是目标语言也进行翻译', 'translator', 'no_text_lang_skip')}
                ${this.createFormRow(this.t('label_max_requests_per_minute', '每分钟最大请求数'), this.createInput('translator', 'max_requests_per_minute', 'number'), '0表示无限制', 'translator', 'max_requests_per_minute')}
            </div>
            <div class="form-section">
                <h3>${this.t('label_ocr', 'OCR设置')}</h3>
                ${this.createFormRow(this.t('label_ocr', 'OCR模型'), this.createSelect('ocr', 'ocr', opts.ocr), '', 'ocr', 'ocr')}
                ${this.createFormRow(this.t('label_secondary_ocr', '备用OCR'), this.createSelect('ocr', 'secondary_ocr', opts.secondary_ocr), '', 'ocr', 'secondary_ocr')}
                ${this.createFormRow(this.t('label_use_hybrid_ocr', '启用混合OCR'), this.createCheckbox('ocr', 'use_hybrid_ocr'), '', 'ocr', 'use_hybrid_ocr')}
                ${this.createFormRow(this.t('label_use_mocr_merge', '合并 MangaOCR 结果'), this.createCheckbox('ocr', 'use_mocr_merge'), '使用Manga OCR的边界框合并', 'ocr', 'use_mocr_merge')}
                ${this.createFormRow(this.t('label_min_text_length', '最小文本长度'), this.createInput('ocr', 'min_text_length', 'number'), '', 'ocr', 'min_text_length')}
                ${this.createFormRow(this.t('label_ignore_bubble', '忽略非气泡文本'), this.createInput('ocr', 'ignore_bubble', 'number', '0', '1', '0.01'), '忽略非气泡区域文本的阈值(0-1)，0=禁用，0.01-0.3=宽松，0.3-0.7=中等，0.7-1.0=严格', 'ocr', 'ignore_bubble')}
                ${this.createFormRow(this.t('label_prob', '文本区域最低概率 (prob)'), this.createInput('ocr', 'prob', 'number'), '文本区域的最小概率阈值', 'ocr', 'prob')}
                ${this.createFormRow(this.t('label_merge_gamma', '合并-距离容忍度'), this.createInput('ocr', 'merge_gamma', 'number'), '值越高越宽容，默认0.8', 'ocr', 'merge_gamma')}
                ${this.createFormRow(this.t('label_merge_sigma', '合并-离群容忍度'), this.createInput('ocr', 'merge_sigma', 'number'), '值越高越宽容，默认2.5', 'ocr', 'merge_sigma')}
                ${this.createFormRow(this.t('label_merge_edge_ratio_threshold', '合并-边缘距离比例阈值'), this.createInput('ocr', 'merge_edge_ratio_threshold', 'number'), '0表示禁用，默认0.0', 'ocr', 'merge_edge_ratio_threshold')}
            </div>
            <div class="form-section">
                <h3>${this.t('label_detector', '文本检测器')}</h3>
                ${this.createFormRow(this.t('label_detector', '文本检测器'), this.createSelect('detector', 'detector', opts.detector), '', 'detector', 'detector')}
                ${this.createFormRow(this.t('label_detection_size', '检测大小'), this.createInput('detector', 'detection_size', 'number'), '', 'detector', 'detection_size')}
                ${this.createFormRow(this.t('label_text_threshold', '文本阈值'), this.createInput('detector', 'text_threshold', 'number'), '', 'detector', 'text_threshold')}
                ${this.createFormRow(this.t('label_box_threshold', '边界框生成阈值'), this.createInput('detector', 'box_threshold', 'number'), '边界框生成阈值，默认0.5', 'detector', 'box_threshold')}
                ${this.createFormRow(this.t('label_unclip_ratio', 'Unclip比例'), this.createInput('detector', 'unclip_ratio', 'number'), '文本骨架扩展比例，默认2.5', 'detector', 'unclip_ratio')}
                ${this.createFormRow(this.t('label_min_box_area_ratio', '最小检测框面积占比'), this.createInput('detector', 'min_box_area_ratio', 'number'), '相对图片总像素，默认0.0009(0.09%)', 'detector', 'min_box_area_ratio')}
                ${this.createFormRow(this.t('label_det_rotate', '旋转图像进行检测'), this.createCheckbox('detector', 'det_rotate'), '旋转图像进行检测，可能提高检测效果', 'detector', 'det_rotate')}
                ${this.createFormRow(this.t('label_det_auto_rotate', '旋转图像以优先检测垂直文本行'), this.createCheckbox('detector', 'det_auto_rotate'), '自动旋转以优先检测垂直文本', 'detector', 'det_auto_rotate')}
                ${this.createFormRow(this.t('label_det_invert', '反转图像颜色进行检测'), this.createCheckbox('detector', 'det_invert'), '反转图像颜色进行检测', 'detector', 'det_invert')}
                ${this.createFormRow(this.t('label_det_gamma_correct', '应用伽马校正进行检测'), this.createCheckbox('detector', 'det_gamma_correct'), '应用伽马校正，可能提高检测效果', 'detector', 'det_gamma_correct')}
                ${this.createFormRow(this.t('label_use_yolo_obb', '启用YOLO辅助检测'), this.createCheckbox('detector', 'use_yolo_obb'), '启用YOLO旋转边界框辅助检测', 'detector', 'use_yolo_obb')}
                ${this.createFormRow(this.t('label_yolo_obb_conf', 'YOLO置信度阈值'), this.createInput('detector', 'yolo_obb_conf', 'number'), '默认0.4', 'detector', 'yolo_obb_conf')}
                ${this.createFormRow(this.t('label_yolo_obb_iou', 'YOLO交叉比(IoU)'), this.createInput('detector', 'yolo_obb_iou', 'number'), 'NMS的IoU阈值，默认0.6', 'detector', 'yolo_obb_iou')}
                ${this.createFormRow(this.t('label_yolo_obb_overlap_threshold', 'YOLO辅助检测重叠率删除阈值'), this.createInput('detector', 'yolo_obb_overlap_threshold', 'number'), '重叠比例阈值(0.0-1.0)，默认0.1', 'detector', 'yolo_obb_overlap_threshold')}
            </div>
        `;
    }
    
    // CLI/输出选项标签页
    renderCliTab() {
        const opts = this.configOptions;
        const formatOptions = ['不指定', 'png', 'jpg', 'webp'];
        return `
            <p class="param-ctrl-hint">✓ 启用 | 🚫 禁用（用户不可见）</p>
            <div class="form-section">
                <h3>${this.t('Output Settings', '输出设置')}</h3>
                ${this.createFormRow(this.t('label_format', '输出格式'), this.createSelect('cli', 'format', formatOptions), '', 'cli', 'format')}
                ${this.createFormRow(this.t('label_save_quality', '保存质量'), this.createInput('cli', 'save_quality', 'number'), '1-100，仅对JPG/WEBP有效', 'cli', 'save_quality')}
                ${this.createFormRow(this.t('label_overwrite', '覆盖已有文件'), this.createCheckbox('cli', 'overwrite'), '', 'cli', 'overwrite')}
                ${this.createFormRow(this.t('label_skip_no_text', '跳过无文本图片'), this.createCheckbox('cli', 'skip_no_text'), '', 'cli', 'skip_no_text')}
                ${this.createFormRow(this.t('label_save_text', '保存文本'), this.createCheckbox('cli', 'save_text'), '', 'cli', 'save_text')}
            </div>
            <div class="form-section">
                <h3>${this.t('Processing Settings', '处理设置')}</h3>
                ${this.createFormRow(this.t('label_attempts', '重试次数'), this.createInput('cli', 'attempts', 'number'), '', 'cli', 'attempts')}
                ${this.createFormRow(this.t('label_ignore_errors', '忽略错误'), this.createCheckbox('cli', 'ignore_errors'), '', 'cli', 'ignore_errors')}
                ${this.createFormRow(this.t('label_context_size', '上下文大小'), this.createInput('cli', 'context_size', 'number'), '', 'cli', 'context_size')}
                ${this.createFormRow(this.t('label_batch_size', '批量大小'), this.createInput('cli', 'batch_size', 'number'), '', 'cli', 'batch_size')}
            </div>
            <div class="form-section">
                <h3>${this.t('GPU Settings', 'GPU设置')}</h3>
                ${this.createFormRow(this.t('label_use_gpu', '使用GPU'), this.createCheckbox('cli', 'use_gpu'), '', 'cli', 'use_gpu')}
            </div>
        `;
    }
    
    // 高级设置标签页：修复器、放大器、上色器
    renderAdvancedTab() {
        const opts = this.configOptions;
        return `
            <p class="param-ctrl-hint">✓ 启用 | 🚫 禁用（用户不可见）</p>
            <div class="form-section">
                <h3>${this.t('label_inpainter', '修复设置')}</h3>
                ${this.createFormRow(this.t('label_inpainter', '修复模型'), this.createSelect('inpainter', 'inpainter', opts.inpainter), '', 'inpainter', 'inpainter')}
                ${this.createFormRow(this.t('label_inpainting_size', '修复大小'), this.createInput('inpainter', 'inpainting_size', 'number'), '太大会导致OOM', 'inpainter', 'inpainting_size')}
                ${this.createFormRow(this.t('label_inpainting_precision', '修复精度'), this.createSelect('inpainter', 'inpainting_precision', opts.inpainting_precision), '', 'inpainter', 'inpainting_precision')}
                ${this.createFormRow(this.t('label_inpainting_split_ratio', '极端长宽比切割阈值'), this.createInput('inpainter', 'inpainting_split_ratio', 'number'), '宽高比阈值，超过则分块处理，默认3.0', 'inpainter', 'inpainting_split_ratio')}
                ${this.createFormRow(this.t('label_force_use_torch_inpainting', '强制使用PyTorch修复'), this.createCheckbox('inpainter', 'force_use_torch_inpainting'), '不使用ONNX，在ONNX内存问题时有用', 'inpainter', 'force_use_torch_inpainting')}
            </div>
            <div class="form-section">
                <h3>${this.t('label_upscaler', '放大设置')}</h3>
                ${this.createFormRow(this.t('label_upscaler', '超分模型'), this.createSelect('upscale', 'upscaler', opts.upscaler), '', 'upscale', 'upscaler')}
                ${this.createFormRow(this.t('label_upscale_ratio', '超分倍数'), this.createSelect('upscale', 'upscale_ratio', opts.upscale_ratio), '', 'upscale', 'upscale_ratio')}
                ${this.createFormRow(this.t('label_revert_upscaling', '还原超分'), this.createCheckbox('upscale', 'revert_upscaling'), '翻译后缩回原始尺寸', 'upscale', 'revert_upscaling')}
                ${this.createFormRow(this.t('label_tile_size', '分块大小'), this.createInput('upscale', 'tile_size', 'number'), '0=不分割，默认400', 'upscale', 'tile_size')}
            </div>
            <div class="form-section">
                <h3>${this.t('label_colorizer', '上色设置')}</h3>
                ${this.createFormRow(this.t('label_colorizer', '上色模型'), this.createSelect('colorizer', 'colorizer', opts.colorizer), '', 'colorizer', 'colorizer')}
                ${this.createFormRow(this.t('label_colorization_size', '上色大小'), this.createInput('colorizer', 'colorization_size', 'number'), '-1使用完整图像尺寸', 'colorizer', 'colorization_size')}
                ${this.createFormRow(this.t('label_denoise_sigma', '降噪强度'), this.createInput('colorizer', 'denoise_sigma', 'number'), '范围0-255，默认30，-1关闭', 'colorizer', 'denoise_sigma')}
            </div>
            <div class="form-section">
                <h3>全局设置</h3>
                ${this.createFormRow(this.t('label_kernel_size', '卷积核大小'), this.createInput('_global', 'kernel_size', 'number'), '文本擦除区域的卷积核大小，默认3', '_global', 'kernel_size')}
                ${this.createFormRow(this.t('label_mask_dilation_offset', '遮罩扩张偏移'), this.createInput('_global', 'mask_dilation_offset', 'number'), '扩展文本掩码以清除残留像素，默认20', '_global', 'mask_dilation_offset')}
            </div>
        `;
    }
    
    // 渲染设置标签页
    renderRenderTab() {
        const opts = this.configOptions;
        return `
            <p class="param-ctrl-hint">✓ 启用 | 🚫 禁用（用户不可见）</p>
            <div class="form-section">
                <h3>${this.t('label_renderer', '渲染器')}</h3>
                ${this.createFormRow(this.t('label_renderer', '渲染器'), this.createSelect('render', 'renderer', opts.renderer), '', 'render', 'renderer')}
                ${this.createFormRow(this.t('label_alignment', '对齐方式'), this.createSelect('render', 'alignment', opts.alignment), '', 'render', 'alignment')}
                ${this.createFormRow(this.t('label_direction', '文本方向'), this.createSelect('render', 'direction', opts.direction), '', 'render', 'direction')}
                ${this.createFormRow(this.t('label_layout_mode', '排版模式'), this.createSelect('render', 'layout_mode', opts.layout_mode), '', 'render', 'layout_mode')}
            </div>
            <div class="form-section">
                <h3>${this.t('label_font_path', '字体设置')}</h3>
                ${this.createFormRow(this.t('label_font_path', '字体路径'), this.createSelect('render', 'font_path', opts.font_path), '', 'render', 'font_path')}
                ${this.createFormRow(this.t('label_font_size', '字体大小'), this.createInput('render', 'font_size', 'number'), '', 'render', 'font_size')}
                ${this.createFormRow(this.t('label_font_size_offset', '字体大小偏移量'), this.createInput('render', 'font_size_offset', 'number'), '', 'render', 'font_size_offset')}
                ${this.createFormRow(this.t('label_font_size_minimum', '最小字体大小'), this.createInput('render', 'font_size_minimum', 'number'), '', 'render', 'font_size_minimum')}
                ${this.createFormRow(this.t('label_max_font_size', '最大字体大小'), this.createInput('render', 'max_font_size', 'number'), '', 'render', 'max_font_size')}
                ${this.createFormRow(this.t('label_font_scale_ratio', '字体缩放比例'), this.createInput('render', 'font_scale_ratio', 'number'), '', 'render', 'font_scale_ratio')}
                ${this.createFormRow(this.t('label_font_color', '字体颜色'), this.createInput('render', 'font_color', 'text'), '如：FFFFFF或FFFFFF:000000', 'render', 'font_color')}
                ${this.createFormRow(this.t('label_line_spacing', '行间距'), this.createInput('render', 'line_spacing', 'number'), '', 'render', 'line_spacing')}
                ${this.createFormRow(this.t('label_stroke_width', '描边宽度比例'), this.createInput('render', 'stroke_width', 'number'), '相对字体大小，默认0.07', 'render', 'stroke_width')}
            </div>
            <div class="form-section">
                <h3>${this.t('Options', '选项')}</h3>
                ${this.createFormRow(this.t('label_uppercase', '大写'), this.createCheckbox('render', 'uppercase'), '', 'render', 'uppercase')}
                ${this.createFormRow(this.t('label_lowercase', '小写'), this.createCheckbox('render', 'lowercase'), '', 'render', 'lowercase')}
                ${this.createFormRow(this.t('label_disable_font_border', '禁用字体边框'), this.createCheckbox('render', 'disable_font_border'), '', 'render', 'disable_font_border')}
                ${this.createFormRow(this.t('label_disable_auto_wrap', 'AI断句'), this.createCheckbox('render', 'disable_auto_wrap'), '', 'render', 'disable_auto_wrap')}
                ${this.createFormRow(this.t('label_rtl', '从右到左'), this.createCheckbox('render', 'rtl'), '', 'render', 'rtl')}
                ${this.createFormRow(this.t('label_no_hyphenation', '禁用连字符'), this.createCheckbox('render', 'no_hyphenation'), '', 'render', 'no_hyphenation')}
                ${this.createFormRow(this.t('label_auto_rotate_symbols', '竖排内横排'), this.createCheckbox('render', 'auto_rotate_symbols'), '自动旋转垂直文本中的符号', 'render', 'auto_rotate_symbols')}
                ${this.createFormRow(this.t('label_center_text_in_bubble', '垂直居中'), this.createCheckbox('render', 'center_text_in_bubble'), '气泡内文本垂直居中', 'render', 'center_text_in_bubble')}
                ${this.createFormRow(this.t('label_optimize_line_breaks', 'AI断句自动扩大文字'), this.createCheckbox('render', 'optimize_line_breaks'), '自动优化换行以找到最佳字体大小', 'render', 'optimize_line_breaks')}
                ${this.createFormRow(this.t('label_check_br_and_retry', 'AI断句检查'), this.createCheckbox('render', 'check_br_and_retry'), '检查翻译是否包含换行标记并重试', 'render', 'check_br_and_retry')}
                ${this.createFormRow(this.t('label_strict_smart_scaling', 'AI断句自动扩大文字下不扩大文本框'), this.createCheckbox('render', 'strict_smart_scaling'), '智能缩放模式下防止文本框扩展', 'render', 'strict_smart_scaling')}
            </div>
        `;
    }

    // 配额限制标签页（只在用户组模式下使用）
    renderQuotaTab() {
        return `
            <div class="form-section">
                <h3>${this.t('web_daily_quota', '每日配额')}</h3>
                ${this.createFormRowSimple(this.t('web_daily_limit', '每日图片限制'), this.createInput('quota', 'daily_image_limit', 'number'))}
                ${this.createFormRowSimple('每日字符限制', this.createInput('quota', 'daily_char_limit', 'number'))}
            </div>
            <div class="form-section">
                <h3>${this.t('Batch Settings', '批量设置')}</h3>
                ${this.createFormRowSimple('最大并发任务', this.createInput('quota', 'max_concurrent_tasks', 'number'))}
                ${this.createFormRowSimple(this.t('label_batch_size', '批量大小'), this.createInput('quota', 'max_batch_size', 'number'))}
            </div>
            <div class="form-section">
                <h3>${this.t('web_upload_limit', '上传限制')}</h3>
                ${this.createFormRowSimple(this.t('web_max_file_size', '单文件最大(MB)'), this.createInput('quota', 'max_image_size_mb', 'number'))}
                ${this.createFormRowSimple(this.t('web_max_files', '最多文件数'), this.createInput('quota', 'max_images_per_batch', 'number'))}
            </div>
        `;
    }
    
    // 功能权限标签页
    renderFeaturesTab() {
        // 预设选择器只在用户组模式下显示，用户模式下预设由用户组管理
        const presetSection = this.mode === 'group' ? `
            <div class="form-section">
                <h3>🔑 ${this.t('Visible Presets', '可见API预设')}</h3>
                <p style="font-size:12px;color:#6b7280;margin-bottom:12px;">选择此用户组可以看到和使用的API预设</p>
                ${this.renderPresetSelector()}
            </div>
        ` : '';
        
        // 翻译器选择器（用户组模式和用户模式都显示）
        const translatorHint = this.mode === 'user' 
            ? '继承用户组设置，勾选可解锁被禁用的翻译器（白名单），取消勾选可额外禁用（黑名单）'
            : '选择允许使用的翻译器，未选中的翻译器将被禁用';
        const translatorSection = `
            <div class="form-section">
                <h3>🔄 ${this.t('label_translator', '翻译器权限')}</h3>
                <p style="font-size:12px;color:#6b7280;margin-bottom:12px;">${translatorHint}</p>
                ${this.renderTranslatorSelector()}
            </div>
        `;
        
        // 工作流选择器（用户组模式和用户模式都显示）
        const workflowHint = this.mode === 'user'
            ? '继承用户组设置，勾选可解锁被禁用的工作流（白名单），取消勾选可额外禁用（黑名单）'
            : '选择允许使用的工作流模式，未选中的工作流将被禁用';
        const workflowSection = `
            <div class="form-section">
                <h3>📋 ${this.t('Translation Workflow Mode:', '工作流权限')}</h3>
                <p style="font-size:12px;color:#6b7280;margin-bottom:12px;">${workflowHint}</p>
                ${this.renderWorkflowSelector()}
            </div>
        `;
        
        return `
            ${presetSection}
            ${translatorSection}
            ${workflowSection}
            <div class="form-section">
                <h3>${this.t('web_resource_management', '资源管理')}</h3>
                ${this.createFormRowSimple(this.t('web_can_upload_font', '可上传字体'), this.createCheckbox('permissions', 'can_upload_fonts'))}
                ${this.createFormRowSimple('允许删除字体', this.createCheckbox('permissions', 'can_delete_fonts'))}
                ${this.createFormRowSimple(this.t('web_can_upload_prompt', '可上传提示词'), this.createCheckbox('permissions', 'can_upload_prompts'))}
                ${this.createFormRowSimple('允许删除提示词', this.createCheckbox('permissions', 'can_delete_prompts'))}
            </div>
            <div class="form-section">
                <h3>${this.t('web_group_permissions', '功能权限')}</h3>
                ${this.createFormRowSimple('允许批量处理', this.createCheckbox('permissions', 'can_use_batch'))}
                ${this.createFormRowSimple('允许API访问', this.createCheckbox('permissions', 'can_use_api'))}
                ${this.createFormRowSimple('允许导出文本', this.createCheckbox('permissions', 'can_export_text'))}
                ${this.createFormRowSimple(this.t('web_can_view_history', '可查看历史'), this.createCheckbox('permissions', 'can_view_history'))}
                ${this.createFormRowSimple(this.t('web_can_view_logs', '可查看日志'), this.createCheckbox('permissions', 'can_view_logs'))}
            </div>
        `;
    }
    
    // 渲染API预设选择器
    renderPresetSelector() {
        const presets = this.presets || [];
        
        // 获取当前配置的可见预设
        const visiblePresets = this.baseConfig?.visible_presets || [];
        const isAllVisible = visiblePresets.length === 0 || visiblePresets.includes('*');
        
        let html = `
            <div class="preset-selector">
                <div class="form-row" style="margin-bottom:12px;">
                    <label style="display:flex;align-items:center;gap:8px;">
                        <input type="checkbox" id="preset-allow-all" ${isAllVisible ? 'checked' : ''} onchange="window._permEditor.toggleAllPresets(this.checked)">
                        <span>允许所有预设</span>
                    </label>
                </div>
                <div id="preset-list" class="translator-list" style="${isAllVisible ? 'opacity:0.5;pointer-events:none;' : ''}">
        `;
        
        if (presets.length === 0) {
            html += `<p style="color:#6b7280;padding:12px;">暂无API预设，请先在"API密钥管理"中创建</p>`;
        } else {
            for (const p of presets) {
                const checked = isAllVisible || visiblePresets.includes(p.id);
                html += `
                    <label class="translator-item">
                        <input type="checkbox" class="preset-cb" data-preset-id="${p.id}" ${checked ? 'checked' : ''}>
                        <span>${this.escapeHtml(p.name)}</span>
                    </label>
                `;
            }
        }
        
        html += `
                </div>
            </div>
        `;
        
        return html;
    }
    
    // 切换所有预设
    toggleAllPresets(allowAll) {
        const list = document.getElementById('preset-list');
        if (list) {
            list.style.opacity = allowAll ? '0.5' : '1';
            list.style.pointerEvents = allowAll ? 'none' : 'auto';
        }
    }
    
    // 渲染翻译器选择器
    renderTranslatorSelector() {
        const translators = this.configOptions?.translator || [];
        
        // 获取当前配置
        const allowedTranslators = this.baseConfig?.allowed_translators || ['*'];
        const deniedTranslators = this.baseConfig?.denied_translators || [];
        
        // 上级配置（用户模式下）
        const parentAllowed = this.parentTranslatorConfig?.allowed_translators || ['*'];
        const parentDenied = this.parentTranslatorConfig?.denied_translators || [];
        
        const isAllAllowed = allowedTranslators.includes('*');
        
        let html = `
            <div class="translator-selector">
                <div class="form-row" style="margin-bottom:12px;">
                    <label style="display:flex;align-items:center;gap:8px;">
                        <input type="checkbox" id="translator-allow-all" ${isAllAllowed ? 'checked' : ''} onchange="window._permEditor.toggleAllTranslators(this.checked)">
                        <span>允许所有翻译器</span>
                    </label>
                </div>
                <div id="translator-list" class="translator-list" style="${isAllAllowed ? 'opacity:0.5;pointer-events:none;' : ''}">
        `;
        
        for (const t of translators) {
            // 计算翻译器状态
            let isAllowed = isAllAllowed || allowedTranslators.includes(t);
            let isDenied = deniedTranslators.includes(t);
            
            // 用户模式下的继承逻辑
            let inheritedDenied = false;
            let isWhitelist = false;
            if (this.mode === 'user') {
                const parentIsAllAllowed = parentAllowed.includes('*');
                const parentAllowsThis = parentIsAllAllowed || parentAllowed.includes(t);
                inheritedDenied = parentDenied.includes(t) || !parentAllowsThis;
                
                // 如果上级禁用了，但当前允许了，就是白名单
                if (inheritedDenied && isAllowed && !isDenied) {
                    isWhitelist = true;
                }
            }
            
            const checked = isAllowed && !isDenied;
            const label = this.t(`translator_${t}`, t);
            
            let extraClass = '';
            let title = '';
            if (this.mode === 'user' && inheritedDenied) {
                if (!checked) {
                    extraClass = ' inherited-disabled';
                    title = '用户组已禁用，勾选可为此用户开启（白名单）';
                } else {
                    extraClass = ' whitelist-override';
                    title = '已为此用户开启（白名单覆盖用户组禁用）';
                }
            }
            
            html += `
                <label class="translator-item${extraClass}" title="${title}">
                    <input type="checkbox" class="translator-cb" data-translator="${t}" data-parent-denied="${inheritedDenied}" ${checked ? 'checked' : ''}>
                    <span>${this.escapeHtml(label)}</span>
                </label>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
        
        return html;
    }
    
    // 切换所有翻译器
    toggleAllTranslators(allowAll) {
        const list = document.getElementById('translator-list');
        if (list) {
            list.style.opacity = allowAll ? '0.5' : '1';
            list.style.pointerEvents = allowAll ? 'none' : 'auto';
        }
    }
    
    // 渲染工作流选择器
    renderWorkflowSelector() {
        const workflows = [
            { id: 'normal', name: '普通翻译' },
            { id: 'export_trans', name: '导出翻译' },
            { id: 'export_raw', name: '导出原文' },
            { id: 'import_trans', name: '导入翻译并渲染' },
            { id: 'colorize', name: '仅上色' },
            { id: 'upscale', name: '仅超分' },
            { id: 'inpaint', name: '仅修复' }
        ];
        
        // 获取当前配置
        const allowedWorkflows = this.baseConfig?.allowed_workflows || ['*'];
        const deniedWorkflows = this.baseConfig?.denied_workflows || [];
        
        // 上级配置（用户模式下）
        const parentAllowed = this.parentTranslatorConfig?.allowed_workflows || ['*'];
        const parentDenied = this.parentTranslatorConfig?.denied_workflows || [];
        
        const isAllAllowed = allowedWorkflows.includes('*') || allowedWorkflows.length === 0;
        
        // 用户模式下不显示"允许所有"选项
        const showAllowAll = this.mode === 'group';
        
        let html = `
            <div class="workflow-selector">
        `;
        
        if (showAllowAll) {
            html += `
                <div class="form-row" style="margin-bottom:12px;">
                    <label style="display:flex;align-items:center;gap:8px;">
                        <input type="checkbox" id="workflow-allow-all" ${isAllAllowed ? 'checked' : ''} onchange="window._permEditor.toggleAllWorkflows(this.checked)">
                        <span>允许所有工作流</span>
                    </label>
                </div>
            `;
        }
        
        html += `
                <div id="workflow-list" class="translator-list" style="${showAllowAll && isAllAllowed ? 'opacity:0.5;pointer-events:none;' : ''}">
        `;
        
        for (const wf of workflows) {
            // 计算工作流状态
            let isAllowed = isAllAllowed || allowedWorkflows.includes(wf.id);
            let isDenied = deniedWorkflows.includes(wf.id);
            
            // 用户模式下的继承逻辑
            let inheritedDenied = false;
            let isWhitelist = false;
            if (this.mode === 'user') {
                const parentIsAllAllowed = parentAllowed.includes('*') || parentAllowed.length === 0;
                const parentAllowsThis = parentIsAllAllowed || parentAllowed.includes(wf.id);
                inheritedDenied = parentDenied.includes(wf.id) || !parentAllowsThis;
                
                // 如果上级禁用了，但当前允许了，就是白名单
                if (inheritedDenied && isAllowed && !isDenied) {
                    isWhitelist = true;
                }
            }
            
            const checked = isAllowed && !isDenied;
            
            let extraClass = '';
            let title = '';
            if (this.mode === 'user' && inheritedDenied) {
                if (!checked) {
                    extraClass = ' inherited-disabled';
                    title = '用户组已禁用，勾选可为此用户开启（白名单）';
                } else {
                    extraClass = ' whitelist-override';
                    title = '已为此用户开启（白名单覆盖用户组禁用）';
                }
            }
            
            html += `
                <label class="translator-item${extraClass}" title="${title}">
                    <input type="checkbox" class="workflow-cb" data-workflow="${wf.id}" data-parent-denied="${inheritedDenied}" ${checked ? 'checked' : ''}>
                    <span>${this.escapeHtml(wf.name)}</span>
                </label>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
        
        return html;
    }
    
    // 切换所有工作流
    toggleAllWorkflows(allowAll) {
        const list = document.getElementById('workflow-list');
        if (list) {
            list.style.opacity = allowAll ? '0.5' : '1';
            list.style.pointerEvents = allowAll ? 'none' : 'auto';
        }
    }
    
    // 设置超分模型联动
    setupUpscalerDependency() {
        const modal = document.getElementById(this.modalId);
        if (!modal) return;
        
        const upscalerSelect = modal.querySelector('#perm-upscale-upscaler');
        const ratioSelect = modal.querySelector('#perm-upscale-upscale_ratio');
        
        if (!upscalerSelect || !ratioSelect) return;
        
        const updateOptions = () => {
            const upscaler = upscalerSelect.value;
            // 获取当前保存的值
            // 注意：PermissionEditor初始化时，getValue可能从mangajanai_model或realcugan_model取值，
            // 但这里我们简单起见，从ratioSelect当前选中值或baseConfig中恢复
            
            // 我们需要判断当前应该显示什么值。
            // 由于 HTML 已经生成，ratioSelect.value 可能是初始的数字（例如4）。
            // 如果当前是 mangajanai 模式，我们需要把它变成 'x4'。
            
            let currentVal = ratioSelect.getAttribute('data-current-value') || ratioSelect.value;
            
            // 从配置中获取更准确的初始值
            const baseRatio = this.getValue('upscale', 'upscale_ratio');
            const baseRealCugan = this.getValue('upscale', 'realcugan_model');
            const baseMangaJaNai = this.getValue('upscale', 'mangajanai_model');
            
            // 第一次运行时，如果 currentVal 是数字，尝试根据 upscaler 修正它
            if (upscaler === 'mangajanai' && !isNaN(currentVal) && baseMangaJaNai) {
                currentVal = baseMangaJaNai;
            } else if (upscaler === 'realcugan' && !isNaN(currentVal) && baseRealCugan) {
                currentVal = baseRealCugan;
            }
            
            const realcuganModels = this.configOptions.realcugan_model || [];
            const standardRatios = [2, 3, 4];
            const mangajanaiOptions = ['x2', 'x4', 'DAT2 x4'];
            
            let newOptions = [];
            
            if (upscaler === 'realcugan') {
                newOptions = ['不使用', ...realcuganModels];
            } else if (upscaler === 'mangajanai') {
                newOptions = ['不使用', ...mangajanaiOptions];
            } else {
                newOptions = ['不使用', ...standardRatios];
            }
            
            // 重建 options
            ratioSelect.innerHTML = newOptions.map(opt => {
                let value = opt;
                if (opt === '不使用') value = ''; 
                
                // 尝试匹配选中状态
                // 1. 完全匹配
                let selected = (String(currentVal) === String(value)) ? 'selected' : '';
                
                // 2. 如果没匹配上，且当前是 mangajanai，尝试默认值
                if (!selected && upscaler === 'mangajanai' && !value) {
                     // 如果 currentVal 是空的，且 opt 是 '不使用'，选中
                     if (!currentVal) selected = 'selected';
                }
                
                // 3. 智能回退：如果 currentVal 是 '4' (数字)，但现在选项是 'x4'，尝试匹配
                if (!selected && value === 'x' + currentVal) selected = 'selected';
                if (!selected && value === currentVal + 'x') selected = 'selected'; // 有些模型带x后缀

                return `<option value="${value}" ${selected}>${opt}</option>`;
            }).join('');
            
            // 更新 data-current-value 以便下次切换保持
            ratioSelect.setAttribute('data-current-value', ratioSelect.value);
        };
        
        upscalerSelect.addEventListener('change', () => {
             // 切换模型时，重置 current-value 为默认值，避免跨模型残留无效值
             ratioSelect.setAttribute('data-current-value', '');
             updateOptions();
        });
        
        ratioSelect.addEventListener('change', () => {
             ratioSelect.setAttribute('data-current-value', ratioSelect.value);
        });
        
        // 初始化调用
        updateOptions();
    }

    // 收集表单数据
    collectFormData() {
        const data = {};
        const paramConfig = {};
        const allowedParams = [];  // 白名单（用户取消了用户组的禁用）
        const deniedParams = [];   // 黑名单（用户额外禁用的）
        const modal = document.getElementById(this.modalId);
        
        // 收集所有输入值
        modal.querySelectorAll('input[data-section], select[data-section]').forEach(el => {
            const section = el.dataset.section;
            const key = el.dataset.key;
            if (!section || !key) return;
            
            // 跳过禁用开关的checkbox
            if (el.classList.contains('param-disabled-cb')) return;
            
            let value;
            if (el.type === 'checkbox') {
                value = el.checked;
            } else if (el.type === 'number') {
                value = el.value === '' ? null : Number(el.value);
            } else {
                value = el.value;
            }
            
            // 用户模式：只保存与用户组不同的值
            if (this.mode === 'user' && this.groupOriginalConfig) {
                const groupValue = this.groupOriginalConfig[section]?.[key];
                // 比较值是否相同（处理类型转换）
                const isSame = String(value) === String(groupValue) || 
                               (value === null && groupValue === undefined) ||
                               (value === undefined && groupValue === null);
                if (isSame) {
                    return; // 跳过，不保存与用户组相同的值
                }
            }
            
            if (!data[section]) data[section] = {};
            data[section][key] = value;
        });

        // 特殊处理：超分模型参数拆分
        if (data.upscale && data.upscale.upscaler) {
            const upscaler = data.upscale.upscaler;
            const ratioValue = data.upscale.upscale_ratio; // 这里可能是数字，也可能是字符串(模型名)
            
            if (upscaler === 'mangajanai') {
                if (ratioValue && typeof ratioValue === 'string') {
                    // 如果选了具体模式（x2, x4...）
                    data.upscale.mangajanai_model = ratioValue;
                    
                    // 推断倍率
                    if (ratioValue.includes('x2')) {
                        data.upscale.upscale_ratio = 2;
                    } else if (ratioValue.includes('x4')) {
                        data.upscale.upscale_ratio = 4;
                    } else {
                        // 默认 fallback
                        data.upscale.upscale_ratio = 4;
                    }
                    // 清除冲突
                    data.upscale.realcugan_model = null;
                } else if (!ratioValue) {
                    // 不使用
                    data.upscale.upscale_ratio = null;
                    data.upscale.mangajanai_model = null;
                }
            } else if (upscaler === 'realcugan') {
                if (ratioValue && typeof ratioValue === 'string') {
                    data.upscale.realcugan_model = ratioValue;
                    data.upscale.upscale_ratio = null; // realcugan 不需要 ratio
                    data.upscale.mangajanai_model = null;
                } else {
                    data.upscale.upscale_ratio = null;
                    data.upscale.realcugan_model = null;
                }
            } else {
                 // 其他模型，ratio 是数字
                 data.upscale.realcugan_model = null;
                 data.upscale.mangajanai_model = null;
            }
        }
        
        // 收集参数禁用配置
        modal.querySelectorAll('.param-disabled-cb').forEach(el => {
            const fullKey = el.dataset.fullkey;
            if (!fullKey) return;
            
            const parentDisabled = el.dataset.parentDisabled === 'true';
            const isDisabled = el.checked;
            
            if (this.mode === 'user') {
                // 用户模式：处理白名单和黑名单
                if (parentDisabled && !isDisabled) {
                    // 用户组禁用了，但用户取消了禁用 = 白名单
                    allowedParams.push(fullKey);
                } else if (!parentDisabled && isDisabled) {
                    // 用户组没禁用，但用户禁用了 = 黑名单
                    deniedParams.push(fullKey);
                    // 同时记录禁用配置和默认值（统一格式）
                    if (!paramConfig[fullKey]) paramConfig[fullKey] = {};
                    paramConfig[fullKey].disabled = true;
                    const [section, key] = fullKey.split('.');
                    if (data[section] && data[section][key] !== undefined) {
                        paramConfig[fullKey].default_value = data[section][key];
                    }
                }
            } else {
                // 用户组模式：记录禁用的参数
                if (isDisabled) {
                    if (!paramConfig[fullKey]) paramConfig[fullKey] = {};
                    paramConfig[fullKey].disabled = true;
                    
                    // 获取对应的值作为默认值
                    const [section, key] = fullKey.split('.');
                    if (data[section] && data[section][key] !== undefined) {
                        paramConfig[fullKey].default_value = data[section][key];
                    }
                }
            }
        });
        
        // 添加参数禁用配置（用户组和用户模式统一格式）
        if (Object.keys(paramConfig).length > 0) {
            data.parameter_config = paramConfig;
        }
        
        // 用户模式：添加白名单和黑名单（用于覆盖用户组设置）
        if (this.mode === 'user') {
            if (allowedParams.length > 0) {
                data.allowed_parameters = allowedParams;
            }
            if (deniedParams.length > 0) {
                data.denied_parameters = deniedParams;
            }
        }
        
        // 收集翻译器配置
        const translatorConfig = this.collectTranslatorConfig();
        if (translatorConfig) {
            Object.assign(data, translatorConfig);
        }
        
        // 收集工作流配置
        const workflowConfig = this.collectWorkflowConfig();
        if (workflowConfig) {
            Object.assign(data, workflowConfig);
        }
        
        // 收集预设配置
        const presetConfig = this.collectPresetConfig();
        if (presetConfig) {
            Object.assign(data, presetConfig);
        }
        
        return data;
    }
    
    // 收集预设配置
    collectPresetConfig() {
        const modal = document.getElementById(this.modalId);
        const allowAllCb = modal.querySelector('#preset-allow-all');
        
        if (!allowAllCb) return null;
        
        const allowAll = allowAllCb.checked;
        
        if (allowAll) {
            return { visible_presets: [] };  // 空数组表示允许所有
        } else {
            const visiblePresets = [];
            modal.querySelectorAll('.preset-cb').forEach(cb => {
                if (cb.checked) {
                    visiblePresets.push(cb.dataset.presetId);
                }
            });
            return { visible_presets: visiblePresets };
        }
    }
    
    // 收集工作流配置
    collectWorkflowConfig() {
        const modal = document.getElementById(this.modalId);
        const allowAllCb = modal.querySelector('#workflow-allow-all');
        
        if (this.mode === 'group') {
            // 用户组模式：直接设置白名单/黑名单
            if (!allowAllCb) return null;
            
            const allowAll = allowAllCb.checked;
            
            if (allowAll) {
                return {
                    allowed_workflows: ['*'],
                    denied_workflows: []
                };
            } else {
                const allowed = [];
                const denied = [];
                modal.querySelectorAll('.workflow-cb').forEach(cb => {
                    const workflow = cb.dataset.workflow;
                    if (cb.checked) {
                        allowed.push(workflow);
                    } else {
                        denied.push(workflow);
                    }
                });
                return {
                    allowed_workflows: allowed.length > 0 ? allowed : [],
                    denied_workflows: denied
                };
            }
        } else {
            // 用户模式：处理白名单/黑名单覆盖
            const allowedWorkflows = [];  // 用户白名单（解锁用户组禁用的）
            const deniedWorkflows = [];   // 用户黑名单（额外禁用的）
            
            modal.querySelectorAll('.workflow-cb').forEach(cb => {
                const workflow = cb.dataset.workflow;
                const parentDenied = cb.dataset.parentDenied === 'true';
                const isChecked = cb.checked;
                
                if (parentDenied && isChecked) {
                    // 用户组禁用了，但用户允许了 = 白名单
                    allowedWorkflows.push(workflow);
                } else if (!parentDenied && !isChecked) {
                    // 用户组允许了，但用户禁用了 = 黑名单
                    deniedWorkflows.push(workflow);
                }
            });
            
            return {
                allowed_workflows: allowedWorkflows.length > 0 ? allowedWorkflows : ['*'],
                denied_workflows: deniedWorkflows
            };
        }
    }
    
    // 收集翻译器配置
    collectTranslatorConfig() {
        const modal = document.getElementById(this.modalId);
        const allowAllCb = modal.querySelector('#translator-allow-all');
        
        if (!allowAllCb) return null;
        
        const allowAll = allowAllCb.checked;
        
        if (this.mode === 'group') {
            // 用户组模式：直接设置白名单/黑名单
            if (allowAll) {
                return {
                    allowed_translators: ['*'],
                    denied_translators: []
                };
            } else {
                const allowed = [];
                const denied = [];
                modal.querySelectorAll('.translator-cb').forEach(cb => {
                    const translator = cb.dataset.translator;
                    if (cb.checked) {
                        allowed.push(translator);
                    } else {
                        denied.push(translator);
                    }
                });
                return {
                    allowed_translators: allowed.length > 0 ? allowed : [],
                    denied_translators: denied
                };
            }
        } else {
            // 用户模式：处理白名单/黑名单覆盖
            const allowedTranslators = [];  // 用户白名单（解锁用户组禁用的）
            const deniedTranslators = [];   // 用户黑名单（额外禁用的）
            
            modal.querySelectorAll('.translator-cb').forEach(cb => {
                const translator = cb.dataset.translator;
                const parentDenied = cb.dataset.parentDenied === 'true';
                const isChecked = cb.checked;
                
                if (parentDenied && isChecked) {
                    // 用户组禁用了，但用户允许了 = 白名单
                    allowedTranslators.push(translator);
                } else if (!parentDenied && !isChecked) {
                    // 用户组允许了，但用户禁用了 = 黑名单
                    deniedTranslators.push(translator);
                }
            });
            
            return {
                allowed_translators: allowedTranslators.length > 0 ? allowedTranslators : ['*'],
                denied_translators: deniedTranslators
            };
        }
    }
    
    save() {
        const data = this.collectFormData();
        this.onSave(data);
        this.close();
    }
    
    close() {
        const modal = document.getElementById(this.modalId);
        if (modal) modal.remove();
        this.onCancel();
    }
}

window.PermissionEditor = PermissionEditor;

