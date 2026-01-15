# API 配置教程

本文档提供常用在线翻译 API 的申请和配置教程。

---

## 📋 目录

- [模型选择建议](#模型选择建议)
- [通用 API 配置说明](#通用-api-配置说明)
- [硅基流动 API 配置](#硅基流动-api-配置)
- [DeepSeek API 配置](#deepseek-api-配置)
- [Google Gemini API 配置](#google-gemini-api-配置)
- [常见问题](#常见问题)

---

## 模型选择建议

- **高质量翻译器**需要多模态模型（能"看图"的 AI），如 Grok、Gemini、ChatGPT 等，AI 能看到漫画画面，翻译更准确
- 一般来说，参数量越大的模型翻译效果越好

### 如何看参数量

模型名称中通常包含参数量信息，例如：
- `Qwen3-235B` → 2350 亿参数
- `DeepSeek-V3-671B` → 6710 亿参数
- `Llama-3-70B` → 700 亿参数

参数量单位：`B` = Billion（十亿），所以 `235B` = 2350 亿参数

### 多模态模型示例

| 模型 | 平台 | 说明 |
|------|------|------|
| `gpt-5.2` | OpenAI | ChatGPT 最新多模态 |
| `gemini-3-pro-preview` | Google | Gemini 最新多模态 |
| `gemini-2.5-pro` | Google | Gemini 多模态 |
| `grok-4.1` | xAI | Grok 最新多模态 |

### 纯文字模型示例

| 模型 | 平台 | 说明 |
|------|------|------|
| `deepseek-chat` | DeepSeek | 速度快 |
| `deepseek-reasoner` | DeepSeek | 有思考，断句稳定 |
| `Qwen/Qwen3-235B-A22B` | 硅基流动 | 通义千问3，2350亿参数 |

---

## 通用 API 配置说明

### 翻译器类型

程序提供了两类翻译器，它们的区别只是**接口不同**：

#### 普通翻译器（OpenAI / Gemini）
- 使用纯文本 API
- 只发送识别出的文字
- 翻译速度快，消耗少
- 适合简单场景

#### 高质量翻译器（高质量翻译 OpenAI / 高质量翻译 Gemini）
- 使用多模态 API（支持图片）
- 发送图片 + 文字
- AI 可以"看到"图片，理解场景
- 翻译更准确，但消耗较多
- **需要模型支持多模态**（如 GPT-4o、Gemini）

> 💡 **提示**：如果你的模型支持多模态，强烈推荐使用"高质量翻译器"获得最佳效果！

### API 地址填写规范

#### OpenAI 兼容接口

OpenAI 翻译器**几乎支持市面上所有模型**，因为几乎所有的 AI 平台都提供 OpenAI 兼容接口。

- **一般情况**：API 地址以 `/v1` 结尾
  - 例如：`https://api.openai.com/v1`
  - 例如：`https://api.deepseek.com/v1`
  - 支持：DeepSeek、Groq、Together AI、OpenRouter、**硅基流动**、**火山引擎**等
- **例外情况**：某些服务商可能使用其他版本号
  - 例如：火山引擎使用 `/v3` 结尾

> 💡 **提示**：只要你的 API 提供商支持 OpenAI 兼容接口，就可以使用 OpenAI 翻译器！

#### Gemini 接口
- **无需添加版本号**：直接填写基础地址即可
  - 填写：`https://generativelanguage.googleapis.com`
  - 程序会自动添加 `/v1beta`
- **使用 AI Studio 官方 key**：无需填写 API 地址（自动使用默认地址）

---

## 硅基流动 API 配置

硅基流动（SiliconFlow）是国内 AI 平台，提供多种模型，新用户有赠送额度，价格便宜，国内访问速度快。

> 💡 **优势**：新用户注册赠送额度，支持 Qwen3、DeepSeek 等多种模型，国内直连无需科学上网。

### 1. 注册账号

1. 访问 [硅基流动官网](https://cloud.siliconflow.cn/)
2. 点击"注册"，使用手机号注册
3. 完成验证

### 2. 创建 API Key

1. 登录后进入控制台
2. 点击左侧"API 密钥"
3. 点击"新建 API 密钥"
4. 复制生成的 API Key

### 3. 配置到程序中

1. 打开程序
2. 在"基础设置"→"翻译器"中选择"OpenAI"
3. 在"高级设置"中填写：
   - **API Key**：你的硅基流动 API Key
   - **Base URL**：`https://api.siliconflow.cn/v1`
   - **模型**：在 [模型广场](https://cloud.siliconflow.cn/models) 查看所有可用模型

---

## DeepSeek API 配置

DeepSeek 提供高质量、低成本的 AI 翻译服务，非常适合漫画翻译使用。

> ⚠️ **注意**：DeepSeek 不支持多模态，无法使用"高质量翻译器"。为了获得最佳翻译效果，建议使用支持多模态的模型（如 OpenAI GPT-4o、Google Gemini）。

### 1. 注册账号

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 点击"注册"按钮，使用邮箱或手机号注册
3. 完成邮箱验证

### 2. 充值

1. 登录后，点击右上角头像 → "充值"
2. 选择充值金额（建议最低 10 元起）
3. 使用支付宝或微信支付

### 3. 创建 API Key

1. 点击左侧菜单"API Keys"
2. 点击"创建 API Key"按钮
3. 输入名称（如"漫画翻译"）
4. 复制生成的 API Key（格式：`sk-xxxxxxxxxxxxxxxx`）
5. ⚠️ **重要**：立即保存 API Key，关闭窗口后无法再次查看

### 4. 配置到程序中

1. 打开程序
2. 在"基础设置"→"翻译器"中选择"OpenAI"
3. 在"高级设置"中填写：
   - **API Key**：填入你的 DeepSeek API Key（`sk-xxxxxxxxxxxxxxxx`）
   - **Base URL**：填入 `https://api.deepseek.com/v1`
   - **模型**：选择以下两种之一：
     - `deepseek-chat`：不思考，速度快，**但可能导致 AI 断句不生效**
     - `deepseek-reasoner`：有思考，速度慢，**但断句稳定可靠** ⭐ 推荐

> 💡 **提示**：AI 断句功能可以智能拆分长文本，避免气泡溢出。如果需要最佳翻译效果，建议使用 `deepseek-reasoner`。

---

## Google Gemini API 配置

Google Gemini 是 Google 最新的多模态 AI 模型，性能强劲。

> ⚠️ **注意**：Google AI Studio 已全面收费，不再提供免费额度。

### 1. 获取 API Key

1. 访问 [Google AI Studio](https://aistudio.google.com/apikey)
2. 登录 Google 账号
3. 点击"Create API Key"
4. 选择 Google Cloud 项目（或创建新项目）
5. 复制生成的 API Key

### 2. 配置到程序中

1. 打开程序
2. 在"基础设置"→"翻译器"中选择"高质量翻译 Gemini"或"Gemini"
3. 在"高级设置"中填写：
   - **API Key**：你的 Gemini API Key
   - **Base URL**：无需填写（自动使用默认地址）
   - **模型**：
     - `gemini-2.5-pro`：断句稳定，质量最高 ⭐ 推荐
     - `gemini-2.5-flash`：速度快，价格便宜

---

## 常见问题

### Q1：哪个 API 最推荐？

**回答**：
- **性价比最高**：DeepSeek（国内用户推荐）
- **质量最高**：OpenAI GPT-4o / Google Gemini

### Q2：API Key 泄露怎么办？

**回答**：
1. 立即到对应平台删除泄露的 API Key
2. 创建新的 API Key
3. 检查账户余额是否异常

### Q3：提示"API Key 无效"怎么办？

**回答**：
1. 检查 API Key 是否完整复制
2. 检查 Base URL 是否正确
3. 确认账户余额充足
4. 检查网络连接（国外 API 可能需要科学上网）

---

返回 [主页](../README.md) | 返回 [使用教程](USAGE.md)

