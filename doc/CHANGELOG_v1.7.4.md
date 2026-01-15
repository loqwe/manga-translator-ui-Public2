# Changelog v1.7.4

## 新增
- **AI断句自动优化**: 新增断句优化功能，自动测试所有断句组合以最大化字体大小。
  - 支持所有排版模式（smart_scaling、balloon_fill、strict、default）
  - 智能选择：优先字体大小，其次段落均匀度
  - 配置项：`optimize_line_breaks`（默认关闭）

## 改进
- **断句提示词优化**: 更新 `system_prompt_line_break.json`
  - 规则3改为更灵活的表述（可插入 X-2 到 X 个标记）
  - 新增规则：避免最后一段只有一个字
  - 去掉字符长度考量部分，简化提示词
  
- **全角括号支持**: 统一支持 `[BR]`、`【BR】`、`<br>` 三种断句标记格式

## 修复
- **打包依赖**: 添加 matplotlib 到依赖列表，修复打包后运行时缺少模块的问题
  - 更新 `requirements_gpu.txt` 和 `requirements_cpu.txt`
  - 更新 PyInstaller 配置文件

- **日志级别**: 根据 `verbose` 配置自动设置控制台日志级别（DEBUG/INFO）

## 技术细节
- 优化算法会生成所有可能的断句组合，计算每种组合的渲染效果
- 跳过规则：第一段字符数<2时不会去掉第一个断句
- 日志输出：优化结果使用 INFO 级别，详细过程使用 DEBUG 级别

