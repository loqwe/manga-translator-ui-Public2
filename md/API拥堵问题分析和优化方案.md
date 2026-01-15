# API调用拥堵问题分析和优化方案

## 问题诊断

### 核心问题
API翻译返回速度快于本地GPU/CPU处理能力,导致任务堆积和资源拥堵。

### 问题链条
```
检测+OCR(慢) → 翻译队列 → API翻译(快) → 修复队列 + 渲染队列 → 修复(慢) + 渲染(慢)
                                          ↓
                                    任务快速堆积
                                          ↓
                              GPU持续高负载 + API偶发超时
```

### 当前配置(瓶颈参数)
```python
batch_size = 3              # 翻译批量:每批3张
max_workers = 4             # 线程池:4个工作线程
translation_concurrency = 1  # 翻译并发:同时1个请求
translation_rpm = 10         # 速率限制:每分钟10个请求
```

### 日志证据
1. **修复(Inpainting)耗时长**:单张672x2048分辨率图片需数秒
2. **检测+OCR慢**:带fallback OCR时更慢
3. **翻译快速完成**:批量API调用,数秒内返回
4. **任务堆积**:翻译完成后大量任务等待修复和渲染
5. **API超时**:偶发Cloudflare 524错误(后端超时)

---

## 优化方案

### 方案1:降低翻译速度(推荐,最简单)

**思路**:减慢翻译速度,让GPU密集型任务有更多处理时间

#### 调整参数
```python
# 修改 manga_translator/mode/local.py 或命令行参数
batch_size = 1              # 从3改为1,减少批量大小
translation_rpm = 5         # 从10改为5,降低每分钟请求数

# 可选:如果仍然拥堵,进一步限制
batch_size = 1
translation_rpm = 3         # 更保守的速率
```

#### 优点
- 无需修改代码逻辑
- 立即生效
- 减少GPU峰值负载
- 降低API超时概率

#### 缺点
- 总体翻译时间增加
- API能力未充分利用

#### 实施方法
修改 `manga_translator/mode/local.py` 第302-312行:

```python
# 方案1a:保守配置(适合低配GPU)
batch_size = cli_config.get('batch_size', 1)  # 默认改为1
translation_rpm = cli_config.get('translation_rpm', 5)  # 新增速率参数

# 方案1b:中等配置(适合中等GPU)
batch_size = cli_config.get('batch_size', 2)
translation_rpm = cli_config.get('translation_rpm', 7)
```

---

### 方案2:动态队列监控和反压控制(复杂,效果好)

**思路**:实时监控队列长度,当下游堆积时暂停翻译

#### 实现逻辑
```python
# 在翻译worker中添加反压检测
async def _translation_worker(self):
    while not self.stop_workers:
        # 检查下游队列长度
        inpaint_backlog = self.inpaint_queue.qsize()
        render_backlog = self.render_queue.qsize()
        
        # 如果下游堆积超过阈值,暂停翻译
        if inpaint_backlog + render_backlog > 5:
            logger.info(f"[翻译] 下游堆积({inpaint_backlog}+{render_backlog}),暂停翻译")
            await asyncio.sleep(5)  # 等待5秒
            continue
        
        # 正常处理...
```

#### 优点
- 自适应调节
- 充分利用API能力
- 避免资源浪费
- 平滑处理峰值

#### 缺点
- 需要修改并发管道代码
- 增加复杂度
- 需要调优阈值

---

### 方案3:增加GPU工作线程(适合高端GPU)

**思路**:如果GPU有余力,增加并行处理能力

#### 调整参数
```python
max_workers = 6  # 从4增加到6
# 允许更多并发:检测、OCR、修复可以同时多个
```

#### 适用场景
- 高端GPU(如RTX 4090)
- VRAM充足(≥16GB)
- 任务主要耗时在GPU等待而非计算

#### 风险
- 可能加剧拥堵(如果GPU已满载)
- 增加VRAM占用
- 需要监控GPU利用率

---

### 方案4:优化Inpainting分辨率(治标)

**思路**:降低修复分辨率,加快处理速度

#### 检查当前配置
```bash
# 查看修复分辨率设置
grep -r "inpaint.*resolution\|inpaint.*size" manga_translator/
```

#### 可能的调整
- 降低修复最大分辨率(如从2048改为1536)
- 使用更快的inpainting模型
- 跳过小文本块的修复

#### 注意
- 可能影响输出质量
- 需要在速度和质量间权衡

---

### 方案5:批量预处理模式(架构改造)

**思路**:分离检测+OCR和翻译+渲染,先批量检测再批量翻译

#### 流程重构
```
阶段1:批量检测+OCR → 保存文本到JSON
阶段2:批量翻译JSON → 更新翻译结果
阶段3:批量修复+渲染 → 生成最终图片
```

#### 优点
- 更好的资源利用
- 可以跨会话处理
- 便于调试和重试
- 支持增量处理

#### 缺点
- 需要大量代码重构
- 改变现有工作流
- 增加磁盘IO

---

## 推荐实施策略

### 阶段1:快速缓解(立即实施)
1. **降低batch_size**:从3改为1或2
2. **降低translation_rpm**:从10改为5
3. **监控效果**:观察队列长度和GPU使用率

### 阶段2:中期优化(1-2天)
1. **实施方案2**:添加反压控制
2. **调优阈值**:根据实际硬件调整队列阈值
3. **添加监控**:输出队列统计日志

### 阶段3:长期改进(可选)
1. **评估GPU能力**:使用nvidia-smi监控
2. **考虑方案3/4**:根据硬件情况调整
3. **探索方案5**:如果需要处理大量图片

---

## 参数调优建议

### 低配置(GTX 1660/2060)
```python
batch_size = 1
max_workers = 4
translation_concurrency = 1
translation_rpm = 3
```

### 中等配置(RTX 3070/3080)
```python
batch_size = 2
max_workers = 4
translation_concurrency = 1
translation_rpm = 5
```

### 高配置(RTX 4080/4090)
```python
batch_size = 3
max_workers = 6
translation_concurrency = 2
translation_rpm = 8
```

---

## 监控指标

建议添加以下日志监控:
```python
logger.info(f"[队列状态] 翻译:{translation_queue.qsize()}, "
           f"修复:{inpaint_queue.qsize()}, "
           f"渲染:{render_queue.qsize()}, "
           f"GPU使用:{gpu_usage}%")
```

---

## 总结

**最快见效**:方案1(降低batch_size和rpm)  
**最优方案**:方案2(动态反压控制)  
**需要测试**:方案3(增加工作线程,可能适得其反)  

建议先尝试方案1,如果效果不明显再考虑方案2。
