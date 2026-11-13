# FAISS + MMR + Chinese-CLIP 复现初版

对应简历第一项目“知识驱动大模型下的多模态智能计算及应用研究”的多模态检索模块。先把**配对样本/上游特征 → 归一化 → 索引 → MMR → 来源结果**跑通，再替换为真实图文编码与领域数据。

当前默认输入是明确给定的4维向量，图片是本仓库生成的几何示意图；固定向量并不是从这些图片提取的。这样可以单独观察算法的输入输出，不需要先下载模型。可选路径提供真实FAISS、PyTorch投影层训练和Chinese-CLIP适配器。

## 1. 快速运行（Windows PowerShell）

使用 Python 3.11 或 3.12；系统默认 Python 如果是3.10，请另选3.12解释器。

```powershell
cd D:\software\github_cun\FAISS-MMR-Chinese_CLIP
py -3.12 -m venv .venv
$projectPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $projectPython -m pip install -e '.[dev]'
& $projectPython -m mmclip demo
& $projectPython -m pytest -q
```

最小演示使用NumPy精确内积检索。没有安装的可选集成测试会显示 `skipped`。安装对应库后运行真实索引、服务和训练：

```powershell
& $projectPython -m pip install -e '.[dev,faiss,api,train]'
& $projectPython -m mmclip demo --backend faiss --kind flat --out runs/faiss-flat
& $projectPython -m mmclip demo --backend faiss --kind hnsw --out runs/faiss-hnsw
& $projectPython -m mmclip evaluate
& $projectPython -m mmclip train --steps 60
& $projectPython -m pytest -q
```

PyTorch较大；需要CPU专用版本时先按PyTorch官方说明安装CPU wheel，再安装 `.[train]`。本初版验证使用CPU，不要求显卡。

## 2. 输入和输出

已知查询向量为 `[1, 0, 0, 0]`，语料向量按 `features.json` 的 `ids` 顺序与 `pairs.json` 一一对齐。

```python
from mmclip.index import VectorIndex
from mmclip.pipeline import RetrievalPipeline

index = VectorIndex(
    [[1, 0], [0.8, 0.6], [0, 1]],
    [{'id': 'a', 'source': 'demo://a'},
     {'id': 'b', 'source': 'demo://b'},
     {'id': 'c', 'source': 'demo://c'}],
)
result = RetrievalPipeline(index).search_vector(
    [1, 0], candidate_k=3, top_k=2, diversity=0.55
)
print([item['id'] for item in result['results']])  # ['a', 'b']
```

`demo` 的结果保存到 `runs/demo/result.json`，索引保存在 `runs/demo/index/`。结果包含 `encoder`、实际 `backend`、`candidate_ids` 和带原始余弦分数、来源与排名的 `results`。`diversity` 参数是MMR公式里的 λ，**越大越偏向查询相关性**，并不是越大越去重。

更换上游输入时，可以先直接传向量：

```powershell
& $projectPython -m mmclip build --backend faiss --out runs/index
& $projectPython -m mmclip search --index runs/index --vector '[1,0,0,0]'
```

向量维度必须匹配索引；零向量、NaN和无穷值都会报错。直接输入向量的调用方负责保证它与索引使用同一模型和投影版本。

## 3. 目录与代码作用

```text
mmclip/
  data.py           配对元数据、清洗、拒绝记录、PyTorch Dataset
  encoders.py       固定特征、Chinese-CLIP、投影后编码器
  features.py       真实图文样本 → 两塔冻结特征JSON
  training.py       新增线性适配层、可学习/固定温度、对称InfoNCE
  config.py         HNSW、训练温度和服务默认参数
  cache.py          查询哈希与有界LRU+TTL编码缓存
  vectors.py        输入校验和增量MMR
  index.py          NumPy / FAISS Flat / FAISS HNSW与ID映射
  pipeline.py       候选检索 → MMR → 来源结果
  batching.py       有界队列、小批次合并、编码LRU缓存
  api.py            FastAPI文本编码、检索、原文元数据接口
  evaluate.py       Flat对HNSW的ANN召回、延迟和序列化大小
  __main__.py       命令行入口
examples/           配对清单、理想特征、请求和合成图片
tests/              算法边界、真实库集成和训练测试
docs/               逐文件说明、真实接入教程、简历对应关系
upstream/           带许可证与commit的Chinese-CLIP参考快照
```

继续阅读：[代码与输入输出](docs/01_代码与输入输出.md)、[真实模型接入](docs/02_真实模型接入.md)、[实验与简历对应](docs/03_实验与简历对应.md)。

## 4. 启动本地接口

```powershell
& $projectPython -m mmclip serve --index runs/index --port 8001
```

另一个PowerShell窗口：

```powershell
$request = @{query='医院设备采购'; top_k=3; candidate_k=6; diversity=0.55} | ConvertTo-Json
Invoke-RestMethod 'http://127.0.0.1:8001/search' -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($request)) -ContentType 'application/json; charset=utf-8'
```

`POST /v1/retrieve`返回供上层RAG消费的chunk列表、版本与MMR观测指标。`GET /health` 查看实际模式和参数；`POST /encode` 返回文本向量；`POST /search` 返回结果；`GET /documents/hospital-1` 返回来源元数据。`/docs` 可交互调试。固定编码器只接受样例中列出的文本；任意句子返回422，避免把查表误认为模型推理。

这是单进程学习服务，未接入身份认证。队列满返回429（附Retry-After）、完整请求等待超时返回504。图像编码目前由离线 `build/export-features` 入口完成，HTTP接口暂未实现任意图片上传。

## 5. 已实现和待验证

已实际验证：真实FAISS Flat/HNSW读写、增量MMR、真实PyTorch投影训练、FastAPI生命周期、批次对应关系、缓存、超时及队列满处理。

Chinese-CLIP权重与真实领域训练未在本次下载/运行；模型适配器和特征导出接口已提供，接入后需要按教程重新训练、建库和评测。当前训练损失来自6组合成特征，不能作为领域适配收益；小样本ANN结果也不能代表百万级性能。

## 6. 开源来源

参考 [Chinese-CLIP官方仓库](https://github.com/OFA-Sys/Chinese-CLIP) 与 [FAISS官方教程](https://github.com/facebookresearch/faiss/wiki/Getting-started)。上游许可和固定提交见 [第三方说明](THIRD_PARTY_NOTICES.md)。本仓库算法、服务、样例与教程是独立编写的复现实现，上游文件仅作对照，不参与默认执行。

## 本地验证与预期产物

参见[验证结果](docs/验证结果.md)。`examples/expected/`保存本次固定输入的实际输出；执行生成的中间产物写入已忽略的`runs/`，可重复运行。


## 实现说明审核（2026-11-12）

已按上传说明核对并同步：初始温度0.07、可学习logit_scale及逆温度上限100；HNSW M=16 / efConstruction=200 / efSearch=64；队列128、合批16、等待5ms、编码缓存TTL3600秒、请求等待10秒及top_k≤50。所有参数均是起点，支持CLI调整。

当前训练对象是冻结embedding之后**额外添加**的两个线性适配层。MMR指标使用同K条平均余弦对照，候选缩减比例单独命名；“62%→71%”“温度贡献3点”等材料中的数字未获得实验支持。

请先读[逐项审核与面试说法](docs/04_实现细节审核与参数同步.md)，再按[参数与接口教程](docs/05_参数与上层接口教程.md)运行。新的验证结果与样例输出同样保留在仓库内。
