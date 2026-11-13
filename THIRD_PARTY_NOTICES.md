# 开源来源与第三方说明

## 保留的源码快照

- 来源：[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)。
- 固定提交：`31863c707501bf1605d36842f43deb78793dbc5d`。
- 提交浏览：[该版本源码](https://github.com/OFA-Sys/Chinese-CLIP/tree/31863c707501bf1605d36842f43deb78793dbc5d)。
- 获取日期：2026-11-11。
- 原始许可：MIT，全文见[原始许可证](upstream/Chinese-CLIP/MIT-LICENSE.txt)。
- 文件清单、逐文件原始URL与SHA-256见[来源清单](upstream/Chinese-CLIP/SOURCE_MANIFEST.json)，共4个原始文件。

快照中的文件保持获取时的原始内容，原有版权声明和许可证一并保留。它是选取的源码参考快照，不是完整Git克隆；没有下载模型权重、大体积素材、Git历史或全部构建产物。

## 与当前实现的关系

图文编码模型加载、预处理和冻结双塔的接口；投影训练、MMR、索引服务为本仓库实现。

默认入口不导入、不安装、不执行`upstream/`中的包；当前可运行包、样例、测试和中文说明位于仓库根目录对应位置。运行上游包需要另建环境并按它的原始README操作，其版本约束与本项目可能不同。

本仓库自己编写部分按根目录[MIT许可证](LICENSE)提供。该许可不替代上游源码及将来下载模型权重的原有许可；发布时保留此说明、来源清单和上游许可证。合成示例是本仓库生成的学习数据。

## 本轮审核新增参考

- [CLIP论文：线性投影、温度初始化及缩放限制](https://arxiv.org/html/2103.00020v1)
- [Chinese-CLIP训练循环](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/training/train.py)
- [FAISS HNSW参数和空间组成](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS HNSW构造函数](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h)

这些链接用于核对概念，本轮未新增复制上游文件，原始4个文件的SHA-256保持不变。

## 技术接口参考

- [FAISS入门与持久化](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [归一化内积与余弦距离](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

以上资料用于核对方法和接口；本仓库没有沿用来源项目的准确率、延迟或任务通过率。实际本地验证见[验证结果](docs/验证结果.md)，真实模型效果需要在接入后独立评测。
