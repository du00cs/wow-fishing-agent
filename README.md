# WoW Fishing Agent 魔兽世界钓鱼

> The main target is Chinese users, if you are interested, please use the translation software, I am too lazy to write English introduction.

本项目旨在通过非侵入的方式，用机器学习的方式来做一个自动钓鱼的工具。

# 0. 更新
2025/01/04 目前准确率已经较为可观了，接近100%(在奥格实测，使用Fishing Helper计数，结合miss/wrong计算得出)
- 流程重构，体验下来发现预测的时间都是20ms级的，并行没必要了，改为了串行实现
- 样本切片使用3s的切片，1s的音频在样本逐渐增加的情况下性能反而会下降
- 修复了numpy到torch内存上可能的一个数据错乱，会导致离线预测与在线预测的结果不一致的问题（这个问题会在验证集上准确率达到99%的情况下，实际使用仍然只有80%，但是预测保存的音频又无法复现）

2024/12/08
- 增加使用键盘按键暂停的功能

2024/12/05
- 增加只做水声检测的流程

2024/12/01
- 音频检测：
  1. 新增从loopback读取，不再需要虚拟声卡支持
  2. 新增根据钓鱼的施放和点击获取样本的功能
  3. 新增负样本采集功能，有基本的启动样本之后（10+段即可），对于背景可以采集预测为1的样本直接作为负样本
  4. 从直接使用yamnet更改为使用facebook/wav2vec2-base
- 目标检测：
  1. 标注数据改用[label-studio](https://labelstud.io/) 打通YOLO的训练与预测更容易了
  2. 调整了训练时的检测尺寸至1920，避免小尺寸的收缩成了一个点，丧失分辨能力
  3. 增加了“没有鱼上钩”的提示的识别
- 增加数据闭环
  1. 正样本：在一次钓鱼期间没有检测到水声，则保存此样本至`datasets/record/miss-bite` (还需要手动去掉非水声)
  2. 负样本：点击收鱼后，如果检测到了“没有鱼上钩”，则保存样本至`dataset/record/wrong-bte`
- 流程：
  1. 增加样本收集
- 更新完成之后，目前已经能够自动收集音频负样本数据了，对于整体系统的迭代还是有正向意义的。基本已经完成了做一个可学习的系统的目的。

# 1. 准备

1. Windows (打游戏不用windows也不行吧)
2. Python == 3.10
3. 目标检测：使用[label-studio](https://labelstud.io/)做图像目标标注，导出YOLO格式训练模型，也可以使用一个训练好的（使用了NDui，字体有变化，不通用），放在`models/od/best.pt`
4. 声音事件识别：放到 `models/bite_model`下，目录起名可以是`checkpoint-1`，也可以在参数中指定，也可以使用训练好的
5. 或者也可以通过百度网盘获得这边训练的模型和训练数据 [wow-fisher-data](https://pan.baidu.com/s/1AVfh9TD9xmA__V27BWcHuA?pwd=zytv)
   - 图像参考意义有限，这边用了NDui，字体变了
   - 水声是音频识别，应该能复用。
5. 参考模型和数据：可以通过百度网盘获得这边训练的模型和训练数据 [wow-fisher-data](https://pan.baidu.com/s/1AVfh9TD9xmA__V27BWcHuA?pwd=zytv) (音频使用 v2版)

## 1.1 安装依赖

不展开
```bash
# 准备好 pdm

pdm sync
```

# 2. 使用说明

启动命令
```bash
pdm run fishing
# python -m fishing
```

F12 开始/暂停

# 3. 流程介绍

代码很简单，介绍一下工作流程。

一般的无插件的正常流程
```mermaid
graph TD
  cast["模拟按键 🎣"]
  capture["截图"]
  od["鱼漂检测"]
  sed["水声检测 🌊"]
  end_["收竿
  (右键点击)"]
  
  cast --> capture
  capture --> od
  cast --> sed
  od -->|坐标|end_
  sed -->|时机|end_
```

使用了 WA(Fishing Helper) 之后会引入有效范围的提示，检测它就能够避开鱼漂检测（目前数据太少，鱼漂的置信度有点低，也不难）。
```mermaid
graph TD
  cast["模拟按键 🎣"]
  capture["截图"]
  valid["有效范围检测"]
  sed["水声检测 🌊"]
  end1["收竿
  (交互键)"]
  judge["'未钓上鱼' 检测"]
  
  cast --> capture
  capture --> valid
  cast --> sed
  valid -->|✅| end1
  valid -->|❌|cast
  sed -->|时机|end1
  end1 --> |截图|judge
```

# 4. 一些问题
1. 水声作为识别在有人跟你一起钓的情况下并不能作为判定依据，此时无法工作。
2. （低优先级）流程上可以将不在有效范围时，直接去点坐标的流程也实现了，节约一次甩杆。
4. 包满了之后会自动停止（其实是卡住了停的）
5. 仅在怀旧服上有真实使用案例，其它的未测试。

# 5. 致谢
1. 使用 facebook/wav2vec2 用来作为了特征抽取的底座
2. WA (Fishing Helper) 新手盒子，DD上面都有
3. YOLO: 特征抽取能力真的强，13年前用SIFT提的特征准确率只有30%
4. [moses-palmer/pynput](https://github.com/moses-palmer/pynput) 用来操作鼠标和键盘

# 6. TODO
- 增加渔点钓鱼模式，结合渔点检测做自动暂停。（想想这样钓乌龟应该就舒服一点了）