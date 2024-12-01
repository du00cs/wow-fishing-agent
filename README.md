# WoW Fishing Agent 魔兽世界钓鱼

> The main target is Chinese users, if you are interested, please use the translation software, I am too lazy to write English introduction.

本项目旨在通过非侵入的方式，用机器学习的方式来做一个自动钓鱼的工具。

# 0. 更新
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

- Windows(Ubuntu现在还登录不了国服游戏, Mac应该可以)
- Python >= 3.10
- 使用label-studio标注目标，借助音频数据采集代码采集音频识别数据
- 也可以通过百度网盘获得我训练的模型和训练数据，图像参考意义不大，我这边用了NDui，字体变了，水声是音频识别，应该能复用。[wow-fisher-data](https://pan.baidu.com/s/1AVfh9TD9xmA__V27BWcHuA?pwd=zytv)

```bash
poetry install

python -m fishing
```

# 2. 流程介绍

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
  judge["未钓上鱼检测"]
  
  cast --> capture
  capture --> valid
  cast --> sed
  valid -->|✅| end1
  valid -->|❌|cast
  sed -->|时机|end1
  end1 --> |截图|judge
```

# 3. 一些问题
1. 水声作为识别在有人跟你一起钓的情况下仍然还是无法正确工作。
2. （低优先级）流程上可以将不在有效范围时，直接去点坐标的流程也实现了，节约一次甩杆。
3. 包满了之后会自动停止

# 4. 致谢

1. facebook/wav2vec2 用来作为了特征抽取的底座
3. WA (Fishing Helper) 新手盒子，DD上面都有
4. YOLO: 特征抽取能力真的强，13年前用SIFT提的特征准确率只有30%
5. [moses-palmer/pynput](https://github.com/moses-palmer/pynput) 用来操作鼠标了
