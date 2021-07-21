近期做了科大讯飞的几个比赛的baseline，其中表情识别是70+，还有两个植物赛道是前十，近期都在这个仓库更新，有兴趣可以留意下

### Iflytek_comp
+ warmup_scheduler的安装:
```
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```
+ 食用方法: 
```
python main.py
```
+ 如果本地计算资源不足的话，可以尝试将配置文件中的amp(混合精度)设置为True
+ vgg, resnet18到efficientnet, swim, vit等模型对整体性能(performance)影响不大，而数据质量对模型表现影响较大
+ 可以尝试在数据质量上做优化: 
  +  使用out of distribution detection相关技术来清理数据
  +  使用多折交叉验证来保证模型对部分数据不可达，进而辅助数据清理
  +  使用诸如cleanlab之类的数据清理工具进行清理
  +  使用蒸馏技术(之后我可能会开源一个自蒸馏的版本)
  +  引入一些无监督的技巧来辅助清理数据
  +  多尝试soft label的参数(\alpha)
  +  To balance the softed loss and the focal loss
+ 数据中存在一些水印的问题，可以通过该论文提出的方法进行改善:https://maureenzou.github.io/ddac/ (不过我认为影响不会太大)
+ 数据增强的方式有待探索: 因为数据主要是三通道的灰度图，但是本baseline的数据增强方式比较适用于彩色图；同时数据的分辨率低，所以参赛选手可以多探索一些数据增强的方式
