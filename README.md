近期做了科大讯飞的几个比赛的baseline，其中表情识别是70+，还有两个植物赛道是前十，近期都在这个仓库更新，有兴趣可以留意下

### Iflytek_comp
+ warmup_scheduler的安装:
```
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```
+ 如果本地计算资源不足的话，可以尝试将配置文件中的amp(混合精度)设置为True
+ vgg, resnet18到efficientnet, swim, vit等模型对整体性能(performance)影响不大，而数据质量对模型表现影响较大
+ 可以尝试在数据质量上做优化: 
  +  使用out of distribution detection相关技术来清理数据
