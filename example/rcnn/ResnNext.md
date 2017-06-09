## ResNext
### symbol 文件
[symbol_resnext](https://github.com/likelyzhao/mxnet/blob/dev-faster-rcnn/example/rcnn/rcnn/symbol/symbol_resnext.py)
### [下载模型文件 resnext-101](http://data.dmlc.ml/mxnet/models/imagenet/resnext/101-layers/resnext-101-0000.params)
### [修改`rcnn/symbol/__init__.py`](https://github.com/likelyzhao/mxnet/blob/dev-faster-rcnn/example/rcnn/rcnn/symbol/__init__.py#L6)
### [修改 `rcnn/config.py`](https://github.com/likelyzhao/mxnet/blob/dev-faster-rcnn/example/rcnn/rcnn/config.py#L172-L180)
### 训练或者测试时候 `--network resnext`