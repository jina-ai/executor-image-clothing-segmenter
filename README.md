# ClothingSegmenter

This project hosts an executor that can be used to perform segmentation on images of fashion products.
The executor is based on a U2NET architecture pre-trained on iMaterialistic fashion 2019. The executor detects the
parts of the image that represent clothing or fashion items and filters out the rest of the content.


## Examples

![Sample 1](docs/assets/sample%231.jpg)
![Output 2](docs/assets/output%231.jpg)
![Sample 2](docs/assets/sample%232.jpg)
![Output 2](docs/assets/output%232.jpg)

## References

* [U2NET](https://arxiv.org/abs/2005.09007)
* [iMaterialistic fashion 2019](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data)
* [Training and inference code](https://github.com/levindabhi/cloth-segmentation)
* [Pre-trained model](https://drive.google.com/u/0/uc?id=1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ&export=download)


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ClothingSegmenter')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ClothingSegmenter')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
