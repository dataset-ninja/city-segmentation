Dataset **CitySegmentation** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzI1NzlfQ2l0eVNlZ21lbnRhdGlvbi9jaXR5c2VnbWVudGF0aW9uLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogIjQvald3UVRLQS9PYlowUjBtWWFsZHJtU3Q3U0tOWjVrSVRuR2Nia0xMdzg9In0=)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='CitySegmentation', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://www.kaggle.com/datasets/cceekkigg/berlin-aoi-dataset/download?datasetVersionNumber=2).