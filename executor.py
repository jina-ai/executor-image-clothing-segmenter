from jina import Executor, DocumentArray, requests


class ClothingSegmenter(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
