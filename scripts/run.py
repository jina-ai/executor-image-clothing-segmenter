import os
import sys
import click
import logging
import warnings

from jina import Document, DocumentArray
from typing import Optional


# import parent dir so as to be able to locate modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from executor import ClothingSegmenter
from executor.model import ClothingSegmentationModel


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger("urllib3.connectionpool").setLevel(level=logging.WARNING)

logging.basicConfig(format='%(asctime)s [%(levelname)-8s] [%(name)-4s] :: %(message)s', level=logging.DEBUG)


@click.command()
@click.option('--model-path', type=str, default=None, help='The path to the pre-trained U2NET model')
@click.option('--input-dir', type=str, default='./input', help='The input data root directory')
@click.option('--output-dir', type=str, default='./output', help='The output data root directory')
@click.option('--batch-size', type=int, default=4, help='Processing batch size')
def run(
    model_path: Optional[str] = None,
    input_dir: str = './input',
    output_dir: str = './output',
    batch_size: int = 4,
):
    """Run the image segmentation executor on an image dataset ✨"""

    logging.info('Loading the executor ...')
    if model_path is not None:
        if not os.path.isfile(model_path):
            logging.critical(f'No such file: {model_path}')
            sys.exit(1)
    else:
        logging.debug('Downloading ...')
        ClothingSegmentationModel.download()
        logging.debug('Finished downloading!')

    executor = ClothingSegmenter(model_path=model_path, batch_size=batch_size)

    logging.info('Reading input data ...')

    if not os.path.isdir(input_dir):
        logging.critical(f'No such directory: {input_dir}')
        sys.exit(1)

    docs = DocumentArray()
    for root, _, fnames in os.walk(input_dir):
        for fname in fnames:
            doc = Document(id=fname, uri=os.path.join(root, fname))
            doc.convert_uri_to_image_blob()
            docs.append(doc)

    logging.debug(f'Found {len(docs)} images!')

    logging.info('Configuring output ...')
    if os.path.exists(output_dir):
        logging.critical(f'File/directory {output_dir} already exists')
        sys.exit(1)

    os.makedirs(output_dir)

    logging.info('Running the executor ...')
    docs = executor.segment(docs)

    for doc in docs:
        doc.dump_image_blob_to_file(os.path.join(output_dir, doc.id))


if __name__ == '__main__':
    run()
