import csv
import logging
import math
import os
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
dataset_path = 'resources/'
output_path = 'output/'


def prepare_dataset():
    if not os.path.exists(dataset_path):
        raise RuntimeError('The similarity dataset has not found')

    logger.info('Read train dataset')
    trainpath = os.path.join(dataset_path, 'train.csv')
    train_dataset = []
    with open(trainpath, 'r', encoding='utf8') as file:
        reader = csv.DictReader(file, delimiter='|', quoting=csv.QUOTE_NONE)
        for row in reader:
            train_dataset.append(
                InputExample(texts=[row['query'], row['sentence']], label=int(row['similarity'])))
            train_dataset.append(
                InputExample(texts=[row['sentence'], row['query']], label=int(row['similarity'])))

    logger.info('Read dev dataset')
    devpath = os.path.join(dataset_path, 'dev.csv')
    dev_dataset = []
    with open(devpath, 'r', encoding='utf8') as file:
        reader = csv.DictReader(file, delimiter='|', quoting=csv.QUOTE_NONE)
        for row in reader:
            dev_dataset.append(InputExample(texts=[row['query'], row['sentence']], label=int(row['similarity'])))

    return train_dataset, dev_dataset


def download_model():
    modelpath = os.path.join(output_path, 'juridics-legal-bert')
    filepath = os.path.join(modelpath, 'model.zip')
    gdd.download_file_from_google_drive(file_id='1j1zaEVU4uO9ukVRHp2iqVN1xPWncmKfR',
                                        dest_path=filepath,
                                        unzip=True)
    return modelpath


def train(train_dataset, dev_dataset, model_filepath):
    train_batch_size = 4
    num_epochs = 4
    checkpoint = 'juridics-legal-bert-sms-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_savepath = os.path.join(output_path, checkpoint)
    model = CrossEncoder(model_filepath, num_labels=1)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_dataset, name='juridics')
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    logger.info("Warmup-steps: {}".format(warmup_steps))
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=5000,
              warmup_steps=warmup_steps,
              output_path=model_savepath)


if __name__ == '__main__':
    train_samples, dev_samples = prepare_dataset()
    model_path = download_model()
    train(train_samples, dev_samples, model_path)
