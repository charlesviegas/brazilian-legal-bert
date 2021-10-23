import logging
import os

import pandas as pd
from more_itertools import pairwise, zip_offset
from sklearn.model_selection import train_test_split


class Logger:
    def __init__(self):
        console_handler = logging.StreamHandler()
        lineformat = '[%(asctime)s] | %(levelname)s | [%(process)d - %(processName)s]: %(message)s'
        dateformat = '%d-%m-%Y %H:%M:%S'
        logging.basicConfig(format=lineformat, datefmt=dateformat, level=20, handlers=[console_handler])
        self.logger = logging.getLogger()

    def info(self, message):
        self.logger.info(message)


class DatasetManager:
    DATASET_ROOTPATH = 'resources/'

    def __init__(self, logger):
        self.logger = logger

    def read(self, filename):
        if not os.path.exists(self.DATASET_ROOTPATH):
            raise RuntimeError('Scraper result has not found')
        filepath = os.path.join(self.DATASET_ROOTPATH, filename)
        dataset = pd.read_csv(filepath, sep='|', encoding='utf-8-sig')
        self.logger.info(f'Dataset read from {filepath}')
        return dataset

    def save(self, dataset, filename):
        dataset = pd.DataFrame(dataset)
        filepath = os.path.join(self.DATASET_ROOTPATH, filename)
        dataset.to_csv(filepath, sep='|', encoding='utf-8-sig', index_label='index')
        self.logger.info(f'Dataset save {filepath}')


class SimilarityCombiner:
    HEADER = {'area': [], 'tema': [], 'discussao': [], 'id1': [], 'ementa1': [], 'id2': [], 'ementa2': [],
              'similarity': []}

    def __init__(self, source, logger):
        self.source = source
        self.target = pd.DataFrame(self.HEADER)
        self.logger = logger

    def process(self):
        self.combine_similar()
        self.combine_unsimilar()
        return self.target

    def combine_similar(self):
        groups = self.source.groupby('DISCUSSÃO')
        for group_name, group in groups:
            self.logger.info(f'Processing group {group_name} with {len(group)} itens')
            pairs = list(pairwise(group.index))
            for pair in pairs:
                sentence1 = group.loc[pair[0]]
                sentence2 = group.loc[pair[1]]
                item = self.__create_item(sentence1, sentence2, similarity=1)
                self.target = self.target.append(item, ignore_index=True)

    def combine_unsimilar(self):
        groups = self.source.groupby('DISCUSSÃO')
        group_pairs = list(pairwise(groups))
        for group_pair in group_pairs:
            group1_name = group_pair[0][0]
            group1_data = group_pair[0][1]
            group2_name = group_pair[1][0]
            group2_data = group_pair[1][1]
            self.logger.info(f'{group1_name} X {group2_name} ')
            sentence_pairs = list(zip(group1_data.index, group2_data.index))
            for pair in sentence_pairs:
                sentence1 = group1_data.loc[pair[0]]
                sentence2 = group2_data.loc[pair[1]]
                item = self.__create_item(sentence1, sentence2, similarity=0)
                self.target = self.target.append(item, ignore_index=True)
            sentence_pairs = list(zip_offset(group1_data.index, group2_data.index, offsets=(0, 1), longest=False))
            for pair in sentence_pairs:
                sentence1 = group1_data.loc[pair[0]]
                sentence2 = group2_data.loc[pair[1]]
                item = self.__create_item(sentence1, sentence2, similarity=0)
                self.target = self.target.append(item, ignore_index=True)

    @staticmethod
    def __create_item(sentence1, sentence2, similarity):
        return {
            'area': sentence1['ÁREA'],
            'tema': sentence1['TEMA'],
            'discussao': sentence1['DISCUSSÃO'],
            'id1': sentence1['TÍTULO'],
            'ementa1': sentence1['EMENTA'],
            'id2': sentence2['TÍTULO'],
            'ementa2': sentence2['EMENTA'],
            'similarity': similarity
        }


class Splitter:
    @staticmethod
    def split_train_test(dataset):
        train_samples, test_samples = train_test_split(dataset, train_size=0.75, test_size=0.25, random_state=42,
                                                       shuffle=True)
        return train_samples, test_samples


if __name__ == '__main__':
    main_logger = Logger()

    dataset_manager = DatasetManager(main_logger)
    source_dataset = dataset_manager.read('consultas-prontas.csv')

    similarity_manager = SimilarityCombiner(source_dataset, main_logger)
    target_dataset = similarity_manager.process()

    train_dataset, dev_dataset = Splitter.split_train_test(target_dataset)

    dataset_manager.save(train_dataset, 'consultas-prontas-train.csv')
    dataset_manager.save(dev_dataset, 'consultas-prontas-dev.csv')
