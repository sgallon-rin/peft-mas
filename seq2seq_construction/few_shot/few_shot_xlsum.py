#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import random
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
"""
{
    "id": datasets.Value("string"),
    "url": datasets.Value("string"),
    "title": datasets.Value("string"),
    "summary": datasets.Value("string"),
    "text": datasets.Value("string"),
}
"""

SEED = 42
NUM_TRAIN = 4407
NUM_VAL = 550
NUM_TEST = 550

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets.shuffle(seed=SEED)
        self.raw_datasets = self.raw_datasets.flatten_indices()
        self.lang = args.seq2seq.lang
        cache_path = os.path.join(cache_root, "few_shot_{}_xlsum_train.cache".format(self.lang))
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            logger.info("Creating cache for few_shot_xlsum_train of language {}".format(self.lang))
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = copy.deepcopy(raw_data)
                    document = extend_data["text"]  # No .lower() for summarization; use text only, title not used
                    summary = extend_data["summary"]
                    if document.strip() == "" or summary.strip() == "":
                        logger.warning("Empty document or summary found! Skip.")
                        logger.warning(extend_data)
                        continue

                    extend_data.update({"struct_in": "",
                                        "text_in": document,
                                        "seq_out": summary})
                    self.extended_data.append(extend_data)
            self.extended_data = self.extended_data[:NUM_TRAIN]
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets.shuffle(seed=SEED)
        self.raw_datasets = self.raw_datasets.flatten_indices()
        self.lang = args.seq2seq.lang
        cache_path = os.path.join(cache_root, "few_shot_{}_xlsum_dev.cache".format(self.lang))
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            logger.info("Creating cache for few_shot_xlsum_dev of language {}".format(self.lang))
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = copy.deepcopy(raw_data)
                    document = extend_data["text"]
                    summary = extend_data["summary"]
                    if document.strip() == "" or summary.strip() == "":
                        logger.warning("Empty document or summary found! Skip.")
                        logger.warning(extend_data)
                        continue

                    extend_data.update({"struct_in": "",
                                        "text_in": document,
                                        "seq_out": summary})
                    self.extended_data.append(extend_data)
            self.extended_data = self.extended_data[:NUM_VAL]
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets.shuffle(seed=SEED)
        self.raw_datasets = self.raw_datasets.flatten_indices()
        self.lang = args.seq2seq.lang
        cache_path = os.path.join(cache_root, "few_shot_{}_xlsum_test.cache".format(self.lang))
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            logger.info("Creating cache for few_shot_xlsum_test of language {}".format(self.lang))
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = copy.deepcopy(raw_data)
                    document = extend_data["text"]
                    summary = extend_data["summary"]
                    if document.strip() == "" or summary.strip() == "":
                        logger.warning("Empty document or summary found! Skip.")
                        logger.warning(extend_data)
                        continue

                    extend_data.update({"struct_in": "",
                                        "text_in": document,
                                        "seq_out": summary})
                    self.extended_data.append(extend_data)
            self.extended_data = self.extended_data[:NUM_TEST]
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
