#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ..peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """peft"""
        # TODO: more PEFT methods
        self.peft_type = args.peft.peft_type
        assert self.peft_type in ["LORA"]
        self.lora_r = args.peft.lora_r
        self.lora_alpha = args.peft.lora_alpha
        self.lora_dropout = args.peft.lora_dropout

        # Load tokenizer and model.
        if "mbart-large-50" in args.bert.location:
            # ValueError for transformer 4.9.2 (but ok in later versions, tested on 4.29.2)
            from transformers import MBart50Tokenizer
            self.tokenizer = MBart50Tokenizer.from_pretrained(args.bert.location, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location,
        )
        # self.config = self.pretrain_model.config

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # for mbart
        if args.model.src_lang and args.model.tgt_lang:
            self.tokenizer.src_lang = args.model.src_lang
            self.tokenizer.tgt_lang = args.model.tgt_lang
            print("Set src_lang: {} and tgt_lang: {}".format(args.model.src_lang, args.model.tgt_lang))

        # apply peft
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        self.pretrain_model = get_peft_model(self.pretrain_model, peft_config)
        self.config = self.pretrain_model.config
        print(peft_config)
        self.pretrain_model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels):
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
        ).loss
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )
        return generated_ids
