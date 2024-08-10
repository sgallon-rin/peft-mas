#!/usr/bin/env python
# -*- coding: utf-8 -*-

from metrics.utils import calculate_rouge


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args
        self.lang = args.evaluate.lang
        print("Initialized EvaluateTool for xlsum with language: {}".format(self.lang))

    def evaluate(self, preds, golds, section):
        golds = [item["summary"] for item in golds]
        print("Example Pred:\n{}\nGold:\n{}".format(preds[0], golds[0]))
        return calculate_rouge(preds, golds, rouge_lang=self.lang)
