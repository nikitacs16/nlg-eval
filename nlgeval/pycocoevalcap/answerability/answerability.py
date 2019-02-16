#!/usr/bin/env python
# 
# File Name : answerability.py
#
# Description : Wrapper for Answerability scorer.
#

from .answerability_scorer import AnswerabilityScorer


class Answerability:
    def __init__(self, ngram_metric='Bleu_4'):
        # default compute Blue score up to 4
        self._ngram_metric = ngram_metric
     
    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        answerability_scorer = AnswerabilityScorer(self._ngram_metric)
      
        score, scores = answerability_scorer.compute_score(gts, res)
        return score, scores

    def method(self):
        return "Answerability"
