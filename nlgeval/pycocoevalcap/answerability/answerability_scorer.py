#!/usr/bin/env python

# answerability_scorer.py


import codecs
import json
import logging
import os
import sys
import tempfile

import numpy as np
import six
from six.moves import reload_module
from ..bleu.bleu import Bleu
from ..rouge.rouge import Rouge
#from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from tokenizer.ptbtokenizer import PTBTokenizer
from six.moves import xrange as range

if six.PY2:
    reload_module(sys)
    sys.setdefaultencoding('utf-8')

stop_words = {"did", "have", "ourselves", "hers", "between", "yourself",
              "but", "again", "there", "about", "once", "during", "out", "very",
              "having", "with", "they", "own", "an", "be", "some", "for", "do", "its",
              "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s",
              "am", "or", "as", "from", "him", "each", "the", "themselves", "until", "below",
              "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were",
              "her", "more", "himself", "this", "down", "should", "our", "their", "while",
              "above", "both", "up", "to", "ours", "had", "she", "all", "no", "at", "any",
              "before", "them", "same", "and", "been", "have", "in", "will", "on", "does",
              "yourselves", "then", "that", "because", "over", "so", "can", "not", "now", "under",
              "he", "you", "herself", "has", "just", "too", "only", "myself", "those", "i", "after",
              "few", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "further",
              "was", "here", "than"}

question_words_global = {'What', 'Which', 'Why', 'Who', 'Whom', 'Whose', 'Where', 'When', 'How', 'Is'}
question_words_global.update([w.lower() for w in question_words_global])

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.keys()}

    def evaluate(self, ngram_metric):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco[imgId]#.imgToAnns[imgId]
            res[imgId] = self.cocoRes[imgId]#.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        if ngram_metric == 'ROUGE_L':
            scorers = [
                (Bleu(1), ["Bleu_1"]),
                (Rouge(), "ROUGE_L")
            ]
        else:
            assert ngram_metric.startswith('Bleu_')
            i = ngram_metric[len('Bleu_'):]
            assert i.isdigit()
            i = int(i)
            assert i > 0
            scorers = [
                (Bleu(i), ['Bleu_{}'.format(j) for j in range(1, i + 1)]),
            ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
        self.setEvalImgs()
        return self.evalImgs
   
    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


class AnswerabilityScorer(object):
  
    def __init__(self, ngram_metric):
        self.ngram_metric = ngram_metric


    def remove_stopwords_and_NER_line(self, question, relevant_words=None, question_words=None):
        if relevant_words is None:

            question = question.split()
            if question_words is None:
               question_words = question_words_global

            temp_words = []
            for word in question_words:
                for i, w in enumerate(question):
                    if w == word:
                        temp_words.append(w)
                        # If the question type is 'what' or 'which' the following word is generally associated with
                        # with the answer type. Thus it is important that it is considered a part of the question.
                        if i+1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                            temp_words.append(question[i+1])

            question_split = [item for item in question if item not in temp_words]
            ner_words = question_split
            temp_words = []

            for i in ner_words:
                if i[0].isupper() == False:
                    if i not in stop_words :
                        temp_words.append(i)

            return " ".join(temp_words)
        else:
            question_words = question.split()
            temp_words = []
            for i in question_words:
                for j in relevant_words:
                    if j.lower() in i:
                        temp_words.append(i)
            return " ".join(temp_words)

    def NER_line(self, question):
        q_types = question_words_global
        question_words = question.split()
        if question_words[0].lower() in q_types:
            question_words = question_words[1:]

        temp_words = []
        for i in question_words:
            if i[0].isupper():
                temp_words.append(i)

        return " ".join(temp_words)


    def get_stopwords(self, question):
        question_words = question.split()
        temp_words = []
        for i in question_words:
            if i.lower() in stop_words:
                temp_words.append(i.lower())

        return " ".join(temp_words)

    def loadJsonToMap(self, json_file):
        with codecs.open(json_file, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        img_to_anns = {}
        for entry in data:
            if entry['image_id'] not in img_to_anns:
                img_to_anns[entry['image_id']] = []
            summary = dict(caption=entry['caption'], image_id=entry['caption'])
            img_to_anns[entry['image_id']].append(summary)
        return img_to_anns


    def questiontype(self, question, questiontypes=None):

        if questiontypes is None:
            types = question_words_global
            question = question.strip()
            temp_words = []
            question = question.split()

            for word in types:
                for i, w in enumerate(question):
                    if w == word:
                        temp_words.append(w)
                        if i+1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                            temp_words.append(question[i+1])

            return " ".join(temp_words)
        else:
            for i in questiontypes:
                if question.startswith(i + " "):
                    return i
                else:
                    return " "
    
    def _get_json_format_qbleu(self, lines, output_path_prefix, relevant_words=None, questiontypes=None):
        if not os.path.exists(os.path.dirname(output_path_prefix)):
            os.makedirs(os.path.dirname(output_path_prefix))
        name = output_path_prefix + '_components'
        pred_sents_impwords = []
        pred_sents_ner = []
        pred_sents_qt = []
        pred_sents_sw = []
        for line in lines:
            line_impwords = self.remove_stopwords_and_NER_line(line, relevant_words)
            line_ner = self.NER_line(line)
            line_qt = self.questiontype(line, questiontypes)
            line_sw = self.get_stopwords(line)
            pred_sents_impwords.append(line_impwords)
            pred_sents_ner.append(line_ner)
            pred_sents_qt.append(line_qt)
            pred_sents_sw.append(line_sw)

        ref_files = [os.path.join(name + "_impwords"), os.path.join(name + "_ner"), os.path.join(name + "_qt"), os.path.join(name + "_fluent"), os.path.join(name + "_sw")]

        data_pred_impwords = []
        data_pred_qt = []
        data_pred_ner = []
        data_pred = []
        data_pred_sw = []

        for index, s in enumerate(pred_sents_impwords):
            data_pred_impwords.append(dict(image_id=index, caption=s))
            data_pred_qt.append(dict(image_id=index, caption=pred_sents_qt[index]))
            data_pred_ner.append(dict(image_id=index, caption=pred_sents_ner[index]))
            data_pred.append(dict(image_id=index, caption=lines[index]))
            data_pred_sw.append(dict(image_id=index, caption=pred_sents_sw[index]))

        with open(ref_files[0], 'w') as f:
            json.dump(data_pred_impwords, f, separators=(',', ':'))
        with open(ref_files[1], 'w') as f:
            json.dump(data_pred_ner, f, separators=(',', ':'))
        with open(ref_files[2], 'w') as f:
            json.dump(data_pred_qt, f, separators=(',', ':'))
        with open(ref_files[3], 'w') as f:
            json.dump(data_pred, f, separators=(',', ':'))
        with open(ref_files[4], 'w') as f:
            json.dump(data_pred_sw, f, separators=(',', ':'))

        return ref_files
    

    def compute_answerability_scores(self, all_scores, ner_weight, qt_weight, re_weight, d, output_dir, ngram_metric="Bleu_4"):
        fluent_scores = [x[ngram_metric] for x in all_scores]
        imp_scores =  [x['imp'] for x in all_scores]
        qt_scores = [x['qt'] for x in all_scores]
        sw_scores = [x['sw'] for x in all_scores]
        ner_scores =  [x['ner'] for x in all_scores]

        new_scores = []

        for i in range(len(imp_scores)):
            answerability = re_weight*imp_scores[i] + ner_weight*ner_scores[i]  + \
                qt_weight*qt_scores[i] + (1-re_weight - ner_weight - qt_weight)*sw_scores[i]

            temp = d*answerability + (1-d)*fluent_scores[i]
            new_scores.append(temp)
           
        mean_answerability_score = np.mean(new_scores)
        mean_fluent_score = np.mean(fluent_scores)
        return mean_answerability_score, mean_fluent_score
            


    def calc_score(self, hypotheses, references):
        ngram_metric = self.ngram_metric
        ner_weight =  0.6 
        qt_weight = 0.2 
        re_weight = 0.1
        delta = 0.7
        data_type='SQuAD'
        relevant_words = None
        question_words = None   
        output_dir = tempfile.gettempdir()
        filenames_1 = self._get_json_format_qbleu(references, os.path.join(output_dir, 'refs'),
                                             relevant_words, question_words)
        filenames_2 = self._get_json_format_qbleu(hypotheses, os.path.join(output_dir, 'hyps'),
                                             relevant_words, question_words)
        final_eval = []
        final_eval_f = []
        for file_1, file_2 in zip(filenames_1, filenames_2):
            coco = self.loadJsonToMap(file_1)
            os.remove(file_1)
            cocoRes = self.loadJsonToMap(file_2)
            os.remove(file_2)
            cocoEval_precision = COCOEvalCap(coco, cocoRes)
            cocoEval_recall = COCOEvalCap(cocoRes, coco)
            cocoEval_precision.params['image_id'] = cocoRes.keys()
            cocoEval_recall.params['image_id'] = cocoRes.keys()
            eval_per_line_p = cocoEval_precision.evaluate(ngram_metric)
            eval_per_line_r = cocoEval_recall.evaluate(ngram_metric)

            f_score = zip(eval_per_line_p, eval_per_line_r)
            temp_f = []
            for p, r in f_score:
                if (p['Bleu_1'] + r['Bleu_1'] == 0):
                    temp_f.append(0)
                    continue
                temp_f.append(2 * (p['Bleu_1'] * r['Bleu_1']) / (p['Bleu_1'] + r['Bleu_1']))

            final_eval_f.append(temp_f)
            final_eval.append(eval_per_line_p)

        metric_scores = [fl[ngram_metric] for fl in final_eval[3]] #only BLEU support 
        save_all = []
        all_scores = zip(final_eval_f[0], final_eval_f[1], final_eval_f[2], final_eval_f[4],
                         metric_scores)
        for imp, ner, qt, sw, metric_score in all_scores:
            d = {'imp': imp, 'ner': ner, 'qt': qt, 'sw': sw, ngram_metric: metric_score}
            save_all.append(d)
        return self.compute_answerability_scores(save_all, ner_weight, qt_weight, re_weight, delta, output_dir, ngram_metric)

        


    def compute_score(self, gts, res):
        #only single-reference support now.
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        ground_truths = []
        hypotheses = []
        score = []
        
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]
            ground_truths.append(ref[0])
            hypotheses.append(hypo[0])

        average_answerability_score, average_fluent_score = self.calc_score(hypotheses, ground_truths)
    
        return average_answerability_score, average_fluent_score #this is different

            