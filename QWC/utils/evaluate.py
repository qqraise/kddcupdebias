# coding=utf-8
from __future__ import division
from __future__ import print_function

import datetime
import json
import sys
import time
from collections import defaultdict

import numpy as np

# the higher scores, the better performance


def evaluate_each_phase(predictions, answers):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        while rank < 50 and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)


def evaluate(submit_fname, answer_fname, current_phase):
    try:
        answers = [{} for _ in range(10)]
        with open(answer_fname, 'r') as fin:
            for line in fin:
                line = [int(x) for x in line.split(',')]
                user_id, item_id, item_degree = line
                phase_id = user_id % 11
				# assert user_id % 11 == phase_id
                # exactly one test case for each user_id
                answers[phase_id][user_id] = (item_id, item_degree)
    except Exception as _:
        print('server-side error: answer file incorrect')
        return

    try:
        predictions = {}
        with open(submit_fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(',')
                user_id = int(line[0])
                if user_id in predictions:
                    print( 'submitted duplicate user_ids')
                    return
                item_ids=[int(i) for i in line[1:]]
                if len(item_ids) != 50:
                    print('each row need have 50 items')
                    return
                if len(set(item_ids)) != 50:
                    print(stdout, 'each row need have 50 DISTINCT items')
                    return
                predictions[user_id]=item_ids
    except Exception as _:
        print('submission not in correct format')

    scores=np.zeros(4, dtype = np.float32)

    # The final winning teams will be decided based on phase T=7,8,9 only.
    # We thus fix the scores to 1.0 for phase 0,1,2,...,6 at the final stage.
    if current_phase >= 7:  # if at the final stage, i.e., T=7,8,9
        scores += 7.0  # then fix the scores to 1.0 for phase 0,1,2,...,6
    phase_beg=(7 if (current_phase >= 7) else 0)
    phase_end=current_phase + 1
    for phase_id in range(phase_beg, phase_end):
        for user_id in answers[phase_id]:
            if user_id not in predictions:
                print(stdout, 'user_id %d of phase %d not in submission' %
                      (user_id, phase_id))
                return
        try:
            # We sum the scores from all the phases, instead of averaging them.
            scores += evaluate_each_phase(predictions, answers[phase_id])
        except Exception as _:
            return print('error occurred during evaluation')
    score = float(scores[0])
    ndcg_50_full, ndcg_50_half, hitrate_50_full, hitrate_50_half = scores
    print("score:%.4f, ndcg_50_full; %.4f, hitrate_50_full: %.4lf,, ndcg_50_half: %.4f, hitrate_50_full: %.4f" 
               % (score, ndcg_50_full, hitrate_50_half, ndcg_50_half, hitrate_50_full))
