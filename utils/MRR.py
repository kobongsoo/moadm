import os
import csv
import time
import torch
import logging

#==================================================================================================
# MRR(Mean Reciprocal Rank) 함수
# => IN : ground_truths - 정답 contextid 1차원 리스트(예: [10001,10002, 10003, 10004,...])
#         predictions - 예측값 contextid 2차원 리스트(예: [[10003,10004,...],[10010, 10007,...],[],[],...]
# => OUT : 각 쿼리에 대한 ranks 값 리스트(0~1범위), 전체 평균 쿼리 ranks 값(MRR) 리턴함
#==================================================================================================
def mean_reciprocal_rank(ground_truths, predictions):
    reciprocal_ranks = []
    
    for gt, prediction in zip(ground_truths, predictions):
        rank = 1
        bsearch=False
        for p in prediction:
            #print(f'pred:{p}-gt:{gt}')
            #if p in gt:
            if p == gt:
                reciprocal_ranks.append(1/rank)
                bsearch=True
                #print(f'gt:{gt}=>{1/rank}')
                break
            rank += 1
            
        if bsearch==False:
            reciprocal_ranks.append(0)
            #print(f'gt:{gt}=>0')
       
    # 각 쿼리에 대한 ranks 값(0~1범위), 전체 평균 쿼리 ranks 값(MRR) 리턴함
    return reciprocal_ranks, sum(reciprocal_ranks) / len(reciprocal_ranks)
#==================================================================================================