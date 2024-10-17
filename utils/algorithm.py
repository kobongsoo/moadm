import os

#----------------------------------------------------------------------------
# RRF (Reciprocal rank fusion: 상호 순위 융합) 이란?
# Reciprocal rank fusion (RRF)은 서로 다른 관련성 지표(relevance indicators)를 가진 여러 개의 결과 집합(result sets)을 하나의 결과 집합으로 결합하는 방법입니다.
# RRF는 튜닝을 필요로 하지 않으며, 서로 다른 관련성 지표들이 상호 관련되지 않아도 고품질의 결과를 얻을 수 있습니다.
# RRF는 각 문서의 순위에 대한 점수를 결정하기 위해 다음과 같은 공식을 사용합니다.
#
#score = 0.0
#for q in queries:
#    if d in result(q):
#  	    score += 1.0 / ( k + rank( result(q), d ) )
#return score
#
# k는 순위에 대한 상수입니다.
# q는 질의(query)의 집합에서 하나의 질의를 의미합니다.
# d는 q의 결과 집합에서의 문서(document)를 나타냅니다.
# result(q)는 질의 q의 결과 집합을 의미합니다.
# rank(result(q), d)는 d가 result(q)에서의 순위를 나타냅니다. 순위는 1부터 시작합니다.
#
# 예시) k=1 일로 가정
# 질의 1 결과(result(q1)): [A, B, C, D]
# 질의 2 결과(result(q2)): [B, D, E, F]
#
# 문서 A의 점수:
# score(A) = 1.0 / (1 + rank(result(q1), A)) + 0
# = 1.0 / (1 + 1) + 0
# = 0.5 
# 문서 B의 점수:
# score(B) = 1.0 / (1 + rank(result(q1), B)) + 1.0 / (1 + rank(result(q2), B)) 
# = 1.0 / (1 + 2) + 1.0 / (1 + 1)
# = 0.33 + 0.5 = 0.83
#
# ===[사용 예시]===
#A = ["a", "c", "b", "e", "d"]
#B = ["b", "c", "a", "e", "f"]
#
# Combine lists A and B into a list of lists
#lists = [A, B]
#
# Define the weights for each list
#weights = [1, 1]
#
# Apply the weighted RRF algorithm
#result = weighted_reciprocal_rank_fusion(lists, weights)
#
#print(f'result:{type(result)}')
#print(result)
#
# Print the re-ranked result with RRF scores
#for item, score in result:
#    print(f"Item: {item}, RRF Score: {score:.4f}")
#===END===
#----------------------------------------------------------------------------

def weighted_reciprocal_rank_fusion(lists:list, weights:list=[1,1], k:int=1):
    # Initialize a dictionary to store the cumulative RRF score of each item
    rrf_scores = {}

    # Iterate over each list and its corresponding weight
    for lst, weight in zip(lists, weights):
        for rank, item in enumerate(lst):
            # Calculate the weighted RRF score for this item
            score = weight / (k + rank +1)
            
            # Accumulate the score in the dictionary
            if item in rrf_scores:
                rrf_scores[item] += score
            else:
                rrf_scores[item] = score

    # Sort items based on their cumulative RRF scores in descending order
    ranked_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the items along with their scores
    return ranked_items
