import os
import torch
import heapq
import pandas as pd
from typing import List


def create_model_path(model_folder, model_name):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    count = 1
    model_path = model_folder + f"{model_name}_{count}.pt"
    while os.path.exists(model_path):
        model_path = model_folder + f"{model_name}_{count}.pt"
        count += 1
    return model_path


def topk(data: pd.DataFrame, model, user_id: int, courses_to_predict: list, k: int) -> (list, list):
    user_id_to_predict = torch.tensor([user_id] * len(courses_to_predict))
    item_id_to_predict = torch.tensor(courses_to_predict)
    
    course_history = set(data[data['user_id']==user_id]['item'].unique())
    predicted_rating: torch.Tensor = model(user_id_to_predict, item_id_to_predict)

    # Max-heap sort
    covered_maxheap = []
    uncovered_maxheap = []
    for rating, course in zip(list(predicted_rating), courses_to_predict):
        if course in course_history:
            heapq.heappush(covered_maxheap, (-rating.item(), course))
        else:
            heapq.heappush(uncovered_maxheap, (-rating.item(), course))
    
    def _revert(x: tuple):
        return -x[0], x[1]

    # Early stop
    cover_iter = range(min(k, len(covered_maxheap)))
    uncover_iter = range(min(k, len(uncovered_maxheap)))
    covered_ranking = [_revert(heapq.heappop(covered_maxheap)) for _ in cover_iter]
    uncovered_ranking = [_revert(heapq.heappop(uncovered_maxheap)) for _ in uncover_iter]
    
    return covered_ranking, uncovered_ranking


def recommend(curriculum, ranking: list) -> List[str]:
    recommended_courses = []
    for item in ranking:
        subject, level = curriculum.decode(item[1])
        recommended_courses.append("_".join([subject, str(level)]))
    return recommended_courses
