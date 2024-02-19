from typing import List
from schema import Curriculum
from config import *


def topk(similar_users, warms, user_table, k: int) -> list:
    """
    Generate top-K recommendations for a given user based on similar users' preferences.
    
    Args:
    - similar_users (list of tuples): List of tuples containing similarity scores and user IDs.
    - warms (pd.DataFrame): DataFrame containing warm users' information.
    - user_table: User table object used for encoding user IDs.
    - k (int): Number of recommendations to generate.
    
    Returns:
    - list: List of top-K recommended courses.
    """
    def _swap(x: tuple):
        return x[1], x[0]
    
    duplicate_checker: dict = {}
    for similarity, user in similar_users:
        user_id = user_table.encode(user)
        history = warms[warms['user_id'] == user_id]
        for i in history.index:
            course, rating = history.loc[i]['item'], history.loc[i]['rating']
            score = rating * similarity
            # Bypass duplicated recommendations.
            if (course in duplicate_checker) and (duplicate_checker[course] > score):
                continue
            duplicate_checker[course] = score
            # Early stop
            if len(duplicate_checker) >= k:
                return sorted(map(_swap, duplicate_checker.items()), reverse=True)
    # len(ranking) < k
    return sorted(map(_swap, duplicate_checker.items()), reverse=True)
    

def recommend(ranking: list, curriculum: Curriculum) -> List[str]:
    """
    Convert course IDs into human-readable course names.
    
    Args:
    - ranking (list): List of tuples containing course IDs and their scores.
    - curriculum (Curriculum): Curriculum object used for decoding course IDs.
    
    Returns:
    - List[str]: List of recommended courses in human-readable format.
    """
    recommended_courses = []
    for item in ranking:
        subject, level = curriculum.decode(item[1])
        recommended_courses.append("_".join([subject, str(level)]))
    return recommended_courses
