import pandas as pd
from tqdm import tqdm
import random
from random import randint, choice

from config import *


# Params
N_USER = 100
N_SAMPLIES = 50
N_COLD_USERS = 10

MIN_TIME = 15
MAX_TIME = 25
MIN_AGE = 36
MAX_AGE = 84

# Invariable User infomation table
User_IDs = []
duplicated_checker = set()
for _ in range(N_USER):
    while True:
        user_id = f"USER{randint(0, 9999):0{4}d}"
        if user_id not in duplicated_checker:
            duplicated_checker.add(user_id)
            User_IDs.append(user_id)
            break
Age        = [ randint(MIN_AGE, MAX_AGE) for i in range(N_USER) ]
Gender     = [ randint(0, 1) for i in range(N_USER) ]  # {0: MAN, 1: WOMAN}
Disability = [ randint(0, 4) for i in range(N_USER) ]  # {0:'정상', 1:'경증자폐', 2:'중증자폐', 3:'발달지연', 4:'기타'}
# 


# Data categories for mapping
Stability = [
    1,  # 'type1' : 머리-어깨-몸통 움직임 축이 각각 다름 
    2,  # 'type2' : 머리-어깨-몸통 움직임 축 일치, 불안정
    3   # 'type3' : 머리-어깨-몸통 움직임 축 일치, 고정 & 안정
]


class Curriculum:
    def __init__(self, config, name):
        # 커리큘럼 정보
        self.name    : str  = name

        # 과목, 레벨 정보
        self.subjects: dict = { subject:subject_id 
                               for subject_id, subject in enumerate(config) }
        self.levels  : dict = { self.subjects[subject]:level 
                               for subject, level in config.items() }
        
        # 각 과목과 레벨에 대한 고유 id 지정
        self.table   : dict = {}
        self.courses : list = []

        course_id = 0
        for subject, levels in config.items():
            for lv in levels:
                self.table[(subject, lv)] = course_id
                self.table[course_id] = (subject, lv)
                self.courses.append(course_id)

    # 과목, 레벨 정보 관련 메서드
    def get_subjects(self):
        return self.subjects.keys()
    
    def to_subject_id(self, subject):
        return self.subjects[subject]
    
    def get_levels(self, subject):
        return self.levels[self.to_subject_id(subject)]
    
    # 코스 정보 관련 메서드
    def get_courses(self):
        return self.courses
    
    def to_subject_and_level(self, course):
        return self.table[course]
    
    def to_course(self, subject, level):
        return self.table[(subject, level)]


def warm_samlpling(curriculum, n_users, n_samplies):
    sample = pd.DataFrame()
    subjects = list(curriculum.get_subjects())
    
    for _ in tqdm(range(n_samplies)):
        user_index = randint(0, n_users-1)
        
        # 'Time' Column depends on 'Solved'.
        solved = randint(0, 1) == 1
        time = randint(MIN_TIME, MAX_TIME+1)
        if not solved:
            time = randint(21, MAX_TIME+1)
        
        bl = randint(1, 7)
        hm = randint(1, 8)
        
        subject = subjects[randint(0, len(subjects)-1)]
        levels = curriculum.get_levels(subject)

        # Data generating.
        new_row = {
            'user':[User_IDs[user_index]],
            'age':[Age[user_index]],
            'gender':[Gender[user_index]],
            'disability':[Disability[user_index]],
            'subject':[subject],
            'level':[levels[randint(0, len(levels)-1)]],
            'solved':[int(solved)],
            'time':[time],
            'bl':[bl],
            'hm':[hm],
            'stabiliy':[Stability[randint(0, len(Stability)-1)]],
        }
        sample = pd.concat([sample, pd.DataFrame(new_row)])
    
    return sample.reset_index().drop(['index'], axis=1)


def generate_unique_user(banned):
    genders = [0, 1]
    disabilities = [0, 1, 2, 3, 4]
    while True:
        user_number = random.randint(1, 9999)
        user_name = f"USER{user_number:04d}"
        new_user = {
            "user": user_name,
            "age": randint(18, 80),
            "gender": choice(genders),
            "disability": choice(disabilities)
        }

        if new_user["user"] not in banned:
            banned.add(new_user["user"])
            return new_user


def cold_sampling(N_USERS, banned):
    data_list = []
    for _ in tqdm(range(N_USERS)):
        random_user = generate_unique_user(banned)
        data_list.append(random_user)
    df = pd.DataFrame(data_list)
    return df


if __name__ == '__main__':
    curr = Curriculum(config, '다감각')
    warm_samlpling(curr, N_USER, N_SAMPLIES).to_csv(WARM_USERS_PATH, index=False)
    print("csv file saved at './data/warm_users.csv'")

    warm_users_df = pd.read_csv(WARM_USERS_PATH)
    warm_users = set(warm_users_df['user'].unique())
    cold_sampling(N_COLD_USERS, warm_users).to_csv(COLD_USERS_PATH, index=False)
    print("csv file saved at './data/cold_users.csv'")
