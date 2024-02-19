class Curriculum:
    def __init__(self, config, name):
        # Save curriculum informations
        self.name: str = name
        
        # Assign a unique interaction `(subject, level)` to  `course: int`
        self.encode_table: dict = {}
        self.decode_table: dict = {}
        self.courses     : list = []
        self.course_id = 0
        for subject, levels in config.items():
            for lv in levels:
                self.encode_table[(subject, lv)] = self.course_id
                self.decode_table[self.course_id] = (subject, lv)
                self.courses.append(self.course_id)
                self.course_id += 1

    def encode(self, subject: str, level: int) -> int:
        return self.encode_table[(subject, level)]
    
    def decode(self, course: int) -> (str, int):
        return self.decode_table[course]

    def get_courses(self):
        return self.courses
    
    def get_courses_num(self):
        return self.course_id
    
    
class UserTable:
    def __init__(self, series):
        self.encode_table = {}
        self.decode_table = {}
        self.users = list(series.unique())
        for user_id, user in enumerate(self.users):
            self.encode_table[user] = user_id
            self.decode_table[user_id] = user            

    def encode(self, user):
        return self.encode_table[user]

    def decode(self, user_id):
        return self.decode_table[user_id]
    
    def get_users(self):
        return self.users
    
    def get_users_num(self):
        return len(self.users)
    
    def get_random_user(self):
        from random import randint
        users = self.get_users()
        return users[randint(0, len(users) - 1)]


def preprocess(df, curriculum, user_table):
    user_ids = df.apply(lambda row: user_table.encode(row['user']), axis=1)
    items = df.apply(lambda row: curriculum.encode(row['subject'], row['level']), axis=1)
    ratings = (df['bl'] + df['hm'] + 1) / 2

    df.drop(columns=['user','subject','level','bl','hm'], inplace=True)

    df.insert(0, 'user_id', user_ids)
    df.insert(1, 'item', items)
    df.insert(2, 'rating', ratings)
    return df
