import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.metrics.pairwise import cosine_similarity
from config import *


class Similarity:
    def __init__(self, warms, colds, scaler):
        """
        Initializes the Similarity class.
        
        Args:
        - warms (pd.DataFrame): DataFrame containing warm users' information.
        - colds (pd.DataFrame): DataFrame containing cold users' information.
        - scaler: Scaler object used for normalization.
        """
        # Category config
        self.disability = [0,1,2,3,4]
        self.disabilities_categories = pd.Categorical(self.disability)

        # Raw data
        self.warms = deepcopy(warms)
        self.colds = deepcopy(colds)

        # Normalized and One-hot encoding
        self.normal_warms, self.normal_colds, self.fitted_scaler = self.normalize(warms, colds, scaler=scaler, init=True)

        # Similarity data
        self.similarity_df = pd.DataFrame(
            cosine_similarity(self.normal_warms, self.normal_colds), 
            index=self.normal_warms.index, 
            columns=self.normal_colds.index
        )

    def normalize(self, warms, colds, scaler, init=True):
        """
        Normalizes the data and performs one-hot encoding.
        
        Args:
        - warms (pd.DataFrame): DataFrame containing warm users' information.
        - colds (pd.DataFrame): DataFrame containing cold users' information.
        - scaler: Scaler object used for normalization.
        - init (bool): Whether it's the initial normalization process or not.
        
        Returns:
        - Tuple of pd.DataFrames and scaler: Normalized warm DataFrame, normalized cold DataFrame, and fitted scaler.
        """
        if init:
            warms.set_index('user', inplace=True)
            scaler.fit(warms[['age']])
            warms['age'] = scaler.transform(warms[['age']])
            warms = pd.get_dummies(warms, columns=["disability"], dtype=int)

        colds.set_index('user', inplace=True)
        colds['age'] = scaler.transform(colds[['age']])
        colds = pd.get_dummies(colds, columns=["disability"], dtype=int)
    
        for category in self.disabilities_categories:
            if f'disability_{category}' not in colds.columns:
                colds[f'disability_{category}'] = 0
        return warms, colds, scaler
    
    def get_cold_users(self):
        """
        Get cold users' information.
        
        Returns:
        - np.array: Array containing cold users' IDs.
        """
        return np.array(self.colds['user'])
    
    def similar_users(self, user):
        """
        Find users similar to the given user.
        
        Args:
        - user (str): ID of the user.
        
        Returns:
        - list of tuples: List of tuples containing similarity scores and user IDs.
        """
        sorted_df = self.similarity_df[user].sort_values(ascending=False)
        users_ranking = [(sorted_df.loc[idx], idx) for idx in sorted_df.index]
        return users_ranking
    
    def __add_row_to_csv(self, file_path, user: pd.DataFrame):
        """
        Append a row to a CSV file.
        
        Args:
        - file_path (str): Path to the CSV file.
        - user (pd.DataFrame): DataFrame containing user information.
        """
        seek = map(str, user.iloc[0].values)
        new_row = ','.join(seek) + '\n'
        with open(file_path, 'a') as file:
            file.write(new_row)
    
    def append_cold_user(self, user_info: dict) -> None:
        """
        Append a cold user to the system.
        
        Args:
        - user_info (dict): Dictionary containing user information.
        """
        # CSV update
        self.__add_row_to_csv(COLD_USERS_PATH, user_info)
        
        # Normalize
        user = pd.DataFrame(user_info)
        _, normal_user, _ = self.normalize(None, deepcopy(user), scaler=self.fitted_scaler, init=False)

        # Update class variables
        self.colds = pd.concat([self.colds, user])

        # Similarity class data update
        self.normal_colds = pd.concat([self.normal_colds, normal_user])
        self.similarity_df[user_info['user']] = cosine_similarity(self.normal_warms, normal_user)

    def cvt_cold_to_warm(self, user: str) -> None:
        """
        Convert a cold user to a warm user.
        
        Args:
        - user (str): ID of the user.
        """
        target = self.colds[self.colds['user'] == user]

        # Update class variable
        self.warms = pd.concat([self.warms, target])
        condition = self.colds['user'] == user
        self.colds = self.colds[~condition]

        # CSV update
        self.colds.to_csv(COLD_USERS_PATH, index=False)
        self.__add_row_to_csv(WARM_USERS_PATH, target)

        # Similarity class data update
        normal_user = pd.DataFrame(self.normal_colds.loc[user]).T
        self.normal_warms = pd.concat([self.normal_warms, normal_user])
        self.normal_colds.drop(index=[user], inplace=True)
        self.similarity_df.drop(columns=[user], inplace=True)
        tmp_similarity_df = pd.DataFrame(
            cosine_similarity(normal_user, self.normal_colds), 
            index=normal_user.index, 
            columns=self.normal_colds.index
        )
        self.similarity_df = pd.concat([self.similarity_df, tmp_similarity_df])
        