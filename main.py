# Import standard libraries
import os
import argparse

# Import third-party libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import custom modules
from config import *
import echo
from schema import Curriculum, UserTable, preprocess

# Import modules related to coldstart
from model.coldstart.similarity import Similarity
import model.coldstart.predict as cold_pred

# Import modules related to warmstart
from model.warmstart.network import NCF
from model.warmstart.train import generate, fit
import model.warmstart.predict as warm_pred


def main():
    # Load Data
    warms = pd.read_csv(WARM_USERS_PATH)
    colds_user_info = pd.read_csv(COLD_USERS_PATH)
    warms_user_info = warms[['user','age','gender','disability']].drop_duplicates(subset='user')

    # Construct Class Instances
    curriculum = Curriculum(config, '다감각')
    user_table = UserTable(warms['user'])

    # Preprocess Data
    warms = preprocess(warms, curriculum, user_table)
    
    # ================= Cold-START ================= #
    similarity = Similarity(warms_user_info, colds_user_info, scaler=StandardScaler())
    
    # Calculate similarity & find similar users
    random_user = np.choose(1, similarity.get_cold_users())
    similar_users = similarity.similar_users(random_user)

    # Score and Rank courses which was learned by similar_users
    courses_ranking = cold_pred.topk(similar_users, warms, user_table, K)
    echo.display_leader_board(courses_ranking, f" Leader board for cold-start user, {random_user}")

    # Convert ranking to subject, level data for recommendation
    recommended_courses = cold_pred.recommend(courses_ranking, curriculum)
    echo.display_courses(recommended_courses)
    # ============================================== #

    # ================= Warm-START ================= #
    nonzero_user_item_pairs = torch.LongTensor(list(zip(warms['user_id'].to_list(), warms['item'].to_list())))
    nonzero_ratings = torch.tensor(warms['rating'], dtype=torch.float32)
    
    # Create the sparse COO tensor
    sparse_coo_tensor = torch.sparse_coo_tensor(
        nonzero_user_item_pairs.t(),
        nonzero_ratings,
        torch.Size([user_table.get_users_num(), curriculum.get_courses_num()])
    )

    # Convert the sparse coordinates tensor to a dense tensor
    interaction_matrix = sparse_coo_tensor.to_dense()

    # Plot
    plt.subplots()[1].matshow(interaction_matrix, cmap='summer')
    plt.show()

    # Tensor for pytorch model training
    user_ids = []
    item_ids = []
    ratings = []
    for (user_id, item_id), rating in np.ndenumerate(interaction_matrix):
        user_ids.append(user_id)
        item_ids.append(item_id)
        ratings.append(rating)
    user_ids = torch.tensor(user_ids)
    item_ids = torch.tensor(item_ids)
    ratings = torch.tensor(ratings)

    # Set parameter '--load {model_name.pt}' 
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--load', type=str, help='Load model from a file')
    args = parser.parse_args()
    
    # Load or Save model
    ncf_model = generate(
        model_class=NCF, 
        params=PARAMS,
        user_ids=user_ids, 
        item_ids=item_ids
    )

    if args.load:
        # Load existing model
        try:
            model_path = LOAD_FOLDER + args.load
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            ncf_model.load_state_dict(torch.load(model_path))
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading the model: {e}")
            exit(1)
    else:
        # Create new model
        ncf_model = fit(
            model=ncf_model, 
            params=PARAMS, 
            user_ids=user_ids, 
            item_ids=item_ids, 
            ratings=ratings
        )
        # And save it
        model_path = warm_pred.create_model_path(LOAD_FOLDER, LOAD_NAME)
        torch.save(ncf_model.state_dict(), model_path)
        print(f"Model saved at {model_path}\n")

    # predict
    with torch.inference_mode():
        target_user = user_table.get_random_user()
        target_user_id = user_table.encode(target_user)
        all_courses = curriculum.get_courses()

        covered_ranking, uncovered_ranking = warm_pred.topk(warms, ncf_model, target_user_id, all_courses, K)
        echo.display_leader_board(covered_ranking, title=f"Covered Courses Leader-board for {target_user}({target_user_id})")
        echo.display_leader_board(uncovered_ranking, title=f"Uncovered Courses Leader-board for {target_user}({target_user_id})")

        topK_not_in_history = warm_pred.recommend(curriculum, uncovered_ranking)
        echo.display_courses(topK_not_in_history)
    # ============================================== #
    

if __name__ == '__main__':
    main()
    