import pandas as pd
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import *
from echo import performance_decorator


def generate(model_class, params, user_ids, item_ids):
    num_users = user_ids.max().item() + 1
    num_items = item_ids.max().item() + 1
    model = model_class(
        num_users=num_users, 
        num_items=num_items,
        embed_size=list(params.values())[0],
        hidden_size=list(params.values())[1]
    )
    return model


@performance_decorator
def fit(model, params, user_ids, item_ids, ratings):
    # Parameters unpacking
    _, _, learning_rate, num_epochs, batch_size = params.values()

    # Loss function & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(num_epochs):
        dataset = TensorDataset(user_ids, item_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_user, batch_item, batch_rating in tqdm(dataloader):
            outputs = model(batch_user, batch_item)
            loss = criterion(outputs, batch_rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model
