import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, embed_size=64):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.item_embedding = nn.Embedding(num_items, embed_size)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        element_product = torch.mul(user_embed, item_embed)
        return element_product


class MLP(nn.Module):
    def __init__(self, num_users, num_items, embed_size=64, hidden_size=64):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.item_embedding = nn.Embedding(num_items, embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        concat_embed = torch.cat([user_embed, item_embed], dim=-1)
        mlp_output = self.mlp(concat_embed)
        return mlp_output


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_size=64, hidden_size=64):
        super(NCF, self).__init__()
        self.gmf = GMF(num_users, num_items, embed_size)
        self.mlp = MLP(num_users, num_items, embed_size, hidden_size)
        self.final_layer = nn.Linear(embed_size * 2, 1)

    def forward(self, user, item):
        gmf_output = self.gmf(user, item)
        mlp_output = self.mlp(user, item)
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        final_output = self.final_layer(concat_output)
        return final_output.squeeze()
