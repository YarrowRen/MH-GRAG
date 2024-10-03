import torch
from .loss_functions import modularity_loss_multi_head

def train_model(model, data, adj, loss_fn, optimizer, epochs=1000, print_every=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        loss = loss_fn(embeddings, adj)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    return model

def train_model_multi_head(model, data, adj, optimizer, num_heads, epochs=200, print_every=20):
    # 训练模型
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings_list = model(data.x, data.edge_index)
        loss = modularity_loss_multi_head(embeddings_list, adj, num_heads, orth_lambda=1e-8)
        loss.backward()
        optimizer.step()
        if (epoch+1) % print_every == 0 or epoch == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    return model