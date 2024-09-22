import torch

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
