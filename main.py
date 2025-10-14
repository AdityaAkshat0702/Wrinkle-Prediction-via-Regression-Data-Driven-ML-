import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import joblib  # For saving PCA
import time

# 1. Load the dataset
print("Loading dataset...")
dataset = pd.read_csv('wrinkle_dataset.csv')

# 2. Separate inputs (X) and outputs (Y)
# X is the first 3 columns ('bend_angle', 'frequency', 'phase')
X = dataset.iloc[:, :3].values
# Y is the rest of the 300 columns (the wrinkle coordinates)
Y = dataset.iloc[:, 3:].values

# 3. Split data into training and testing sets
# We'll train the model on 80% of the data and test it on the remaining 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Data loaded and split successfully.")
print("Shape of Y_train (original):", Y_train.shape)
# 4. Apply PCA
print("\nApplying PCA to simplify the data...")

# Initialize PCA to keep 95% of the information
pca = PCA(n_components=0.95)

# Fit PCA on the training data and transform it
Y_train_pca = pca.fit_transform(Y_train)

# Use the SAME fitted PCA to transform the test data
Y_test_pca = pca.transform(Y_test)

print("PCA applied successfully.")
print("Number of components kept:", pca.n_components_)
print("Shape of Y_train_pca (simplified):", Y_train_pca.shape)

# Generate PCA plots like in the paper (Figure 3a, 3b)
# Figure 3a: Cumulative explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(range(1, len(cum_var)+1), cum_var)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.savefig('cum_var.png')
plt.close()

# Figure 3b: Reconstruction error vs. number of components
reconstruction_errors = []
max_k = 20  # Test up to 20 components
for k in range(1, max_k + 1):
    pca_k = PCA(n_components=k)
    Y_train_pca_k = pca_k.fit_transform(Y_train)
    Y_recon = pca_k.inverse_transform(Y_train_pca_k)
    error = np.mean(np.square(Y_train - Y_recon))
    reconstruction_errors.append(error)

plt.figure()
plt.plot(range(1, max_k + 1), reconstruction_errors)
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs. Number of Components')
plt.savefig('recon_error.png')
plt.close()

# --- Prepare Data for PyTorch ---
# Our X data needs to be reshaped to be a 2D array
if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)
if X_test.ndim == 1:
    X_test = X_test.reshape(-1, 1)

# Convert all data to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_pca, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_pca, dtype=torch.float32)

# --- Ablation Study: Test Multiple Architectures ---
input_size = X_train.shape[1]   # 3
output_size = Y_train_pca.shape[1] # e.g., 6

def build_model(hidden_layers):
    layers = []
    prev_size = input_size
    for size in hidden_layers:
        layers.append(nn.Linear(prev_size, size))
        layers.append(nn.ReLU())
        prev_size = size
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)

architectures = [
    [],            # Arch 0: Linear regression baseline (no hidden layers)
    [64, 128],     # Arch 1: Deeper with more neurons
    [16, 32],      # Arch 2: Shallower with fewer neurons
    [32, 64, 128], # Arch 3: Extra layer
    [64]           # Arch 4: Skip hidden layer 2
]

results = {}
for i, hidden in enumerate(architectures):
    print(f"\n--- Testing Architecture {i}: {'Linear' if not hidden else '-'.join(map(str, hidden))} ---")
    model = build_model(hidden)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    # Train
    for epoch in range(1000):
        predictions = model(X_train_tensor)
        loss = loss_function(predictions, Y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = loss_function(test_predictions, Y_test_tensor)
        results[f'Arch {i}'] = test_loss.item()
    print(f'Final Test Loss for Arch {i}: {test_loss.item():.4f}')

print("\nAblation Study Results:", results)

# Find and retrain the best architecture
best_arch = min(results, key=results.get)
best_arch_idx = int(best_arch.split()[1])  # Extract index from "Arch X"
best_hidden_layers = architectures[best_arch_idx]

print(f"\n🏆 Best Architecture: {best_arch} with structure {best_hidden_layers if best_hidden_layers else 'Linear'}")
print(f"Best Test Loss: {results[best_arch]:.6f}")

# Retrain the best model for saving and visualization
print(f"\n--- Retraining Best Architecture ({best_arch}) for Final Use ---")
model = build_model(best_hidden_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

for epoch in range(1000):
    predictions = model(X_train_tensor)
    loss = loss_function(predictions, Y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f'Best Model Epoch [{epoch+1}/1000], Loss: {loss.item():.6f}')

# --- Procedural Baseline Runtime ---
print("\n--- Procedural Baseline Runtime ---")
grid_size = 10
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x, y)
bend_angle = 0.15
frequency = 10.0
phase = 0.0

start_time = time.time()
for _ in range(1000):  # Simulate time steps
    z = bend_angle * np.sin(frequency * xx + phase)
    phase += 0.01
proc_runtime = time.time() - start_time
print(f"Procedural Simulation Runtime: {proc_runtime:.6f} seconds")

# --- NN Runtime ---
start_time = time.time()
test_predictions = model(X_test_tensor)  # Assuming last model
nn_runtime = time.time() - start_time
print(f"Neural Network Prediction Runtime: {nn_runtime:.6f} seconds")

# Save model and PCA (using the last model for simplicity)
torch.save(model.state_dict(), 'wrinkle_model.pth')
joblib.dump(pca, 'pca_transformer.pkl')
print("Model and PCA saved successfully.")

# --- Evaluate the Model ---
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = loss_function(test_predictions, Y_test_tensor)
    print(f'\nFinal Loss on Test Data: {test_loss.item():.4f}')

# --- Visualize a Sample Prediction ---
# Let's pick a more interesting sample (not just the first one)
sample_idx = 5  # Try a different sample
original_pca = Y_test_tensor[sample_idx].unsqueeze(0)
predicted_pca = test_predictions[sample_idx].unsqueeze(0)

# Also get the input parameters for this sample to understand what we're predicting
sample_params = X_test[sample_idx]
print(f"\nSample Parameters - Bend Angle: {sample_params[0]:.4f}, Frequency: {sample_params[1]:.4f}, Phase: {sample_params[2]:.4f}")

original_wrinkle = pca.inverse_transform(original_pca.numpy())
predicted_wrinkle = pca.inverse_transform(predicted_pca.numpy())

grid_size = 10
original_vertices = original_wrinkle.reshape(-1, 3)
predicted_vertices = predicted_wrinkle.reshape(-1, 3)

original_z = original_vertices[:, 2]
predicted_z = predicted_vertices[:, 2]

original_z_grid = original_z.reshape(grid_size, grid_size)
predicted_z_grid = predicted_z.reshape(grid_size, grid_size)

# Calculate correlation between actual and predicted
correlation = np.corrcoef(original_z.flatten(), predicted_z.flatten())[0, 1]
mse = np.mean(np.square(original_z - predicted_z))
print(f"Correlation between actual and predicted: {correlation:.4f}")
print(f"MSE for this sample: {mse:.6f}")

x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x, y)

fig = plt.figure(figsize=(18, 6))

# Plot 1: Actual wrinkle
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(xx, yy, original_z_grid, cmap='viridis', alpha=0.8)
ax1.set_title(f'Actual Wrinkle\n(Bend: {sample_params[0]:.3f}, Freq: {sample_params[1]:.1f})')
ax1.set_zlim(-0.2, 0.2)

# Plot 2: Predicted wrinkle
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(xx, yy, predicted_z_grid, cmap='viridis', alpha=0.8)
ax2.set_title(f'Predicted Wrinkle\n(Correlation: {correlation:.3f})')
ax2.set_zlim(-0.2, 0.2)

# Plot 3: Error visualization
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
error_z_grid = np.abs(original_z_grid - predicted_z_grid)
ax3.plot_surface(xx, yy, error_z_grid, cmap='Reds', alpha=0.8)
ax3.set_title(f'Absolute Error\n(MSE: {mse:.6f})')

plt.tight_layout()
plt.savefig('wrinkle_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Per-Vertex Error and Map (Figure 4) ---
per_vertex_error = np.square(original_z - predicted_z)
avg_per_vertex_error = np.mean(per_vertex_error)
max_per_vertex_error = np.max(per_vertex_error)
print(f"Average Per-Vertex Error: {avg_per_vertex_error:.6f}")
print(f"Max Per-Vertex Error: {max_per_vertex_error:.6f}")

error_map = per_vertex_error.reshape(grid_size, grid_size)
plt.figure()
plt.imshow(error_map, cmap='Reds', interpolation='nearest')
plt.colorbar(label='Squared Error')
plt.title('Per-Vertex Error Map')
plt.savefig('error_map.png')
plt.close()

# --- Histogram of Distances (Figure 2a) ---
distances = np.mean(np.square(Y), axis=1)  # Squared L2 norms
plt.figure()
plt.hist(distances, bins=30)
plt.xlabel('Squared Distance')
plt.ylabel('Number of Examples')
plt.title('Histogram of Squared Distances')
plt.savefig('histogram_distances.png')
plt.close() 