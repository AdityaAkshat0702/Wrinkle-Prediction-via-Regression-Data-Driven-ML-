import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
# --- Define the Neural Network Architecture ---
input_size = X_train.shape[1]   # Should be 3 (bend_angle, frequency, phase)
output_size = Y_train_pca.shape[1] # Should be 6 (the number of PCA components)

model = nn.Sequential(
    nn.Linear(input_size, 32), # Input layer (3 features) -> Hidden layer 1 (32 neurons)
    nn.ReLU(),                 # Activation function
    nn.Linear(32, 64),         # Hidden layer 1 (32) -> Hidden layer 2 (64)
    nn.ReLU(),                 # Activation function
    nn.Linear(64, output_size) # Hidden layer 2 (64) -> Output layer (6 features)
)

print("\n--- Neural Network Architecture ---")
print(model)
# --- Define Loss Function and Optimizer ---
loss_function = nn.MSELoss()  # Mean Squared Error, a good choice for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam is a popular and effective optimizer

# --- Train the Model ---
num_epochs = 1000  # How many times to loop through the data

print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    # 1. Forward pass: Make a prediction
    predictions = model(X_train_tensor)
    
    # 2. Calculate the loss (how wrong the prediction is)
    loss = loss_function(predictions, Y_train_tensor)
    
    # 3. Zero the gradients before the backward pass
    optimizer.zero_grad()
    
    # 4. Backward pass: Calculate the gradients
    loss.backward()
    
    # 5. Update the weights: Adjust the model to get better
    optimizer.step()
    
    # Print the loss every 100 epochs to check progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("--- Training Finished ---")

# --- Evaluate the Model ---
# Set the model to evaluation mode
model.eval()

# We don't need to calculate gradients for evaluation
with torch.no_grad():
    # Make predictions on the test data
    test_predictions = model(X_test_tensor)
    
    # Calculate the loss on the test data
    test_loss = loss_function(test_predictions, Y_test_tensor)
    print(f'\nFinal Loss on Test Data: {test_loss.item():.4f}')

# --- Visualize a Sample Prediction ---
# Take the first sample from our test set
original_pca = Y_test_tensor[0].unsqueeze(0)
predicted_pca = test_predictions[0].unsqueeze(0)

# Inverse transform from PCA space back to original 300 coordinates
original_wrinkle = pca.inverse_transform(original_pca.numpy())
predicted_wrinkle = pca.inverse_transform(predicted_pca.numpy())

# Reshape the 300 coordinates back into 100 vertices of (x, y, z)
grid_size = 10 # Should match the GRID_SIZE from your data_gen script
original_vertices = original_wrinkle.reshape(-1, 3)
predicted_vertices = predicted_wrinkle.reshape(-1, 3)

# Get just the z-coordinate (the wrinkle height) from the third column
original_z = original_vertices[:, 2]
predicted_z = predicted_vertices[:, 2]

# Reshape the z-coordinates into a 10x10 grid for plotting
original_z_grid = original_z.reshape(grid_size, grid_size)
predicted_z_grid = predicted_z.reshape(grid_size, grid_size)

# Create the x, y grid for plotting
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x, y)

# Create the 3D plots
fig = plt.figure(figsize=(12, 6))

# Plot the actual wrinkle
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(xx, yy, original_z_grid, cmap='viridis')
ax1.set_title('Actual Wrinkle')
ax1.set_zlim(-0.2, 0.2) # Keep z-axis scale consistent

# Plot the predicted wrinkle
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(xx, yy, predicted_z_grid, cmap='viridis')
ax2.set_title('Predicted Wrinkle')
ax2.set_zlim(-0.2, 0.2) # Keep z-axis scale consistent

plt.show()