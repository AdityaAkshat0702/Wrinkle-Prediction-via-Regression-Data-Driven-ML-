import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
from mpl_toolkits.mplot3d import Axes3D

# --- Load the Saved Model and PCA ---
input_size = 3  # bend_angle, frequency, phase
output_size = 6  # This must match the number of components from main.py

# --- IMPORTANT CHANGE HERE ---
# This model architecture MUST MATCH the one that was saved by main.py.
# The enhanced main.py saves the last architecture tested: [64] (single hidden layer)
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, output_size)
)

# Load the saved weights into the model structure
model.load_state_dict(torch.load('wrinkle_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load the saved PCA transformer
pca = joblib.load('pca_transformer.pkl')

# --- Fixed Parameters for Animation ---
bend_angle = 0.15
frequency = 10.0
grid_size = 10
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x, y)

# --- Set Up the Figure ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# --- Animation Function ---
def animate(frame):
    # Vary phase over time (0 to 2π)
    phase = (frame / 100) * 2 * np.pi
    
    # Prepare input tensor
    input_params = np.array([[bend_angle, frequency, phase]])
    input_tensor = torch.tensor(input_params, dtype=torch.float32)
    
    # Predict PCA components
    with torch.no_grad():
        predicted_pca = model(input_tensor).numpy()
    
    # Inverse transform to get full coordinates
    predicted_wrinkle = pca.inverse_transform(predicted_pca)
    predicted_vertices = predicted_wrinkle.reshape(-1, 3)
    predicted_z = predicted_vertices[:, 2]
    predicted_z_grid = predicted_z.reshape(grid_size, grid_size)
    
    # Update the plot
    ax.clear()
    ax.plot_surface(xx, yy, predicted_z_grid, cmap='viridis')
    ax.set_title(f'Animated Predicted Wrinkle (Phase: {phase:.2f})')
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# --- Create and Run the Animation ---
# frames=101 to get a smooth loop from 0 to 2*pi
ani = FuncAnimation(fig, animate, frames=101, interval=50, blit=False)

plt.show()