import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
from mpl_toolkits.mplot3d import Axes3D

# --- Load the Saved Model and PCA ---
input_size = 3  # bend_angle, frequency, phase
output_size = 6  # Adjust if your PCA components differ

model = nn.Sequential(
    nn.Linear(input_size, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, output_size)
)

model.load_state_dict(torch.load('wrinkle_model.pth'))
model.eval()

pca = joblib.load('pca_transformer.pkl')

# --- Fixed Parameters for Animation ---
bend_angle = 0.15  # Example fixed value (tweak as needed)
frequency = 10.0   # Example fixed value
grid_size = 10
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x, y)

# --- Animation Function ---
def animate(frame):
    # Vary phase over time (e.g., 0 to 2π in 100 frames)
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
    predicted_z = predicted_vertices[:, 2]  # Extract z (wrinkle height)
    predicted_z_grid = predicted_z.reshape(grid_size, grid_size)
    
    # Update the plot
    ax.clear()
    ax.plot_surface(xx, yy, predicted_z_grid, cmap='viridis')
    ax.set_title(f'Animated Predicted Wrinkle (Phase: {phase:.2f})')
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# --- Set Up the Figure ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# --- Create and Run the Animation ---
ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=False)

plt.show()  # Displays the real-time animation
# Optional: Save as GIF (requires imagemagick or ffmpeg installed)
# ani.save('wrinkle_animation.gif', writer='pillow', fps=20)