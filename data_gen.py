import numpy as np
import pandas as pd
# --- Parameters for our dataset ---
NUM_SAMPLES = 2000  # How many examples to generate
GRID_SIZE = 10      # The cloth will be a 10x10 grid of points

# --- Create the flat cloth (our base shape) ---

# Create a 10x10 grid of x and y points
x = np.linspace(0, 1, GRID_SIZE)
y = np.linspace(0, 1, GRID_SIZE)
xx, yy = np.meshgrid(x, y)

# Add a z-coordinate of 0 to make it flat, then stack them together
zz = np.zeros_like(xx)
flat_cloth_vertices = np.stack([xx, yy, zz], axis=-1)

# Reshape the grid into a long list of vertices (100 vertices, each with 3 coordinates)
# The shape will be (100, 3)
flat_cloth_vertices = flat_cloth_vertices.reshape(-1, 3)

print("Created a flat cloth with shape:", flat_cloth_vertices.shape)
print("Here are the first 5 vertices:")
print(flat_cloth_vertices[:5])
def apply_wrinkle(vertices, bend_angle, freq, phase):
    """
    Applies a more varied wrinkle pattern to the cloth.
    """
    new_vertices = vertices.copy()
    x = new_vertices[:, 0]

    # Calculate new z-coordinate with varied frequency and phase
    new_z = bend_angle * np.sin(x * freq + phase)

    new_vertices[:, 2] = new_z
    return new_vertices

# --- Generate the full dataset ---
all_inputs = []
all_outputs = []

print("\n--- Generating 2000 varied data samples ---")

for i in range(NUM_SAMPLES):
    # 1. Create random parameters for each sample
    random_bend_angle = np.random.uniform(0.05, 0.2) # smaller, more realistic wrinkles
    random_freq = np.random.uniform(5, 20)          # random wave frequency
    random_phase = np.random.uniform(0, np.pi)      # random wave shift

    # 2. Store the input parameters
    input_row = [random_bend_angle, random_freq, random_phase]
    all_inputs.append(input_row)

    # 3. Generate the wrinkled cloth using these parameters
    wrinkled_vertices = apply_wrinkle(flat_cloth_vertices, random_bend_angle, random_freq, random_phase)

    # 4. Flatten and store the output
    output_row = wrinkled_vertices.flatten()
    all_outputs.append(output_row)

# --- Save the dataset to a CSV file ---

# Create a Pandas DataFrame for the outputs
output_df = pd.DataFrame(all_outputs)

# Create a Pandas DataFrame for the inputs with new column names
input_df = pd.DataFrame(all_inputs, columns=['bend_angle', 'frequency', 'phase'])

# Combine them into one big DataFrame
full_dataset = pd.concat([input_df, output_df], axis=1)

# Save to a CSV file
full_dataset.to_csv('wrinkle_dataset.csv', index=False)

print("Successfully created and saved the new varied wrinkle_dataset.csv!")
print("Dataset shape:", full_dataset.shape)