import pandas as pd
import numpy as np

def calculate_neuron_output(csv_file):
    # 1. TODO: Load the data from csv_file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # 2. TODO: Convert the DataFrame to a NumPy array (X)
    X = df.to_numpy()
    
    # 3. Define Weights (W) and Bias (b)
    W = np.array([0.5, -0.2])
    b = 1.0
    
    # 4. TODO: Calculate z = XW + b 
    # Hint: Use the  .dot() method for multiplication
    z = (X.dot(W)) + b
    
    # 5. TODO: Apply Step Activation
    # Return 1 if z > 0, else 0
    z = (z > 0).astype(int)
    
    return X, z
