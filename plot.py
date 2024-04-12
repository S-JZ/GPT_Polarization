import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read cosine_similarity.csv
df = pd.read_csv('cosine_similarity_wrt_center.csv')

custom_order = ['far-left', 'left',  'right', 'far-right']

# Create an empty matrix
matrix = np.zeros((len(custom_order), len(custom_order)))

# Fill the matrix with similarity values
for index, row in df.iterrows():
    row_index = custom_order.index(row['ideology1'])
    col_index = custom_order.index(row['ideology2'])
    if row_index >= col_index:
        matrix[row_index][col_index] = row['similarity']
    # matrix[row_index][col_index] = row['similarity']

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, xticklabels=custom_order, yticklabels=custom_order, cmap="magma",fmt=".2f",annot=True, square=True)
plt.title('Pairwise Cosine Similarity between Bias Vectors with respect to Center')
# plt.show()
plt.savefig('redd_cos_heatmap_wrt_center.png', bbox_inches='tight')