import nbformat as nbf
import pandas as pd
import json

nb = nbf.v4.new_notebook()

text = """\
# Validation Loss vs Number of Experiments
This notebook plots the validation loss across all the experiments carried out in Phase 1 and Phase 2.
"""

code = """\
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results.tsv', sep='\\t')
df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
df.dropna(subset=['loss'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Add experiment numbers
df['experiment_number'] = range(1, len(df) + 1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['experiment_number'], df['loss'], marker='o', linestyle='-', color='b', label='Validation Loss')

# Highlight best models (Kept)
kept = df[df['status'] == 'keep']
plt.scatter(kept['experiment_number'], kept['loss'], color='g', zorder=5, s=100, label='Kept (New Best / Baseline)')

# Highlight discarded models
discarded = df[df['status'] == 'discard']
plt.scatter(discarded['experiment_number'], discarded['loss'], color='r', zorder=5, s=100, marker='x', label='Discarded')

plt.title('Validation Loss vs Number of Experiments')
plt.xlabel('Experiment Number')
plt.ylabel('Validation Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('loss_plot_final.png')
plt.show()

# Show dataframe
df[['experiment_number', 'loss', 'status', 'description']]
"""

nb['cells'] = [nbf.v4.new_markdown_cell(text), nbf.v4.new_code_cell(code)]

with open('plot_loss.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook plot_loss.ipynb created successfully.")
