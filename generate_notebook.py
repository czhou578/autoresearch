import csv
import matplotlib.pyplot as plt
import nbformat as nbf

def generate():
    losses = []
    statuses = []
    
    with open('results.tsv', 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            losses.append(float(row['loss']))
            statuses.append(row['status'])
            
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Validation Loss')
    
    for i, (loss, status) in enumerate(zip(losses, statuses)):
        if status == 'crash':
            plt.scatter(i + 1, loss, color='red', marker='x', s=100, label='Crash' if i == 0 else "")
            
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    plt.xlabel('Experiment Number')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Experiments')
    plt.grid(True)
    if by_label:
        plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig('sample_plot.png')
    
    nb = nbf.v4.new_notebook()
    markdown_cell = nbf.v4.new_markdown_cell("# Autoresearch Results\\nPlot of validation loss over iterations.")
    code_cell = nbf.v4.new_code_cell("""import csv
import matplotlib.pyplot as plt

losses = []
statuses = []
with open('results.tsv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        losses.append(float(row['loss']))
        statuses.append(row['status'])

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Validation Loss')

for i, (loss, status) in enumerate(zip(losses, statuses)):
    if status == 'crash':
        plt.scatter(i + 1, loss, color='red', marker='x', s=100)
        
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.xlabel('Experiment Number')
plt.ylabel('Validation Loss')
plt.title('Validation Loss over Experiments')
plt.grid(True)
if by_label:
    plt.legend(by_label.values(), by_label.keys())
plt.show()""")
    
    nb['cells'] = [markdown_cell, code_cell]
    with open('experiment_results.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    generate()
    print("Notebook and plot generated.")
