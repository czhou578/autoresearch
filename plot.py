import matplotlib.pyplot as plt

experiments = []
losses = []
best_losses = []

with open('results.tsv', 'r') as f:
    next(f)
    best_loss = float('inf')
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            loss = float(parts[1])
            if loss > 0:
                experiments.append(len(experiments) + 1)
                losses.append(loss)
                if loss < best_loss:
                    best_loss = loss
                best_losses.append(best_loss)

plt.figure(figsize=(10, 6))
plt.plot(experiments, best_losses, marker='o', linestyle='-', color='b', label='Best Loss')
plt.plot(experiments, losses, marker='x', linestyle='--', color='r', alpha=0.5, label='Run Loss')
plt.title('Validation Loss vs Experiments')
plt.xlabel('Experiment Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('sample_plot.png')
plt.savefig('plot.png')
print("Plot saved to plot.png and sample_plot.png")
