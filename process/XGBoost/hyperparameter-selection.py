import random
import matplotlib.pyplot as plt

# List of total scores extracted from each line
xx = []
yy = []
legends = []
switch = False
# Read total scores from the file names
starts_with = "0.0"

with open("./output.log", 'r') as file:
    for line in file:
        if line.startswith('Scores'):
            score_str = line.split(':')[5].strip()
            score_str = score_str[:len(score_str)-1]
            yy.append(float(score_str))
        elif line.endswith('.csv\n'):
            if not line.startswith(starts_with):
                starts_with = line.split('_')[0]
                col = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                legends.append(line.split('_')[0])
                plt.plot(xx, yy, marker='o', color=col)
                plt.xlabel("Average Pulse")

                xx = []
                yy = []
                xx.append(float(line.split('_')[1]))
            else:
                xx.append(float(line.split('_')[1]))

plt.plot(xx, yy, marker='o', color=col)
plt.xlabel("'map' weighting")
plt.ylabel("XGBoost Recall Score")
plt.title("Recall score vs weighted 'ndcg', 'map' and 'pairwise'")
plt.legend(legends, title="'pairwise' weighting", prop={'size': 5})

plt.grid(True)
plt.tight_layout()
plt.savefig('hyperparameter-selection')
