import random
import matplotlib.pyplot as plt

# List of total scores extracted from each line
xx = []
yy = []
legends = []
switch = False
# Read total scores from the file names
starts_with = "0.0"

with open("process/XGBoost/output.log", 'r') as file:
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
plt.xlabel('File Index')
plt.ylabel('Total Score')
plt.title('Total Score vs. File Index')
plt.legend(legends)
# plt.xticks(range(len(file_names)), file_names, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
