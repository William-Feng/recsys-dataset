"""
Create word2vec graph containing hyperparameter tuning scores
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

fig = plt.figure()
ax = plt.axes()

# Window size scores
x1 = [0.62910673152698, 0.6278955343262882, 0.6260999868751859]
# Epoch scores
x2 = [0.6138698652251765, 0.627628607104487, 0.6286526004642453]
# Negative exponent score
x3 = [0.6298949912087992, 0.6298498788863978, 0.6301857112062177]

# Window size
y1 = [1, 5, 10]
# Epochs
y2 = [1, 5, 10]
# Negative exponent
y3 = [-0.5, 0.5, 0.75]

plt.rcParams.update({"figure.figsize": (10, 8), "figure.dpi": 100})
plt.scatter(
    x1, y1, label=f"Window Size Correlation = {np.round(np.corrcoef(x1, y1)[0, 1], 2)}"
)
plt.scatter(
    x2, y2, label=f"Epoch Correlation = {np.round(np.corrcoef(x1, y2)[0, 1], 2)}"
)
plt.scatter(
    x3,
    y3,
    label=f"Negative Sampling Exponent Correlation = {np.round(np.corrcoef(x1, y3)[0, 1], 2)}",
)

# Adding trend lines
fit1 = np.polyfit(x1, y1, 1)
plt.plot(
    x1,
    np.polyval(fit1, x1),
    linestyle="-",
    color="b",
    alpha=0.6,
    label="Window Size Trend Line",
)

fit2 = np.polyfit(x2, y2, 1)
plt.plot(
    x2,
    np.polyval(fit2, x2),
    linestyle="-",
    color="r",
    alpha=0.6,
    label="Epoch Trend Line",
)

fit3 = np.polyfit(x3, y3, 1)
plt.plot(
    x3,
    np.polyval(fit3, x3),
    linestyle="-",
    color="g",
    alpha=0.6,
    label="Negative Sampling Exponent Trend Line",
)

plt.title("The relationship between Word2Vec hyperparameter values and score")
plt.xlabel("Hyperparameter test value")
plt.ylabel("Total Score")

plt.xlim(0.6125, 0.631)
plt.ylim(-0.7, 10.5)
plt.legend()
plt.show()
