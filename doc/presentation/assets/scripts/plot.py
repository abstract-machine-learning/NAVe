import sys
from pylab import *

name = sys.argv[1]

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4.5, 4.5]
}
rcParams.update(params)
axes(frameon=0)
grid()
xlabel("epsilon")
ylabel("stability %")
title(name.replace("-", " ").title())

k = [1, 3, 5, 7]
abstractions = ["interval", "raf"]

for abstraction in abstractions:
    data = np.loadtxt(name + "-" + abstraction + ".dat")
    for i in range(1, data.shape[1]):
        plot(data[:, 0], data[:, i], label="k = " + str(k[i - 1]) + ", " + abstraction)
legend()
ylim(0, 100)
savefig(name + ".pdf")
