import matplotlib.pyplot as plt
import numpy as np
with open("Performance.txt") as f:
    lines = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
lines = [x.strip() for x in lines]
lines = [x.split() for x in lines]

labels = [x[0] for x in lines]
times = [float(x[1]) for x in lines]


def plot_suffixes(suffixes):
    _times = np.zeros(len(suffixes))
    for label_id in range(len(labels)):
        for suffix_id in range(len(suffixes)):
            if labels[label_id].endswith(suffixes[suffix_id]):
                _times[suffix_id]+=times[label_id]/1000


    ind=np.arange(len(suffixes))
    fig, ax = plt.subplots()
    ax.bar(ind,_times)
    ax.set_ylabel('time(ms)')
    ax.set_title('Execution time per type')
    ax.set_xticks(ind)
    ax.set_xticklabels(suffixes)



suffixes = ["expand","dwise","linear","preprocessing"]
suffixes2 = ["depth_Parrallel","group_Parrallel"]
plot_suffixes(suffixes)
plot_suffixes(suffixes2)

print("Total time : " + str(np.sum(times)/1000) + " ms")
plt.show()

