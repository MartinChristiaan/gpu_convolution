import matplotlib.pyplot as plt
import numpy as np
with open("Performance.txt") as f:
    lines = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
lines = [x.strip() for x in lines]
lines = [x.split() for x in lines]

labels = [x[0] for x in lines]
times = [float(x[1]) for x in lines]

def calc_amdahl():
    suffixes = ["Parrallel_dim"]
    dims = np.zeros(len(suffixes))
    n_exec = 0
    for label_id in range(len(labels)):
        for suffix_id in range(len(suffixes)):
            if labels[label_id].endswith(suffixes[suffix_id]):
                dims[suffix_id]+=times[label_id]
                n_exec+=1
    N = dims[0]/(n_exec)
    print("N : " + str(N))

def tp():
    suffixes = ["tp","pre","total"]
    dims = np.zeros(len(suffixes))
    for label_id in range(len(labels)):
        for suffix_id in range(len(suffixes)):
            if labels[label_id].endswith(suffixes[suffix_id]):
                dims[suffix_id]+=times[label_id]/1000
    P = dims[0]/( dims[2])
    print("tp : " + str(P))
    print("pre : " + str(dims[1]))


def plot_suffixes(suffixes):
    _times = np.zeros(len(suffixes))
    num_execs = np.zeros(len(suffixes))
    for label_id in range(len(labels)):
        for suffix_id in range(len(suffixes)):
            if labels[label_id].endswith(suffixes[suffix_id]):
                _times[suffix_id]+=times[label_id]/1000
                num_execs[suffix_id]+=1


    ind=np.arange(len(suffixes))
    plt.figure()
    plt.subplot(3,1,1)
    
    plt.bar(ind,_times)
    plt.ylabel('time(ms)')
    plt.title('Execution time per type')
    plt.xticks(ind,suffixes)
    plt.subplot(3,1,2)

    plt.bar(ind,num_execs)
    plt.ylabel('Number of executions (-)')
    plt.xticks(ind,suffixes)

    plt.subplot(3,1,3)

    plt.bar(ind,_times/num_execs)
    plt.ylabel('tpe (ms)')
    plt.xticks(ind,suffixes)
    print(np.sum(_times))

calc_amdahl()
tp()
#calc_amdahl()
suffixes = ["expand","dwise","linear","preprocessing"]
suffixes2 = ["depth_Parrallel_simple","depth_Parrallel_complex","group_Parrallel"]
plot_suffixes(suffixes)
plot_suffixes(suffixes2)

#print("Total time : " + str(np.sum(times)/1000) + " ms")
plt.show()

