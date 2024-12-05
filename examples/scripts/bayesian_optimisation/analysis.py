import numpy as np

init_calls = []
for i in range(1, 6):
    init_call = []
    with open('rand_init%i.txt' %i, 'r') as rand_file:
        for line in rand_file:
            init_call.append(int(line.split()[-1]))
    init_calls.append(np.sum(np.asarray(init_call)))

bh_calls = []
for i in range(1, 6):
    bh_call = []
    with open('bh%i.txt' %i, 'r') as bh_file:
        for line in bh_file:
            f = int(line.split()[-1])
            g = int(line.split()[-2])
            bh_call.append(f + 15*g)
    bh_calls.append(np.sum(np.asarray(bh_call)))

print("init calls", init_calls)
print("bh calls", bh_calls)

print("BH: ", np.mean(np.asarray(bh_calls)), " +- ", np.std(np.asarray(bh_calls)))
print("Init: ", np.mean(np.asarray(init_calls)), " +- ", np.std(np.asarray(init_calls)))

