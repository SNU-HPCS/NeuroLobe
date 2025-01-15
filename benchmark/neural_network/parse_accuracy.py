from os.path import exists

models=["FC", "RNNTANH", "SlayerFCSNN"]
bins=[1,2,4,8,16,32,64,128]
learning_rates=["1e-2","1e-3","1e-4"]
batches = [1, 64]
gaussian = [0, 1]


accuracy_dat = {}
for model in models:
    accuracy_dat[model] = {}
    for bin in bins:
        accuracy_dat[model][bin] = {}
        for gauss in gaussian:
            accuracy_dat[model][bin][gauss] = None

for model in models:
    for bin in bins:
        for lr in learning_rates:
            for batch in batches:
                for gauss in gaussian:
                    name = str(model) + \
                           "_bin_" + str(bin) + \
                           "_gaussian_" + str(gauss) + \
                           "_batch_" + str(batch) + \
                           "_lr_" + str(lr) + \
                           ".txt"
                    if not exists(name): continue
                    f = open(name)
                    lines = f.readlines()
                    best_accuracy = -1
                    for ind in range(len(lines)):
                        if "BEST" in lines[ind]:
                            accuracy = float(lines[ind-1].rstrip().split("R2: ")[-1])
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy

                    if not accuracy_dat[model][bin][gauss] or accuracy_dat[model][bin][gauss] < best_accuracy:
                        accuracy_dat[model][bin][gauss] = best_accuracy

for model in models:
    for bin in bins:
        for gauss in gaussian:
            print(model, bin, gauss, accuracy_dat[model][bin][gauss])
