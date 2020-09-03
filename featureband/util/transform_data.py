from featureband.util.data_util import load_dataset

DATASET = "madelon"  # ["madelon", "basehock", "usps", "coil20"]

x, y = load_dataset(DATASET)

data_file = "../InfoFeatureSelection/data/" + DATASET + ".txt"
label_file = "../InfoFeatureSelection/data/" + DATASET + "_labels.txt"

with open(data_file, 'w') as fout:
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            fout.write(str(x[i][j]))
            fout.write('\t')
        fout.write('\n')

with open(label_file, 'w') as fout:
    for i in range(y.shape[0]):
        label = int(y[i])
        if label == -1:
            fout.write("0")
        else:
            fout.write(str(label))
        fout.write('\n')
