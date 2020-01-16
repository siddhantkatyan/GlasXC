with open('/home/siddharth/Documents/eurlex_train.txt', 'r') as fin:
    data = fin.read().splitlines(True)
with open('train_file.txt', 'w') as fout:
    fout.writelines(data[1:])