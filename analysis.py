import os
import numpy as np
from XMC.loaders import LibSVMLoader
from matplotlib import pyplot as plt

data_root= "Eurlex/"
train_filename = "eurlex_train.txt"
train_file_path = os.path.join("/home/shrutimoy.das/Extreme_Classification/Dataset/",data_root, train_filename)

#train_subsample_filename = "amazonCat_ss_train.txt"
#train_subsample_file_path = os.path.join("/home/shrutimoy.das/Extreme_Classification/Dataset/",data_root, train_subsample_filename)


sample_size = 15539         #1186239    #508542  #(amazonCat)


# this part stores the subsampled matrices

"""
with open(train_file_path, 'r') as fin:
    data = fin.read().splitlines(True)

rand_samples = np.random.shuffle(data) # shuffle the data
#print(type(rand_samples))




with open(train_subsample_file_path, 'w') as fout:
    fout.writelines(data[:sample_size])  # keep only the top "sample_size" data
"""
 
#dset_opts = {'train_opts' : {'num_data_points' : sample_size, 'input_dims' : 203882, 'output_dims' : 13330}} # amazonCat
dset_opts = {'train_opts' : {'num_data_points' : sample_size, 'input_dims' : 5000, 'output_dims' : 3993}} # Eurlex

train_loader = LibSVMLoader(train_file_path, dset_opts['train_opts'])

#print(type(train_loader))
#print(train_loader.features.shape)
X = train_loader.features
Y = train_loader.classes

print(X.shape)
print(Y.shape)


Y_sum = np.squeeze(np.asarray(Y.sum(axis=0)))
Y_sum = np.sort(Y_sum)[::-1]
#Y_sum_log = np.log(Y_sum)
print(Y_sum[2000:2010])

"""
# This is for checking if all labels are annotated
zero_indices = [] # to store the labelsnot present in the dataset
for i in range(Y_sum.size):
	if Y_sum[i] == 0:
		#print("i : ", i, " sum : ", Y_sum[i])
		zero_indices.append(i)

print("zero indices : ", zero_indices[0:10])
print("Size of zero indices : ", len(zero_indices))
"""

fig = plt.figure()

#plt.plot(Y_sum_log)
plt.plot(Y_sum)
plt.xlabel("Label Indices")
plt.ylabel("# samples")
plt.suptitle('Label distribution for Eurlex-4K')
#plt.show()
plt.savefig('label_dist_eurlex.png')


"""
os.rename('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/train_file.txt', \
	'/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/deliciousLarge_train.txt')
"""

"""
with open('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/deliciousLarge_test.txt', 'r') as fin:
    data = fin.read().splitlines(True)
with open('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/test_file.txt', 'w') as fout:
    fout.writelines(data[1:])

os.rename('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/test_file.txt', \
	'/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/deliciousLarge_test.txt')
"""