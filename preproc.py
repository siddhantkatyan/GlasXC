import os



with open('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/deliciousLarge_train.txt', 'r') as fin:
    data = fin.read().splitlines(True)
with open('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/train_file.txt', 'w') as fout:
    fout.writelines(data[1:])

os.rename('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/train_file.txt', \
	'/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/deliciousLarge_train.txt')

with open('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/deliciousLarge_test.txt', 'r') as fin:
    data = fin.read().splitlines(True)
with open('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/test_file.txt', 'w') as fout:
    fout.writelines(data[1:])

os.rename('/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/test_file.txt', \
	'/home/shrutimoy.das/Extreme_Classification/Dataset/DeliciousLarge/deliciousLarge_test.txt')