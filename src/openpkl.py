import pickle

# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
f = open('D:\SRT\data\cub200\processed\test.pkl','rb')
data = pickle.load(f)
print(data)