# -*- coding: UTF-8 -*-
'''
Created on 2017年5月16日

@author: Conter
'''
import numpy as np


def load_data(file_path):
    prefer = []
    for line in open(file_path, 'r'):  # 打开指定文件
        (userid, movieid, rating, ts) = line.split('\t')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = np.array(prefer)
    return data
def sort_data(data, axis):
    data = data[data[:, axis].argsort()]
    temp = data[0, axis]
    pIndex = 0
    for index, couple in enumerate(data):
        if couple[axis] != temp or index == len(data)-1:
            index = index + 1 if index == len(data)-1 else index
            print pIndex, index
            data[pIndex:index] = data[pIndex : index][data[pIndex : index][:, 0 if axis == 1 else 1].argsort()]
            pIndex = index
            temp += 1
    return data

def write_data(data, axis, file_path):
    data = sort_data(data, axis)
    fw = open(file_path, "w")
    for line in data:
        reslt = ""
        for field in line:
            reslt += "%s" % int(field) + "\t"
        fw.write(reslt[:-1] + "\n")
    fw.close()

def write_mini_data(data, axis, file_path):
    data = sort_data(data, axis)
    fw = open(file_path, "w")
    for line in data:
        reslt = ""
        (u, i, r) = line
        if i <= 50 and u <= 10:
            for field in line:
                reslt += "%s" % int(field) + "\t"
            fw.write(reslt[:-1] + "\n")
    fw.close()


if __name__ == '__main__':
    file_path = "data/ml-100k/u.data"
    data = load_data(file_path)
    write_mini_data(data, 0, "data/ml-100k/mini-user-item.data")
    write_data(data, 0, "data/ml-100k/user_sorted-item.data")
    print "---------------------华丽的分割线------------------"
    write_data(data, 1, "data/ml-100k/user-item_sorted.data")
