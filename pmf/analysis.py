# -*- coding: UTF-8 -*-
'''
Created on 2017年5月16日

@author: Conter
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    prefer = []
    for line in open(file_path, 'r'):  # 打开指定文件
        (userid, movieid, rating) = line.split('\t')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = np.array(prefer)
    return data
def show_data(data):
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    zhfont = matplotlib.font_manager.FontProperties(fname=r"C:\\WINDOWS\\Fonts\\STKAITI.TTF", size=14)
    size = [40, 40, 40, 40, 40]
    ratings = range(1, 6)
    color = ['red', 'yellow', 'green', 'blue', 'grey']
    marker = ['o', 'x', 'v', '+', '*']
    cp_x = [[], [], [], [], []]
    cp_y = [[], [], [], [], []]
    for i in range(len(data) - 1):
        for j in ratings:
            if data[i, 2] == j:
                cp_x[j - 1].append(data[i, 0])
                cp_y[j - 1].append(data[i, 1])
    type = []
    for i in range(len(ratings)):
        type.append(axes.scatter(cp_x[i], cp_y[i], s=size[i], marker=marker[i], c=color[i]))
    plt.xlabel("UserID")
    plt.ylabel("ItemID")
    plt.title("The MovieLens Dataset ")
    axes.legend(tuple(type), (u'评分1', u'评分2', u'评分3', u'评分4', u'评分5'), loc=2, prop=zhfont)
    plt.show()

def search(data, u, i):
    for line in data:
        if line[0] == u and line[1] == i:
            return line[2]
    else:
        return -10

def show_rating_matrix(data): 
    print "\t", 
    for k in range(1, 51): print "%s\t" %k, 
    print
    for i in range(1,11):
        print i, "\t",
        for j in range(1, 51):
            rate = search(data, i, j)
            if rate == -10:
                print"-\t",
            else:
                print "%s\t" % int(rate),
        print "\n",
            

if __name__ == '__main__':
    data = load_data("data/ml-100k/mini-user-item.data")
#     print data
#     show_data(data)
    show_rating_matrix(data)
