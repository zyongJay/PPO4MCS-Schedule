"""
该脚本用于我自己测试开发中的bug的，不要理会
"""
from core import euclidean_distance

pos = [[104.106739, 30.682329],
       [104.063202, 30.694779],
       [104.021542, 30.6666],
       [104.013521, 30.650209],
       [104.017023, 30.63751],
       [104.017115, 30.637547]]

for i in range(len(pos)):
    if i == len(pos) - 1:
        break
    dist = euclidean_distance(pos[i][0], pos[i][1], pos[i + 1][0], pos[i + 1][1]) / 1000.0
    print(dist)
