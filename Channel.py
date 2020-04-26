import numpy as np
from scipy.stats import multivariate_normal

dependency = [[0,1],[2,3],[4,5],[6,7],[0,1],[2,3],[4,5],[6,7]]
pw = []

def noise():
    file = open("noise.txt", "r")
    data = file.read()
    datanoise = data.split()
    datanoise = np.array(datanoise).astype(np.float)
    return datanoise


def readCoefficient():
    file = open("h.txt", "r")
    data = file.read()
    datah = data.split()
    datah = np.array(datah).astype(np.float)
    coeff_no = len(datah)
    return coeff_no, datah


def channel(a, b, datah, noise_mean, noise_variance):
    x = float(a)*float(datah[0]) + float(b)*float(datah[1]) + np.random.normal(noise_mean, noise_variance)
    return x


def readtrain():
    print('Training:')
    print()
    datanoise = noise()
    noise_mean = datanoise[0]
    noise_variance = datanoise[1]
    coeff_no, datah = readCoefficient()
    file = open("now.txt", "r")
    datas = file.read()
    data = []
    for i in range(0, len(datas)):
        data.append(float(datas[i]))
    x = [0]
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []

    for i in range(2, len(data)):
        #print(data[i-2], data[i-1], data[i])
        xk1 = (channel(data[i-1], data[i-2], datah, noise_mean, noise_variance))
        xk2 = (channel(data[i], data[i-1], datah, noise_mean, noise_variance))
        x.append([xk1, xk2])

        if data[i-2]==0 and data[i-1]==0 and data[i]==0:
            x0.append([xk1,xk2])
        elif data[i-2]==0 and data[i-1]==0 and data[i]==1:
            x1.append([xk1,xk2])
        elif data[i-2]==0 and data[i-1]==1 and data[i] == 0:
            x2.append([xk1, xk2])
        elif data[i-2]==0 and data[i-1]==1 and data[i]==1:
            x3.append([xk1,xk2])
        elif data[i-2]==1 and data[i-1]==0 and data[i]==0:
            x4.append([xk1,xk2])
        elif data[i-2]==1 and data[i-1]==0 and data[i]==1:
            x5.append([xk1,xk2])
        elif data[i-2]==1 and data[i-1]==1 and data[i]==0:
            x6.append([xk1,xk2])
        else:
            x7.append([xk1,xk2])

    meanVector = [[float(sum(col)) / len(col) for col in zip(*x0)], [float(sum(col)) / len(col) for col in zip(*x1)],
            [float(sum(col)) / len(col) for col in zip(*x2)], [float(sum(col)) / len(col) for col in zip(*x3)],
            [float(sum(col)) / len(col) for col in zip(*x4)], [float(sum(col)) / len(col) for col in zip(*x5)],
            [float(sum(col)) / len(col) for col in zip(*x6)], [float(sum(col)) / len(col) for col in zip(*x7)]]

    covVector = [np.cov(np.matrix(x0).T), np.cov(np.matrix(x1).T), np.cov(np.matrix(x2).T), np.cov(np.matrix(x3).T),
                 np.cov(np.matrix(x4).T), np.cov(np.matrix(x5).T), np.cov(np.matrix(x6).T), np.cov(np.matrix(x7).T)]

    pw.append(len(x0) / len(x))
    pw.append(len(x1) / len(x))
    pw.append(len(x2) / len(x))
    pw.append(len(x3) / len(x))
    pw.append(len(x4) / len(x))
    pw.append(len(x5) / len(x))
    pw.append(len(x6) / len(x))
    pw.append(len(x7) / len(x))

    # print(len(x))
    print('mean: \n',meanVector)
    print('cov: \n', covVector)

    return meanVector, covVector

def readtest():
    print()
    print('Testing:')
    print()
    datanoise = noise()
    noise_mean = datanoise[0]
    noise_variance = datanoise[1]
    coeff_no, datah = readCoefficient()
    file = open("test0.txt", "r")
    datas = file.read()
    data = []
    for i in range(0, len(datas)):
        data.append(float(datas[i]))
    x = []

    for i in range(2, len(data)):
        #print(data[i-2], data[i-1], data[i])
        xk1 = (channel(data[i-1], data[i-2], datah, noise_mean, noise_variance))
        xk2 = (channel(data[i], data[i-1], datah, noise_mean, noise_variance))
        x.append([xk1, xk2])
    return x


def viterbi(x, i, j, meanVector, covVector):

    if i==0:
        s = '1'
        if j<=4 :
            s='0'
        return np.log(pw[j])+np.log(multivariate_normal.pdf(x[i], meanVector[j], covVector[j])+ 0.00001), s
    p, path1 = viterbi(x, i-1, dependency[j][0], meanVector, covVector)
    q, path2 = viterbi(x, i-1, dependency[j][1], meanVector, covVector)
    s = ''
    if p>=q:
        s = path1
        if dependency[j][0]>=4:

            s+='1'
        else:
            s+='0'
        return max(p, q) + np.log(multivariate_normal.pdf(x[i], meanVector[j], covVector[j]) + 0.00001), s
    else:
        s = path2
        if dependency[j][1]>=4:
            s+='1'
        else:
            s+='0'
        return max(p, q) + np.log(multivariate_normal.pdf(x[i], meanVector[j], covVector[j]) + 0.00001), s


def test(x, meanVector, covVector):
    largest = 0
    largest_index = 0
    total_path = []
    for i in range(0, 8):
        v, pathr = viterbi(x, len(x)-1, i, meanVector, covVector)
        if v > largest:
            largest = v
            largest_index = i
        total_path.append(pathr)
    return total_path, largest_index


def main():
    meanVector, covVector = readtrain()
    x=readtest()
    f = open("write.txt", "w")
    for item in x:
        f.write("%s\n" % item)
    total_path, largest_index = test(x, meanVector, covVector)
    print(total_path[largest_index])

main()