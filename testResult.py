import prettytable as PT
import json
import matplotlib.pyplot as plt


def printTable(filename):
    table = PT.PrettyTable()
    data_file = open(filename, 'r')
    rows = json.loads(data_file.read())
    table.field_names = rows[0].keys()
    for row in rows:
        tmp = []
        for field in row:
            tmp.append(row[field])
        table.add_row(tmp)

    print(table)


def toSave():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    data_file = open('pDFT.txt', 'r')
    rows = json.loads(data_file.read())
    x, y = [], []
    for row in rows:
        x.append(row['pDFT'])
        y.append(row['res'])
    ax1.plot(x, y)
    ax1.set_xlabel('pDFT')
    ax1.set_ylabel('Результат')

    data_file = open('pDCT.txt', 'r')
    rows = json.loads(data_file.read())
    x, y = [], []
    for row in rows:
        x.append(row['pDCT'])
        y.append(row['res'])
    ax2.plot(x, y)
    ax2.set_xlabel('pDCT')
    ax2.set_ylabel('Результат')

    data_file = open('S.txt', 'r')
    rows = json.loads(data_file.read())
    x, y = [], []
    for row in rows:
        x.append(row['s'])
        y.append(row['res'])
    ax3.plot(x, y)
    ax3.set_xlabel('s')
    ax3.set_ylabel('Результат')

    data_file = open('c.txt', 'r')
    rows = json.loads(data_file.read())
    x, y = [], []
    for row in rows:
        x.append((row['c'] - 1) * 40)
        y.append(row['res'])
    ax4.plot(x, y)
    ax4.set_xlabel('Количество изображений')
    ax4.set_ylabel('Результат')

    plt.show()


def showPlot(filename):
    dataFile = open(filename, 'r')
    data = json.loads(dataFile.read())
    field, result = data[0].keys()
    x, y = [[],[]]
    for row in data:
        x.append(row[field])
        y.append(row[result])
    fig = plt.subplot()
    fig.plot(x, y)
    fig.set_xlabel(field)
    fig.set_ylabel(result)
    plt.show()


printTable('pDFT.txt')
printTable('pDCT.txt')
printTable('gradC.txt')
printTable('BINC.txt')
printTable('scaleC.txt')
printTable('dctC.txt')
printTable('dftC.txt')

showPlot('dftC.txt')
showPlot('dctC.txt')
showPlot('scaleC.txt')
showPlot('BINC.txt')
showPlot('gradC.txt')