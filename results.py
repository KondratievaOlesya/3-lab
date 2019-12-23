import prettytable as PT
import json
import matplotlib.pyplot as plt


def printTable(filename, fieldName):
    table = PT.PrettyTable()
    table.field_names = ['m', 'n', 'pDFT', 'pDCT', 'BIN', 'W', 'S', 'Результат']
    data_file = open(filename, 'r')
    rows = json.loads(data_file.read())
    fieldToSend = []
    res = []
    for row in rows:
        tmp = []
        for field in row:
            tmp.append(row[field])
        fieldToSend.append(row[fieldName])
        res.append(row['res'])
        table.add_row(tmp)

    print(table)
    return [fieldToSend, res]


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
x, y = printTable('testResult.txt', 'pDFT')
x, y = printTable('testResult1.txt', 'pDCT')

ax2.plot(x, y)
ax2.set_xlabel('pDCT')
ax2.set_ylabel('Результат')
x, y = printTable('testResultS.txt', 's')
ax3.plot(x, y)
ax3.set_xlabel('S')
ax3.set_ylabel('Результат')
x, y = printTable('testResultW.txt', 'w')
ax4.plot(x, y)
ax4.set_xlabel('W')
ax4.set_ylabel('Результат')

table = PT.PrettyTable()
table.field_names = ['Количество тестовых изображений', 'm', 'n', 'pDFT', 'pDCT', 'BIN', 'W', 'S', 'Результат']
data_file = open('testResultCount.txt', 'r')
rows = json.loads(data_file.read())
x = []
y = []
for row in rows:
    tmp = []
    for field in row:
        if field == 'c':
            tmp.append((row['c'] - 1)*40)
            x.append((row['c'] - 1)*40)
        elif field == 'res':
            y.append(row['res'])
            tmp.append(row[field])
        else:
            tmp.append(row[field])
    table.add_row(tmp)

print(table)

ax1.plot(x, y)
ax1.set_xlabel('Количество изображений')
ax1.set_ylabel('Результат')
plt.show()
