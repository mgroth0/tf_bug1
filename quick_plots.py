import matplotlib.pyplot as plt
import json


print('hello world')

with open('data_tfbug/data_result/1611690609/data_result.json','r') as f:
    data = json.loads(f.read())

plt.plot(
    data[0]['history']['accuracy'],
)
plt.show()
