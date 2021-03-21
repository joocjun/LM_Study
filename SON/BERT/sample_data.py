import pandas as pd


data = pd.read_csv('https://raw.githubusercontent.com/muhwagua/color-bert/main/data/raw/data_27636.csv')
data = data['0'].values

sample_data = []
for num in range(len(data)):
    if num % 2 == 0:
        sample_data.append(data[num] + '.' +data[num+1])