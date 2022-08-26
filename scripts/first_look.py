import csv
import pandas as pd


root_dir = 'D:/OpenClassrooms/projet_7'
data_path = root_dir + '/data/' + 'training.1600000.processed.noemoticon.csv'

with open(data_path, 'r') as f:
    csv_data = csv.reader(f)

    df_list = []
    for row in csv_data:
        df_list.append(row)
        break

df = pd.DataFrame(df_list)
print(df, '\n')

for column in df:
    print(df[column])
    print()