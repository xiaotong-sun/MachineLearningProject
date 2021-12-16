import pandas as pd

data = pd.read_table('dev.txt')
data.to_csv('dev.csv', header=1, encoding='utf_8_sig')
