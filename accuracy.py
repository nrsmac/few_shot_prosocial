import sys
import pandas as pd
from sklearn.metrics import accuracy_score

if len(sys.argv) < 2:
    print(f'USAGE: {sys.argv[0]} [CSV FILENAME]')
path = sys.argv[1]
df = pd.read_csv(path)
accuracy = accuracy_score(df['gold'], df['y_hat'])
print(f'{accuracy=}')