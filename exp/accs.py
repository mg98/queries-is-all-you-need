import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
test_cases = df.groupby(['test_input', 'test_output'])

for k in [1, 5, 10]:
    acc = test_cases.apply(lambda group: group['test_output'].iloc[0] == group.head(k)['output']).sum()
    print(f'Top-{k} accuracy: \t{acc/float(test_cases.ngroups)}')