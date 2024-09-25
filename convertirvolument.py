import pandas as pd

ypf = pd.read_csv('YPF.csv', index_col='Date', parse_dates=['Date'])
for vol in ypf['Vol.']:
    if 'M' in vol:
        vol.rstrip(vol[-1])
        vol.float()*1000000
    if 'K' in vol:
        vol.rstrip(vol[-1])
        vol.float()*1000

ypf.DataFrame.to_csv('ypf.csv')