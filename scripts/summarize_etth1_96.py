import re
from pathlib import Path


result_path = Path('result_long_term_forecast.txt')
if not result_path.exists():
    raise SystemExit('result_long_term_forecast.txt not found')

lines = result_path.read_text().splitlines()
target = 'long_term_forecast_ETTh1_336_96'
rows = []

for i, line in enumerate(lines):
    if target not in line:
        continue
    if i + 1 >= len(lines):
        continue
    metric_line = lines[i + 1]
    match = re.search(r'mse:([^,]+), mae:([^,]+), dtw:([^\s]+)', metric_line)
    if not match:
        continue
    setting = line.strip()
    if '_iTransformer_' in setting:
        label = 'iTransformer'
    elif '_PIBR_' in setting and '_phase_only_' in setting:
        label = 'PIBR phase_only'
    elif '_PIBR_' in setting and '_fixed_avg_' in setting:
        label = 'iTransformer + PIBR fixed_avg'
    else:
        label = 'other'
    rows.append((label, float(match.group(1)), float(match.group(2)), setting))

if not rows:
    raise SystemExit('No ETTh1 336->96 results found')

print('ETTh1 seq_len=336 pred_len=96')
for label, mse, mae, setting in rows[-8:]:
    print(f'{label:32s} mse={mse:.6f} mae={mae:.6f}')
    print(f'  {setting}')
