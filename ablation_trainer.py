import main
import json

import config as CFG
import numpy as np

import datetime

trials = 1

ablation_dict = {
    1: [],
    5: [],
    10: [],
    25: []
}

run_times = []

model_size = 1000
epochs = 10

for i in range(trials):
    results, time = main.main(size=model_size, epochs=epochs)

    run_times.append(time)

    for k, v in results.items():
        ablation_dict[k].append(v)

output_file = f'ablations/{model_size}t_{epochs}e_triplet_results.json'
with open(output_file, 'w+') as f:
    json.dump(ablation_dict, f)

print(f'For models trained on {model_size} items over {epochs} epochs')
print(f'Recall Averages @K and STD over {trials} trials:')
for k, v in ablation_dict.items():
    print(f'\t@{k}: {np.average(v)*100:.2f} ± {np.std(v)*100:.2f}')

print(f'Average Training time: {datetime.timedelta(seconds=np.average(run_times))}')
