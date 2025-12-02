# import json
# import matplotlib.pyplot as plt
#
# log_file = "elc-bert-base_len-512-1000-steps_5_epochs_42.txt"
#
# epochs = []
# losses = []
#
# # Read JSON-lines log file
# with open(log_file, "r") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         entry = json.loads(line)
#         epochs.append(entry["epoch"])
#         losses.append(entry["loss"])
#
# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(epochs, losses)#, marker='o')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss per Epoch")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_file = 'base_len-512_42.txt'
# log_file = 'zero_len-512_42.txt'

with open(log_file, 'r') as f:
  filedata = f.read()

filedata = filedata.replace('\'', '\"')

# Write the file out again
with open(log_file, 'w') as f:
  f.write(filedata)

with open(log_file) as f:
    dict_train = json.load(f)

df = pd.DataFrame.from_dict(dict_train)
print(df.head())
print(df.columns)

sns.lineplot(x=df['epoch'], y=df['loss'], marker='o', color='blue', label='train')
sns.lineplot(x=df['epoch'], y=df['eval_loss'], marker='o', color='orange', label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(log_file.strip('.txt')+'.png')
