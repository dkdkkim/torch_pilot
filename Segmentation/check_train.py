import json
from visual import plot_training
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open(f"/data/dk/exp/torch_seg_test/training_losses.json",'r') as file:
    training_losses = json.load(file)
with open(f"/data/dk/exp/torch_seg_test/validation_losses.json",'r') as file:
    validation_losses = json.load(file)
with open(f"/data/dk/exp/torch_seg_test/lr_rates.json",'r') as file:
    lr_rates = json.load(file)
# print(training_losses)
# print(validation_losses)
plot_training(training_losses, validation_losses, lr_rates, gausian=True, sigma=1, figsize=(10,4))
plt.show()