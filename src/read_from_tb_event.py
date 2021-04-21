from argparse import ArgumentParser
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import terminalplot as tp

parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--scalar", type=str)

args = parser.parse_args()

event_acc = EventAccumulator(args.path)


event_acc.Reload()
# Show all tags in the log file

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('val_IoU'))
plt.plot(step_nums, vals)
# for w_time, step_num, val in event_acc.Scalars(args.scalar):
#     print(step_num, val)

