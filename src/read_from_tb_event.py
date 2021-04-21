from argparse import ArgumentParser
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import terminalplot as tp

parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--scalar", type=str)

args = parser.parse_args()

path1 = '/home/pierre/PycharmProjects/RemoteSensing/outputs/baseline_christchurch_noaug_2021-04-16/events.out.tfevents.1618562118.bender.2376589.0'
path2 = '/home/pierre/PycharmProjects/RemoteSensing/outputs/baseline_christchurch_b0/events.out.tfevents.1619012521.bender.228002.0'
path3 = '/home/pierre/PycharmProjects/RemoteSensing/outputs/baseline_christchurch_wce/events.out.tfevents.1619005201.bender.218752.0'
path4 = '/home/pierre/PycharmProjects/RemoteSensing/outputs/baseline_christchurch_b0_swa/events.out.tfevents.1619012551.bender.228666.0'

event_acc = EventAccumulator(path1)
event_acc.Reload()
w_times, step_nums, vals = zip(*event_acc.Scalars('val_IoU'))
plt.plot([32*x for x in step_nums], vals)

for path in [path4, path2, path3]:
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    w_times, step_nums, vals = zip(*event_acc.Scalars('val_IoU'))
    plt.plot([64*x for x in step_nums], vals)

plt.show()

# for w_time, step_num, val in event_acc.Scalars(args.scalar):
#     print(step_num, val)

