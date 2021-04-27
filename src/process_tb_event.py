from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt

path1 = '/home/pierre/PycharmProjects/RemoteSensing/outputs/baseline_christchurch_noaug_2021-04-16/events.out.tfevents.1618562118.bender.2376589.0'
path2 = '/home/pierre/PycharmProjects/RemoteSensing/outputs/tensorboard/baseline_christchurch_2021-04-24/events.out.tfevents.1619224809.bender.1146366.0'

event_acc1 = EventAccumulator(path1)
event_acc1.Reload()
_, step_nums1, vals1 = zip(*event_acc1.Scalars('val_IoU'))
plt.plot([32*x for x in step_nums1], vals1)

event_acc2 = EventAccumulator(path2)
event_acc2.Reload()
_, step_nums2, vals2 = zip(*event_acc2.Scalars('Val IoU class 1'))
_, step_nums3, vals3 = zip(*event_acc2.Scalars('Val IoU class 0'))
plt.plot([32*x for x in step_nums2], [(x+y)/2 for x,y in zip(vals2, vals3)])

plt.show()



