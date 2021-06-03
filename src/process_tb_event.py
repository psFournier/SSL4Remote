from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt

path1 = '/home/pierre/PycharmProjects/RemoteSensing/outputs/history/from_onera/austin_d4_allcolor_2021-05-06/events.out.tfevents.1620320668.calculon.835824.0'

event_acc1 = EventAccumulator(path1)
event_acc1.Reload()
print(event_acc1.Tags())
w_times_1, step_nums1, vals1 = zip(*event_acc1.Scalars('Train_IoU_0'))

plt.show()



