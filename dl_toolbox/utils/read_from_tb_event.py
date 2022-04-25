from argparse import ArgumentParser
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

#parser = ArgumentParser()
#parser.add_argument("--path", type=str)
#parser.add_argument("--scalar", type=str)
#args = parser.parse_args()

PATHS = [
    '/d/pfournie/ai4geo/outputs/version_50/events.out.tfevents.1650532912.gpgpu01.sis.cnes.fr.48359.0',
]

for path in PATHS:
    event_acc = EventAccumulator(path)
    print("reloading")
    event_acc.Reload()
    print("reloaded")
    wall_times, steps, values = zip(*event_acc.Scalars('Val_f1'))
    print(values)

#fig = plt.figure()
#ax1 = fig.add_subplot(111)


