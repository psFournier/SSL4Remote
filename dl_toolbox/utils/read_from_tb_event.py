from argparse import ArgumentParser
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--scalar", type=str)
args = parser.parse_args()

event_acc = EventAccumulator(args.path)
event_acc.Reload()
for w_time, step_num, val in event_acc.Scalars(args.scalar):
    print(step_num, val)

