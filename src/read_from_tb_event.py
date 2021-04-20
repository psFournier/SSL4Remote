from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator(
    '/home/pierre/PycharmProjects/RemoteSensing/outputs'
    '/baseline_christchurch_noaug_2021-04-16/events.out.tfevents.1618562118.bender.2376589.0')
event_acc.Reload()
# Show all tags in the log file

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
# w_times, step_nums, vals = zip(*event_acc.Scalars('val_IoU'))
for w_time, step_num, val in event_acc.Scalars('val_IoU')[::20]:
    print(step_num, val)