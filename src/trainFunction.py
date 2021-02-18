import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def train(network,
          sup_train_dataloader,
          test_dataloader,
          unsup_train_dataloader,
          optimizer,
          sup_loss,
          unsup_loss,
          epochs,
          tensorboard_writer,
          save_dir,
          use_cuda):

    if use_cuda:
        network.cuda()

    train_loss, val_loss = [], []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Training phase
        network.train(True)

        running_supervised_loss = 0.0
        running_acc = 0.0
        running_unsupervised_loss = 0.0

        for (sup_data, unsup_data) in zip(sup_train_dataloader,
                                          unsup_train_dataloader):

            train_inputs, train_labels = sup_data
            if use_cuda:
                train_inputs.cuda()
                train_labels.cuda()
                unsup_data.cuda()

            optimizer.zero_grad()
            outputs = network(train_inputs)
            supervised_loss = sup_loss(outputs, train_labels)

            np_outputs = outputs.cpu().data.numpy()
            np_predictions = np.argmax(np_outputs, axis=1)
            np_train_labels = train_labels.cpu().data.numpy()
            cm = confusion_matrix(np_train_labels.ravel(),
                                  np_predictions.ravel(),
                                  labels=list(range(out_channels)))
            global_cm += cm
            error += error_.item()

            # scores
            overall_acc = metrics.stats_overall_accuracy(global_cm)
            average_acc, _ = metrics.stats_accuracy_per_class(global_cm)
            average_iou, _ = metrics.stats_iou_per_class(global_cm)
            average_f1, _ = metrics.stats_f1score_per_class(global_cm)
            loss = error / global_cm.sum()

            running_acc += train_accuracy * \
                           sup_train_dataloader.batch_size
            running_supervised_loss += supervised_loss * \
                                       sup_train_dataloader.batch_size

            rotation_1, rotation_2 = np.random.choice(
                [0,1,2,3],
                size=2,
                replace=False
            )
            augmented_1 = torch.rot90(unsup_data, k=rotation_1, dims=[2, 3])
            augmented_2 = torch.rot90(unsup_data, k=rotation_2, dims=[2, 3])
            outputs_1 = network(augmented_1)
            outputs_2 = network(augmented_2)
            unaugmented_1 = torch.rot90(outputs_1, k=-rotation_1, dims=[2, 3])
            unaugmented_2 = torch.rot90(outputs_2, k=-rotation_2, dims=[2, 3])

            unsupervised_loss = unsup_loss(
                unaugmented_1,
                unaugmented_2
            )

            running_unsupervised_loss += unsupervised_loss * \
                                         unsup_train_dataloader.batch_size

            # For now, test supervised learning
            total_loss = supervised_loss
            # total_loss = supervised_loss + unsupervised_loss

            total_loss.backward()
            optimizer.step()

        examples_seen = len(sup_train_dataloader) * \
                        sup_train_dataloader.batch_size
        epoch_loss = running_supervised_loss / examples_seen
        epoch_acc = running_acc / examples_seen
        tensorboard_writer.add_scalar('train loss',
                                      epoch_loss,
                                      epoch * examples_seen
                                      )
        tensorboard_writer.add_scalar('train acc',
                                      epoch_acc,
                                      epoch * examples_seen
                                      )

        print(
            'Train Loss: {:.4f} Acc: {}'.format(epoch_loss, epoch_acc))

        train_loss.append(epoch_loss)

        # Training phase
        network.train(False)

        running_supervised_loss = 0.0
        running_acc = 0.0

        for val_data in test_dataloader:

            # input_batch.cuda()
            # label_batch.cuda()

            val_inputs, val_labels = val_data
            with torch.no_grad():
                outputs = network(val_inputs)
                supervised_loss = sup_loss(outputs, val_labels)

            val_acc = (outputs.argmax(dim=1) == val_labels).float().mean()

            running_acc += val_acc * test_dataloader.batch_size
            running_supervised_loss += supervised_loss * \
                                       test_dataloader.batch_size


        examples_seen = len(test_dataloader) * \
                        test_dataloader.batch_size
        epoch_loss = running_supervised_loss / examples_seen
        epoch_acc = running_acc / examples_seen
        tensorboard_writer.add_scalar('val loss',
                                      epoch_loss,
                                      epoch * examples_seen
                                      )
        tensorboard_writer.add_scalar('val acc',
                                      epoch_acc,
                                      epoch * examples_seen
                                      )

        print(
            'Val Loss: {:.4f} Acc: {}'.format(epoch_loss, epoch_acc))

        val_loss.append(epoch_loss)

        if epoch % 10 == 0:
            # save the network
            torch.save(network.state_dict(),
                       os.path.join(save_dir, "unet_epoch_%03d.pth" % epoch))

    return train_loss, val_loss
