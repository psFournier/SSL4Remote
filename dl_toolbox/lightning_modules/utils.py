def ramp_down(epoch, initial_lr, final_lr, max_epochs, milestones):

    m = epoch / max_epochs 
    if m < milestones[0]:
        return 1
    elif m < milestones[1]:
        return 1 + ((m - milestones[0]) / (milestones[1] - milestones[0])) * (final_lr/initial_lr - 1)
    else:
        return final_lr/initial_lr

def get_masked_labels(mask):

    if not self.train_with_void:
        # Granted that the first label is the void/unknown label, this extracts
        # from labels the mask to use to ignore this class
        labels_onehot = mask[:, 1:, :, :]
        loss_mask = 1. - mask[:, [0], :, :]
    else:
        labels_onehot, loss_mask = mask, torch.ones_like(mask)

    return labels_onehot, loss_mask

def compute_sup_loss(logits, labels_onehot, loss_mask):

    loss1_noreduce = self.ce_loss(logits, labels_onehot)
    # The mean over all pixels is replaced with a mean over unmasked ones
    loss1 = torch.sum(loss_mask * loss1_noreduce) / torch.sum(loss_mask)
    loss2 = self.dice_loss(logits * loss_mask, labels_onehot * loss_mask)

    return loss1, loss2, loss1 + 2*loss2



