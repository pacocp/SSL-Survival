import torch
import torch.nn as nn
from lifelines.utils import concordance_index

# from https://github.com/gevaertlab/MultiScaleFusionPrognostic/blob/master/pathology_models.py
def cox_loss(cox_scores, times, status):
    '''
    :param cox_scores: cox scores, size (batch_size)
    :param times: event times (either death or censor), size batch_size
    :param status: event status (1 for death, 0 for censor), size batch_size
    :return: loss of size 1, the sum of cox losses for the batch
    '''

    # yj >= yi
    times, sorted_indices = torch.sort(-times)

    # z*beta
    cox_scores = cox_scores[sorted_indices]
    status = status[sorted_indices]

    # why?
    cox_scores = cox_scores -torch.max(cox_scores)

    # z*beta - log(sum(exps))
    exp_scores = torch.exp(cox_scores)
    loss = cox_scores - torch.log(torch.cumsum(exp_scores, dim=0)+1e-5)

    # only consider uncensored
    loss = - loss * status 
    # TODO maybe divide by status.sum()

    if (loss != loss).any():
        import pdb;
        pdb.set_trace()

    return loss.mean()

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss,self).__init__()

    def forward(self,cox_scores,times,status):
        return cox_loss(cox_scores,times,status)

def get_survival_CI(output_list, survival_months, vital_status):
    CI = concordance_index(survival_months, -output_list, vital_status)

    return CI
