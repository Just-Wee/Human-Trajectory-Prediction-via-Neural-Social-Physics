import torch


def get_neighbors_info_with_prediction(supplement, pos_pd, v_pd, past_len, pd_time, first_frame, first_part):
    """
    `supplement_pd` is the neighbors infor with prediction at time `pd_time`
    """
    assert pd_time >= past_len, \
        "The time to predict should be greater than the length of time in the past"

    supplement_pd = torch.zeros_like(supplement[:, 0, :, :])

    for p_i in range(supplement.shape[0]):
        nb_all_num = int(supplement[p_i, 0, -1, 0]) # num of neighbors who appeared all the time
        
        # For neighbors who don't appeared all the time, use the last frame of history
        supplement_pd[p_i, nb_all_num:, :] = supplement[p_i, past_len-1, nb_all_num:, :]

        supplement_pd[p_i, :len(first_part[p_i]), :2] = pos_pd[first_part[p_i]] + \
            first_frame[first_part[p_i]] - first_frame[p_i]
        supplement_pd[p_i, :len(first_part[p_i]), 2:4] = v_pd[first_part[p_i]]
        supplement_pd[p_i, :len(first_part[p_i]), 4] = 1

    return supplement_pd
