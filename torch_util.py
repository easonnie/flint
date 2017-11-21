import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import functools


def pad_1d(seq, pad_l):
    """
    The seq is a sequence having shape [T, ..]. Note: The seq contains only one instance. This is not batched.
    
    :param seq:  Input sequence with shape [T, ...]
    :param pad_l: The required pad_length.
    :return:  Output sequence will have shape [Pad_L, ...]
    """
    l = seq.size(0)
    if l >= pad_l:
        return seq[:pad_l, ]  # Truncate the length if the length is bigger than required padded_length.
    else:
        pad_seq = Variable(seq.data.new(pad_l - l, *seq.size()[1:]).zero_())  # Requires_grad is False
        return torch.cat([seq, pad_seq], dim=0)


def pad(seqs, length, batch_first=True):
    #TODO The method seems useless to me. Delete?
    """
    Padding the sequence to a fixed length.
    
    :param seqs: [B, T, D] or [B, T] if batch_first else [T, B * D] or [T, B]
    :param length: [B]
    :param batch_first:
    :return:
    """
    if batch_first:
        # [B * T * D]
        if length <= seqs.size(1):
            return seqs
        else:
            batch_size = seqs.size(0)
            pad_seq = Variable(seqs.data.new(batch_size, length - seqs.size(1), *seqs.size()[2:]).zero_())
            # [B * T * D]
            return torch.cat([seqs, pad_seq], dim=1)
    else:
        # [T * B * D]
        if length <= seqs.size(0):
            return seqs
        else:
            return torch.cat([seqs, Variable(seqs.data.new(length - seqs.size(0), *seqs.size()[1:]).zero_())])


def batch_first2time_first(inputs):
    """
    Convert input from batch_first to time_first:
    [B, T, D] -> [T, B, D]
    
    :param inputs:
    :return:
    """
    return torch.transpose(inputs, 0, 1)


def time_first2batch_first(inputs):
    """
    Convert input from batch_first to time_first:
    [T, B, D] -> [B, T, D] 
    
    :param inputs:
    :return:
    """
    return torch.transpose(inputs, 0, 1)


def get_state_shape(rnn: nn.RNN, batch_size, bidirectional=False):
    """
    Return the state shape of a given RNN. This is helpful when you want to create a init state for RNN.

    Example:
    c0 = h0 = Variable(src_seq_p.data.new(*get_state_shape([your rnn], 3, bidirectional)).zero_())
    
    :param rnn: nn.LSTM, nn.GRU or subclass of nn.RNN
    :param batch_size:  
    :param bidirectional:  
    :return: 
    """
    if bidirectional:
        return rnn.num_layers * 2, batch_size, rnn.hidden_size
    else:
        return rnn.num_layers, batch_size, rnn.hidden_size


def pack_list_sequence(inputs, l, max_l=None, batch_first=True):
    """
    Pack a batch of Tensor into one Tensor with max_length.
    :param inputs: 
    :param l: 
    :param max_l: The max_length of the packed sequence.
    :param batch_first: 
    :return: 
    """
    batch_list = []
    max_l = max(list(l)) if not max_l else max_l
    batch_size = len(inputs)

    for b_i in range(batch_size):
        batch_list.append(pad_1d(inputs[b_i], max_l))
    pack_batch_list = torch.stack(batch_list, dim=1) if not batch_first \
        else torch.stack(batch_list, dim=0)
    return pack_batch_list


def pack_for_rnn_seq(inputs, lengths, batch_first=True):
    """
    :param inputs: Shape of the input should be [B, T, D] if batch_first else [T, B, D].
    :param lengths:  [B]
    :param batch_first: 
    :return: 
    """
    if not batch_first:
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.cat(s_inputs_list, 1)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

        return packed_seq, reverse_indices

    else:
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[i, :, :])
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.stack(s_inputs_list, dim=0)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list, batch_first=batch_first)

        return packed_seq, reverse_indices


def unpack_from_rnn_seq(packed_seq, reverse_indices, batch_first=True):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=batch_first)
    s_inputs_list = []

    if not batch_first:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
        return torch.cat(s_inputs_list, 1)
    else:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[i, :, :].unsqueeze(0))
        return torch.cat(s_inputs_list, 0)


def auto_rnn(rnn: nn.RNN, seqs, lengths, batch_first=True, init_state=None, output_last_states=False):
    batch_size = seqs.size(0) if batch_first else seqs.size(1)
    state_shape = get_state_shape(rnn, batch_size, rnn.bidirectional)

    if not init_state:
        h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())
    else:
        h0 = init_state['h0'].expand(state_shape)
        c0 = init_state['c0'].expand(state_shape)

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths, batch_first)
    output, (hn, cn) = rnn(packed_pinputs, (h0, c0))
    output = unpack_from_rnn_seq(output, r_index, batch_first)

    if not output_last_states:
        return output
    else:
        return output, (hn, cn)


def pack_sequence_for_linear(inputs, lengths, batch_first=True):
    """
    :param inputs: [B, T, D] if batch_first 
    :param lengths:  [B]
    :param batch_first:  
    :return: 
    """
    batch_list = []
    if batch_first:
        for i, l in enumerate(lengths):
            # print(inputs[i, :l].size())
            batch_list.append(inputs[i, :l])
        packed_sequence = torch.cat(batch_list, 0)
        # if chuck:
        #     return list(torch.chunk(packed_sequence, chuck, dim=0))
        # else:
        return packed_sequence
    else:
        raise NotImplemented()


def chucked_forward(inputs, net, chuck=None):
    if not chuck:
        return net(inputs)
    else:
        output_list = [net(chuck) for chuck in torch.chunk(inputs, chuck, dim=0)]
        return torch.cat(output_list, dim=0)


def unpack_sequence_for_linear(inputs, lengths, batch_first=True):
    batch_list = []
    max_l = max(lengths)

    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.cat(inputs)

    if batch_first:
        start = 0
        for l in lengths:
            end = start + l
            batch_list.append(pad_1d(inputs[start:end], max_l))
            start = end
        return torch.stack(batch_list)
    else:
        raise NotImplemented()


def seq2seq_cross_entropy(logits, label, l, chuck=None, sos_truncate=True):
    """
    :param logits: [exB, V] : exB = sum(l)
    :param label: [B] : a batch of Label
    :param l: [B] : a batch of LongTensor indicating the lengths of each inputs
    :param chuck: Number of chuck to process
    :return: A loss value
    """
    packed_label = pack_sequence_for_linear(label, l)
    cross_entropy_loss = functools.partial(F.cross_entropy, size_average=False)
    total = sum(l)

    assert total == logits.size(0) or packed_label.size(0) == logits.size(0),\
        "logits length mismatch with label length."

    if chuck:
        logits_losses = 0
        for x, y in zip(torch.chunk(logits, chuck, dim=0), torch.chunk(packed_label, chuck, dim=0)):
            logits_losses += cross_entropy_loss(x, y)
        return logits_losses * (1 / total)
    else:
        return cross_entropy_loss(logits, packed_label) * (1 / total)


def max_along_time(inputs, lengths, list_in=False):
    """
    :param inputs: [B, T, D] 
    :param lengths:  [B]
    :return: [B * D] max_along_time
    :param list_in: 
    """
    ls = list(lengths)

    if not list_in:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i, :l, :]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)
    else:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)


def start_and_end_token_handling(inputs, lengths, sos_index=1, eos_index=2, pad_index=0,
                                 op=None):
    """
    :param inputs: [B, T]
    :param lengths: [B]
    :param sos_index: 
    :param eos_index: 
    :param pad_index: 
    :return: 
    """
    batch_size = inputs.size(0)

    if not op:
        return inputs, lengths
    elif op == 'rm_start':
        inputs = torch.cat([inputs[:, 1:], Variable(inputs.data.new(batch_size, 1).zero_())], dim=1)
        return inputs, lengths - 1
    elif op == 'rm_end':
        for i in range(batch_size):
            pass
            # Potential problems!?
            # inputs[i, lengths[i] - 1] = pad_index
        return inputs, lengths - 1
    elif op == 'rm_both':
        for i in range(batch_size):
            pass
            # Potential problems!?
            # inputs[i, lengths[i] - 1] = pad_index
        inputs = torch.cat([inputs[:, 1:], Variable(inputs.data.new(batch_size, 1).zero_())], dim=1)
        return inputs, lengths - 2


def seq2seq_att(mems, lengths, state, att_net=None):
    """
    :param mems: [B, T, D_mem] This are the memories.
                    I call memory for this variable because I think attention is just like read something and then
                    make alignments with your memories.
                    This memory here is usually the input hidden state of the encoder.

    :param lengths: [B]
    :param state: [B, D_state]
                    I call state for this variable because it's the state I percepts at this time step.

    :param att_net: This is the attention network that will be used to calculate the alignment score between
                    state and memories.
                    input of the att_net is mems and state with shape:
                        mems: [exB, D_mem]
                        state: [exB, D_state]
                    return of the att_net is [exB, 1]

                    So any function that map a vector to a scalar could work.

    :return: [B, D_result] 
    """

    d_state = state.size(1)

    if not att_net:
        return state
    else:
        batch_list_mems = []
        batch_list_state = []
        for i, l in enumerate(lengths):
            b_mems = mems[i, :l] # [T, D_mem]
            batch_list_mems.append(b_mems)

            b_state = state[i].expand(b_mems.size(0), d_state) # [T, D_state]
            batch_list_state.append(b_state)

        packed_sequence_mems = torch.cat(batch_list_mems, 0) # [sum(l), D_mem]
        packed_sequence_state = torch.cat(batch_list_state, 0) # [sum(l), D_state]

        align_score = att_net(packed_sequence_mems, packed_sequence_state) # [sum(l), 1]

        # The score grouped as [(a1, a2, a3), (a1, a2), (a1, a2, a3, a4)].
        # aligned_seq = packed_sequence_mems * align_score

        start = 0
        result_list = []
        for i, l in enumerate(lengths):
            end = start + l

            b_mems = packed_sequence_mems[start:end, :] # [l, D_mems]
            b_score = align_score[start:end, :] # [l, 1]

            softed_b_score = F.softmax(b_score.transpose(0, 1)).transpose(0, 1) # [l, 1]

            weighted_sum = torch.sum(b_mems * softed_b_score, dim=0, keepdim=False) # [D_mems]

            result_list.append(weighted_sum)

            start = end

        result = torch.stack(result_list, dim=0)

        return result
