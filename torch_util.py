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


def auto_rnn(rnn: nn.RNN, seqs, lengths, batch_first=True):
    batch_size = seqs.size(0) if batch_first else seqs.size(1)
    state_shape = get_state_shape(rnn, batch_size, rnn.bidirectional)

    h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths, batch_first)
    output, (hn, cn) = rnn(packed_pinputs, (h0, c0))
    output = unpack_from_rnn_seq(output, r_index, batch_first)

    return output


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


def seq2seq_cross_entropy(logits, label, l, chuck=None):
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