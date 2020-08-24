import torch

def greedy_decode(model, src, max_len, start_symbol_ind=0):
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    memory = model.encode(src)
    ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    for i in range(max_len - 1):
        # ys_i:(batch_size, T_pred=i+1)
        target_mask = model.decoder.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, nhid)
        prob = model.decoder.generator(out[-1, :])  # (T_-1, batch_size, nhid)
        next_word = torch.argmax(prob, dim=1)  # (batch_size)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        # ys_i+1: (batch_size,T_pred=i+2)
    return ys


def beam_search(model, src, max_len, start_symbol_ind=0, beam_size=10, search_depth=10):
    """
    占位
    """
    ys = None
    return ys