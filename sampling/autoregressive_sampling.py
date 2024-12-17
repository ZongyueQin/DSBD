import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample
import random
import numpy as np

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, eos_token_id : int,  
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, pad_token_id = None):
#    torch.manual_seed(123)

    n = len(x)
    T = len(x) + N

    past_key_values = None
    if pad_token_id is None:
        pad_token_id = eos_token_id

    decoder_x = torch.LongTensor([[pad_token_id]]).to(x.device)
    prob_list = []
    while n < T:
        # outputs = model(x)
        if past_key_values:
            if model.config.is_encoder_decoder:
                last_ids = decoder_x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = torch.unsqueeze(last_ids, 0)
                outputs = model(x, decoder_input_ids = last_ids, past_key_values = past_key_values, use_cache = True)
            else:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = torch.unsqueeze(last_ids, 0)
                outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)

        else:
            if model.config.is_encoder_decoder:
                outputs = model(x, decoder_input_ids = decoder_x)
            else:
                outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        prob_list.append(last_p.cpu())
        idx_next = sample(last_p)
        #print(last_p.nonzero())
        #print(idx_next)

     #   print(last_p.nonzero())
     #   print(idx_next)
        if model.config.is_encoder_decoder:
            decoder_x = torch.cat((decoder_x, idx_next), dim=1)
        else:
            x = torch.cat((x, idx_next), dim=1)
        n += 1
        if idx_next == eos_token_id:
            break

    #xxx = input()
    if model.config.is_encoder_decoder:
        x = torch.cat((x, decoder_x), dim=1)
    return x#, prob_list

@torch.no_grad()
def random_width_beam_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, 
                            eos_token_id : int, max_num_beams : int, min_num_beams : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, 
                            pad_token_id = None, debug=False):
    n = len(x)
    T = len(x) + N
#    torch.manual_seed(123)

    past_key_values = None
    debug_past_key_values = None
    if pad_token_id is None:
        pad_token_id = eos_token_id
    prob_list = []

    num_beams = max_num_beams

    if model.config.is_encoder_decoder:
        beams = torch.LongTensor([[pad_token_id]]).to(x.device).repeat(num_beams, 1)
        init_len = 1
    else:
        beams = x.repeat(num_beams, 1)
        init_len = n

    debug_x = x

    beam_scores = torch.zeros(num_beams).to(x.device)
    candidates = []
    while n < T:
        # outputs = model(x)
        repeat_x = x.repeat(num_beams, 1)
        if past_key_values:
            if model.config.is_encoder_decoder:
                last_ids = beams[:, -1]
                if last_ids.dim() == 1:
                    last_ids = last_ids[:,None] 
                outputs = model(repeat_x, decoder_input_ids = last_ids, past_key_values = past_key_values, use_cache = True)
            else:
                last_ids = beams[:, -1]
                if last_ids.dim() == 1:
                    last_ids = last_ids[:,None] 
#                print(last_ids.size())
#                print(past_key_values[0][0].size())
#                xxx = input()
                outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
            if max_num_beams <= 1 and debug == True: ## debug
                last_ids = debug_x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = torch.unsqueeze(last_ids, 0)
                debug_outputs = model(last_ids, past_key_values = debug_past_key_values, use_cache = True)


        else:
            if model.config.is_encoder_decoder:
                outputs = model(repeat_x, decoder_input_ids = beams)
            else:
                outputs = model(beams)
            if max_num_beams <= 1 and debug == True: ## debug
                debug_outputs = model(x)

        token_p = torch.nn.functional.log_softmax(outputs.logits[:, -1, :], dim=-1)
#        token_p[:] = float('-inf')
#        token_p[:,1] = np.log(0.4)
#        token_p[:,12] = np.log(0.6)
        vocab_size = token_p.size(-1)
        if max_num_beams <= 1 and debug == True:
            last_p = outputs.logits[:,-1,:] 
        else:
            last_p = token_p + beam_scores[:,None].expand_as(token_p)
        last_p = norm_logits(last_p.view(1,-1), temperature = temperature, top_k = top_k, top_p = top_p).squeeze()
        prob_list.append(last_p.cpu())

        
        num_beams = random.randint(min_num_beams, max_num_beams) 

        t = sample(last_p.view(1,-1), num_samples = num_beams)
        #print(last_p.nonzero())
        #print(t)
        t = t.view(-1)
        beam_idx = torch.div(t, vocab_size, rounding_mode='floor')
        beam_idx = beam_idx.long().cpu()
        token = t % vocab_size
        token = token[:,None]
        beam_scores = last_p[t].log().view(-1)


#        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = []
        for values in outputs.past_key_values:
            _values = []
            for val in values:
                _values.append(val[beam_idx])
            past_key_values.append(_values)
#        print(past_key_values[0][0].size())

        if model.config.is_encoder_decoder:
            beams = torch.cat((beams[beam_idx], token), dim=1)
        else:
            beams = torch.cat((beams[beam_idx], token), dim=1)

        if max_num_beams <= 1 and debug == True:
            debug_last_p = norm_logits(debug_outputs.logits[::, -1, :], temperature, top_k, top_p)
            debug_past_key_values = outputs.past_key_values

            if (debug_last_p-last_p).abs().sum() > 1e-3:
                print(x)
                print(debug_x)
                print((debug_last_p-last_p).abs().sum())
                print((debug_outputs.logits[::,-1,:]-outputs.logits[:,-1,:]).abs().sum())
                xxx = input("mismatch")
            debug_x = torch.cat((debug_x, token), dim=1)

 #       print(beams.size())
 #       print(num_beams)
        n += 1
        #if idx_next == eos_token_id:
        #    break
        for i in  range(num_beams):
            if beams[i,-1] == eos_token_id:
                output_candidate = beams[i]
                cdd_score = beam_scores[i]/(output_candidate.size(-1)-init_len)
                candidates.append((output_candidate, cdd_score))
                beam_scores[i] = float('-inf')
        if beam_scores.max() < -10000:
            break
    
    for i in  range(num_beams):
        output_candidate = beams[i]
        cdd_score = beam_scores[i]/(output_candidate.size(-1)-init_len)
        candidates.append((output_candidate, cdd_score))


    best_score = -10000
    for cdd, cdd_score in candidates:
        if cdd_score > best_score:
            output = cdd
            best_score = cdd_score
    if max_num_beams <= 1 and debug == True and False:
        output = beams[0]

    if model.config.is_encoder_decoder:
        output = torch.cat((x, output[None,:]), dim=1)
        return output
    else:
        return output[None,:]#, prob_list

