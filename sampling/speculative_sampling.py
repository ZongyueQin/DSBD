import torch
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from sampling.utils import get_num_acc_prob, get_expect_cnt_by_thres
from time import process_time_ns
import numpy as np
import os
from sampling.utils import get_seq_att_mask
from collections import Counter


""" beam speculative decoding with tree attention """
@torch.no_grad()
def beam_speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id, max_len : int , gamma : int = 4, width : int = 8, 
                         num_beams: int = 8, min_num_beams: int = 1, extra_sample_cnt: int = -1,
                         expect_thres: float = 0.7,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False, debug_dict = None) -> torch.Tensor:

    #print(prefix)
    #xxx = input()
    """ extra_sample_cnt can only be 1 or num_beams """
    if extra_sample_cnt != 1:
        extra_sample_cnt = num_beams

    assert approx_model.config.is_encoder_decoder == False
    assert target_model.config.is_encoder_decoder == False

    if pad_token_id is None:
        pad_token_id = eos_token_id

    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    acc_len = []
    acc_rate = []

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    #debug_target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    assert prefix.shape[0] == 1, "input batch size must be 1"

    approx_time = 0
    target_time = 0
    sample_time = 0
    target_call_times = 0
    approx_call_times = 0
    compute_expect_time = 0
    d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times,
                'target_model_time': 0,
                'target_pre_cache_time': 0,
                'target_post_prob_time': 0
            }
    num_beams_list = []

    output_prefix = prefix
    init_len = seq_len

    start_t = process_time_ns()

#    with tqdm(total=T, desc="speculative sampling") as pbar:
    first_input = True
    candidates = []
    acc_sample_list = []
    extra_sample_list = []
    expect_cnt_list = []

    try:
        while output_prefix.shape[1] < T:
            prefix_len = output_prefix.shape[1]

            # generate x of size width * (prefix_len+gamma)
            tt = process_time_ns()

            ret = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       return_intermediate_results = True,
                       output_scores = True,
                       ret_seq_scores = True,
                       optimization=False,
                       )
            out, all_seq, all_beam_idx, all_next_token, all_score, all_prob, all_input_idx = ret[0],ret[1],ret[2],ret[3],ret[4], ret[5], ret[6]
            if extra_sample_cnt == 1:
                for i in range(len(all_input_idx)):
                    all_input_idx[i][:] = 0


            #max_len = all_seq[-1].size(1)
            #x = [F.pad(seq, (0,max_len-seq.size(1),0,0), 'constant', pad_token_id) for seq in all_seq]
            #att_mask = [F.pad(torch.ones_like(seq), (0,max_len-seq.size(1),0,0), 'constant', pad_token_id) for seq in all_seq]

            #x = torch.concat(x, dim=0)
            #att_mask = torch.concat(att_mask, dim=0)
            
            #x = out['sequences'] # width * (prefix_len+gamma)
            #q, seq_q = out['scores'] # tuples of gamma * (width * vocab) ?
 
            inc_len = len(all_next_token)
            approx_call_times += 1
            approx_time += process_time_ns() - tt


            tt = process_time_ns()
            # compute extra_attention_mask
            input_cnt = extra_sample_cnt
            out_seq, extra_att_mask, pos, position_ids = get_seq_att_mask(input_cnt, 
                                                                          all_input_idx[1:], 
                                                                          all_beam_idx, 
                                                                          all_next_token, 
                                                                          prefix_len, 
                                                                          pad_token_id, 
                                                                          device=output_prefix.device)
            if extra_sample_cnt == 1:
                idx = torch.zeros(num_beams).long().cpu()
                pos = torch.concat((pos[idx], pos[1:]), dim=0)

            p = target_model_cache.forward_tree_attention(out_seq,
                                                          output_prefix,
                                                          extra_att_mask,
                                                          position_ids,
                                                          pos,
                                                          )
#            print(p.size())
            target_call_times += 1
            """ for debugging """
            """ 
            concat_all_input_idx = torch.concat(all_input_idx, dim=0)
            _ = debug_target_model_cache.generate(x, 1, attention_mask = att_mask, copy_cache_index = concat_all_input_idx)
 #           print(x)

            gt_p = debug_target_model_cache._prob_history
            start = 0
            for i in range(inc_len):
                end = start + num_beams
                cur_p = p[start:end]
                err = torch.sum(torch.abs(cur_p - gt_p[start:end, prefix_len+i-1].squeeze()))
#                print(err)
                if err > 0.01:
                    print(start,end,prefix_len+i-1)
                    print(err)
                    print(torch.sum(torch.abs(cur_p - gt_p[start:end, prefix_len+i-1].squeeze()), dim=-1))
                    xxx = input('error too large')
                start += num_beams
            """

            vocab_size = p.size(-1)
            
            target_time += process_time_ns() - tt

            

            """ verification process """
            tt = process_time_ns()
           
            if extra_sample_cnt == 1:
                first_input = True

            if first_input == True:
                """ for the first input, make all the beams into the first beam """
                cur_valid_beam = torch.zeros_like(all_beam_idx[0])
                cur_valid_beam[0] = 1
                cur_valid_beam = cur_valid_beam.bool()
                beam_scores = torch.zeros_like(all_score[0])
            else:
                cur_valid_beam = torch.ones_like(all_beam_idx[0]).bool()

            n = prefix_len - 1

            max_l = 0
            start = 0
            for i in range(inc_len):
                end = start + num_beams
                cur_beam_idx = all_beam_idx[i]
                #print(cur_beam_idx)
                # get sampled distribution of the small model
                q_scores = all_score[i]
                q_prob = all_prob[i]
                #print(q_scores)
                # speacial treatment for i==0
                if first_input:
                    """ for the first input, make all the beams the first beam """
                    cur_beam_idx[:] = 0
                    q_scores = q_scores * num_beams
                    q_prob = q_prob * num_beams
                    q_prob[vocab_size:] = 0
                    first_input = False

                """ shift cur_beam_idx by cur_valid_beam """
                shift = torch.cumsum(cur_valid_beam.long(),dim=0)-1
                shift_beam_idx = shift[cur_beam_idx]

                """ Step 1 get sampling distribution of the large model """

                cur_p = p[start:end]
                cur_p = cur_p[cur_valid_beam] # shape: num_valid_beam * V
                from_valid_beam = cur_valid_beam[cur_beam_idx]

                p_next_token_scores = beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()
                #""" for debug """
                #cur_p[:] = 0
                #cur_p[:,1] = 0.6 
                #cur_p[:,12] = 0.4 
                #p_next_token_scores = 0*beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()

                p_next_token_scores = norm_logits(p_next_token_scores.view(1,-1), temperature, top_k, top_p).view(-1) 


                cur_p_prob = p_next_token_scores
                q_prob = q_prob.view(num_beams, -1)
                q_prob = q_prob[cur_valid_beam]
                q_prob = q_prob.view(-1)

                """ determine if there is any current beam is sampled from previous valid beams """
                #TODO I need to minus q for the time that samples are not from valid beams

                if True: 
                    shift_beam_idx = torch.clamp(shift_beam_idx, min=0)
                    cur_sample_idx = shift_beam_idx * vocab_size + all_next_token[i]


 #                   print('q prob', q_prob[:15])
                    valid_beam_cnt = from_valid_beam.float().sum()


                    #expect_cnt = torch.sum(valid_beam_cnt * q_prob * torch.clamp(p_next_token_scores/(q_prob+1e-6), 
                    #              max=1)).floor().item()
                    ttt = process_time_ns() 
                    
                    p_width, e_width = get_num_acc_prob(p_next_token_scores, 
                                                        q_prob,
                                                        num_beams,
                                                        # use num beams because q_score 
                                                        # is not normalized after
                                                        # selection
                                                        )
                    compute_expect_time += process_time_ns() - ttt
                    #expect_cnt = torch.multinomial(p_width, 1).item()
                    if expect_thres < 0:
                        expect_cnt = e_width.floor().item()
                    else:
                        expect_cnt = get_expect_cnt_by_thres(p_width, expect_thres, num_beams)
                    expect_cnt = max(expect_cnt, min_num_beams)
                    
                    #expect_cnt = 1

                    #expect_cnt = 2
                    expect_cnt_list.append(expect_cnt)


                    #r = torch.rand(1, device = p.device)-1e-5
                    #p_scores = q_scores
                    #accept = (p_scores/(q_scores+1e-5)) > r
                    # go through each beam one by one
                    accept = from_valid_beam.clone()
                    acc_cnt = 0
                    for j in range(num_beams):
                        p_score = cur_p_prob[cur_sample_idx[j]]
                        r = torch.rand(1, device=p.device) 
  #                      print(r, p_score, q_scores[j])
                        #xxx = input()
                        if acc_cnt >= expect_cnt:
                            accept[j] = False
                            continue
                        if accept[j] == True: # from a valid beam
                            accept[j] = (p_score/(q_scores[j]+1e-6)) > r

                        if accept[j] == False:
                            # change accept rate
#                            print(f'{cur_sample_idx[j]} rejected')
#                            print(cur_p_prob[:20])
                            cur_p_prob = max_fn(cur_p_prob-q_prob)
 #                           print('residual:')
  #                          print(cur_p_prob[:20])

                        else:
   #                         print(f'{cur_sample_idx[j]} accepted')

                            cur_p_prob = p_next_token_scores
                            acc_cnt += 1
                            acc_sample_list.append(cur_sample_idx[j].item())
    #                        print(cur_p_prob[:20])
     #           print('after verification')
      #          print(cur_p_prob[:20])
       #         print(p_next_token_scores[:20])
        #        print(cur_sample_idx[accept])
         #       xxx = input('pause')

             

                
                acc_r = accept.float().mean().item()
                acc_rate.append(acc_r)


                #if acc_cnt >= min_num_beams:
                if acc_cnt >= expect_cnt:  
                    assert acc_cnt == expect_cnt
                    num_beams_list.append(acc_cnt)
                    # Step 5 update cur_valid_beam
                    cur_valid_beam = accept
                    p_scores = torch.gather(p_next_token_scores, dim=0, index=cur_sample_idx)
                    #mask = from_valid_beam.logical_not()
                    #p_scores[mask] = 0
                    p_scores[torch.logical_not(accept)] = 0

                    beam_scores = p_scores.log() 
                    n += 1
                    max_l += 1
                    start = end
                else:
                    # if all draft are rejected, terminate and start re-sample
                    num_beams_list.append(num_beams)
                    break

            #TODO re-sample based on cur_valid_beam and p
            end=start + num_beams
            acc_len.append(max_l)
     #       print('max_l',max_l)
     #       print('inc len', inc_len)
            
            
            if max_l == inc_len: # all accept
                
                cur_p = p[start:end]
                cur_p = cur_p[cur_valid_beam]
                p_next_token_scores = beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()
                op = p_next_token_scores 
                p_next_token_scores = norm_logits(p_next_token_scores.view(1,-1), temperature = temperature, top_k = top_k, top_p = top_p).squeeze()
#                """ for debug """
#                cur_p[:] = 0
#                cur_p[:,1] = 0.6 
#                cur_p[:,12] = 0.4 
#                p_next_token_scores = 0*beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()
#                p_next_token_scores = norm_logits(p_next_token_scores.view(1,-1), temperature = temperature, top_k = top_k, top_p = top_p).squeeze()
#                print('max_l == inc_len')
#                print(p_next_token_scores[:20])

                """ sample next token """

                #try:
                t = sample(p_next_token_scores, num_samples = extra_sample_cnt)
      #          print(p_next_token_scores[:15])
                #xxx = input()
                #except:
                #    t = sample(torch.softmax(op.view(-1), dim=0), num_samples = extra_sample_cnt)

                beam_idx = torch.div(t, vocab_size, rounding_mode='floor')
                beam_idx = beam_idx.long()
                token = t % vocab_size
                token = token[:,None]
                beam_scores = p_next_token_scores[t].log().view(-1)


                choice = cur_valid_beam.nonzero()[beam_idx].squeeze()
                output_prefix = all_seq[start//num_beams][choice, :n+1]

                if pos[:,1].max() > inc_len:
                    pos[:,1] -= prefix_len 
                
                acc_pos = pos[start+choice]

                if extra_sample_cnt == 1:
                    output_prefix = output_prefix[None,:]
                    acc_pos = acc_pos[None,:]

                #print(output_prefix.size(), token.size())
                output_prefix = torch.concat([output_prefix, token], dim=1)
                accepted_input_idx = acc_pos[:,0]
                accepted_mask = extra_att_mask[acc_pos[:,0], acc_pos[:,1]]
                target_model_cache.rollback_tree_attention(accepted_input_idx.to(target_model.device), accepted_mask)



                #debug_target_model_cache._past_key_values = None
            else:
                """ 
                t = sample(cur_p_prob, num_samples = extra_sample_cnt)
                """
                if extra_sample_cnt > 1:
                    """
                    print('cur valid beam')
                    print(cur_valid_beam)
                    print('cur beam idx')
                    print(cur_beam_idx)
                    print('accept')
                    print(accept)
                    print('shift beam idx')
                    print(shift_beam_idx)
                    print('cur sample idx')
                    print(cur_sample_idx)
                    print(cur_sample_idx[accept])
                    """
                    #cur_p = p[start:end]
                    #cur_p = cur_p[cur_valid_beam]
                    #p_next_token_scores = beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()
                    op = p_next_token_scores 
                    #p_next_token_scores = norm_logits(p_next_token_scores.view(1,-1), temperature = temperature, top_k = top_k, top_p = top_p).squeeze()
                    t = sample(p_next_token_scores, num_samples = extra_sample_cnt)
                    if acc_cnt > 0:
                        acc_token = cur_sample_idx[accept] 
                        t[:acc_cnt] = acc_token
                 #       print(t.size(), acc_token.size())

                    new_t = sample(cur_p_prob, num_samples = 1)
                    t[acc_cnt] = new_t
#                    print('else')
#                    print(cur_p_prob[:20])
#                    print(p_next_token_scores[:20])
#                    print(t)
                else:
                   # print(extra_sample_cnt)
                   # print(cur_p_prob[:20])
                   # xxx = input()
                    t = sample(cur_p_prob, num_samples = extra_sample_cnt)
#                    t = sample(p_next_token_scores, num_samples = extra_sample_cnt)
               
                


                beam_idx = torch.div(t, vocab_size, rounding_mode='floor')
                beam_idx = beam_idx.long()
                token = t % vocab_size
                token = token[:,None]

                choice = cur_valid_beam.nonzero()[beam_idx].squeeze()
                output_prefix = all_seq[start//num_beams][choice, :n+1]

                if pos[:,1].max() > inc_len:
                    pos[:,1] -= prefix_len 
                acc_pos = pos[start+choice]
                if extra_sample_cnt == 1:
                    output_prefix = output_prefix[None,:]
                    acc_pos = acc_pos[None,:]

                accepted_input_idx = acc_pos[:,0]
                accepted_mask = extra_att_mask[acc_pos[:,0], acc_pos[:,1]]
                if pos[:,1].min() == -1:
                    accepted_mask[:,prefix_len:] = False


                beam_scores = p_next_token_scores[t].log().view(-1)

                #print(output_prefix.size(), token.size())
                output_prefix = torch.concat([output_prefix, token], dim=1)
                target_model_cache.rollback_tree_attention(accepted_input_idx, accepted_mask)
                #debug_target_model_cache._past_key_values = None

#            xxx = input('extra sample done')

            if max_l == inc_len:
                last_beam_idx = all_beam_idx[-1] 
                approx_model_cache.beam_rollback(max_l, last_beam_idx[choice%num_beams])
            else:
                approx_model_cache.beam_rollback(max_l, choice%num_beams)

            cur_valid_beam = torch.ones_like(all_beam_idx[0]).bool()


            # check each sequence for eos_token, if there is, save it as one of the candidates, continue the search
            mask = (output_prefix == eos_token_id)
            end_cnt = 0
            for i in range(mask.size(0)):
                if mask[i].int().sum() > ori_eos_cnt: #encounter eos
                    if extra_sample_cnt > 1:
                        end_cnt += 1
                        row_mask = torch.cumsum(mask[i].float(), dim=0)
                        row_mask = (row_mask < ori_eos_cnt+1)
                        end = row_mask.int().sum()
                        if end < mask.size(1):
                            row_mask[end] = True
                        output_candidate = output_prefix[i][row_mask] 

                        cdd_score = beam_scores[i]/(output_candidate.size(-1) - init_len)
                        cur_valid_beam[i] = False
                        candidates.append((output_candidate, cdd_score))
                    else:
                        end_cnt = 1000
                        row_mask = torch.cumsum(mask[i].float(), dim=0)
                        row_mask = (row_mask < ori_eos_cnt+1)
                        end = row_mask.int().sum()
                        if end < mask.size(1):
                            row_mask[end] = True
                        output_prefix = output_prefix[i][row_mask].view(1,-1)
                        break
            if end_cnt >= mask.size(0):
                break
            



            sample_time += process_time_ns() - tt
    except Exception as e:
        print(e)
        raise RuntimeError('')

    if extra_sample_cnt > 1:
        for i in range(output_prefix.size(0)):
            output_candidate = output_prefix[i]
            cdd_score = beam_scores[i]/(output_candidate.size(-1)-init_len)
            candidates.append((output_candidate, cdd_score))

        best_score = -10000
        for cdd, cdd_score in candidates:
            if cdd_score > best_score:
                output_prefix = cdd
                best_score = cdd_score
    else:
        output_prefix = output_prefix[0]


    if approx_model.config.is_encoder_decoder:
        output_prefix = torch.cat((prefix, output_prefix[None,:]), dim=1)
    else:
        output_prefix = output_prefix[None,:]


    if verbose:
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times,
                'num_beams_list': num_beams_list,
                'target_model_time': target_model_cache.forward_time_dict['_model_time'],
                'target_pre_cache_time': target_model_cache.forward_time_dict['prepare_cache_time'],
                'target_post_prob_time': target_model_cache.forward_time_dict['norm_prob_time'],
                'compute_expect_time': compute_expect_time,
                'expect_cnt_list': expect_cnt_list,
            }
        return output_prefix, d
    else:
        return output_prefix



@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id,
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    #assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    approx_time = 0
    target_time = 0
    sample_time = 0
    approx_call_times = 0
    target_call_times = 0

    start_t = process_time_ns()

    acc_rate = []
    acc_len = []

    if pad_token_id is None:
        pad_token_id = eos_token_id
    decoder_input_ids = torch.LongTensor([[pad_token_id]]).to(prefix.device)

    try:
        while prefix.shape[1] + decoder_input_ids.shape[1] - 1 < T:
            #print('loop')
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            tt = process_time_ns()

            # TODO for debug
#            approx_model_cache.reset_cache()
        
            if approx_model.config.is_encoder_decoder == False:
                x = approx_model_cache.generate(prefix, gamma)
                prefix_len = prefix.shape[1]
            else:
                x = approx_model_cache.generate(prefix, gamma, decoder_input_ids = decoder_input_ids)
                prefix_len = decoder_input_ids.shape[1]

            x = x.to(device)
            approx_call_times += 1
        
            approx_time += process_time_ns() - tt
            tt = process_time_ns()
        
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1)
            else:
                _ = target_model_cache.generate(prefix, 1, decoder_input_ids = x)
            target_call_times += 1

            target_time += process_time_ns() - tt
            tt = process_time_ns()
        
            n = prefix_len + gamma - 1

            for i in range(gamma):
                j = x[:, prefix_len + i].item()

                acc_rate.append(((target_model_cache._prob_history[:, prefix_len + i - 1, j].item()) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j].item())))
                if acc_rate[-1] > 1:
                    acc_rate[-1] = 1
        

            l = 0
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = device)
                j = x[:, prefix_len + i].item()
            
                if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j].item()) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j].item()):
                    # reject
                    n = prefix_len + i - 1
                    break
            
                if verbose:
                    print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

                accepted_count += 1
                l += 1
            acc_len.append(l)
        
            # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            if approx_model.config.is_encoder_decoder == False:
                prefix = x[:, :n + 1]
            else:
                decoder_input_ids = x[:, :n+1]
        
            approx_model_cache.rollback(n+1)
            #print('after roll back')
            #print(approx_model_cache._prob_history.size())
            assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
            if n < prefix_len + gamma - 1:
                # reject someone, sample from the pos n
                try:
                    t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
                except:
                    t = sample(max_fn(target_model_cache._prob_history[:, n, :]))

                if verbose:
                    print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
                resample_count += 1
                target_model_cache.rollback(n+1)
            else:
                 # all approx model decoding accepted
                assert n == target_model_cache._prob_history.shape[1] - 1
                t = sample(target_model_cache._prob_history[:, -1, :])
                if verbose:
                    print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
                target_sample_count += 1
                target_model_cache.rollback(n+2)
        
        
            if approx_model.config.is_encoder_decoder == False:
                prefix = torch.cat((prefix, t), dim=1)
                out = prefix
            else:
                decoder_input_ids = torch.cat((decoder_input_ids, t), dim=1)
                out = decoder_input_ids

            mask = (out == eos_token_id)
            if mask.int().sum() > ori_eos_cnt:
                mask = torch.cumsum(mask.float(), dim=1)
                mask = (mask < ori_eos_cnt+1)
                end = mask.int().sum()
                if end < mask.size(1):
                    mask[:, end] = True
                out = out[mask][None,:] 
                break

            sample_time += process_time_ns() - tt
    except Exception as e:
        print(e)
        raise RuntimeError('s')

        #print(f'n={n}, l={l}')

    if approx_model.config.is_encoder_decoder == True:
        out = torch.cat((prefix, out), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print(f"Acc rate: {np.mean(acc_rate)}")
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times,
                'target_model_time': target_model_cache.forward_time_dict['_model_time'],
                'target_pre_cache_time': target_model_cache.forward_time_dict['prepare_cache_time'],
                'target_post_prob_time': target_model_cache.forward_time_dict['norm_prob_time'],
            }
        return out, d
    else:
        return out


@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None, details : bool = False) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    approx_time = 0
    target_time = 0
    sample_time = 0


    acc_rate = []
    acc_len = []


    #with tqdm(total=T, desc="speculative sampling") as pbar:
    if True:
        while prefix.shape[1] < T:
            tt = process_time_ns()
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            approx_time += process_time_ns() - tt
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            tt = process_time_ns()
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            target_time += process_time_ns() - tt

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            tt = process_time_ns()
            
            is_all_accept = True
            n = prefix_len - 1
            l = 0
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                acc_rate.append(torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]).item())
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                    l += 1
                else:
                    # reject
                    try:
                        t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    except Exception as e:
                        print(e)
                        print('reject, r, t: ', r, p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j])
                        print(prefix_len+i-1, n)
                        print(torch.sum(p[:,n,:]), torch.sum(q[:,n,:]))
                        print(torch.sum(((p[:,n,:]-q[:,n,:])>0).float()))
                        raise RuntimeError(f'{e}')

                    is_all_accept = False
                    break
            acc_len.append(l)
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            sample_time += process_time_ns() - tt
#            pbar.update(n - pbar.n)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate)
            }
        return prefix, d
    else:
        return prefix


