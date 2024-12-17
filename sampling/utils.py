import torch
from torch.nn import functional as F
import pickle
import re
import string
import sqlite3

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = exact_match_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
#    if max(scores_for_ground_truths) < 1:
#        print(prediction, ground_truths)
    return max(scores_for_ground_truths)

def exact_match_references(predictions, references, debug = False):

    em = 0
    for pred, refer in zip(predictions, references):
        score = metric_max_over_ground_truths(pred, refer)
        em += score
        if debug == True and score < 1.:
            print("pred")
            print(pred)
            print("refer")
            print(refer)
    return {"exact_match": 100.0 * em / len(predictions)}

def match(result1, result2):
    str_result1 = set()
    str_result2 = set()
    for row in result1:
        str_row = [str(c) for c in row]
        str_row.sort()
        str_result1.add(tuple(str_row))
    for row in result2:
        str_row = [str(c) for c in row]
        str_row.sort()
        str_result2.add(tuple(str_row))

    return str_result1 == str_result2

def execution_accuracy(db, pred, sql):
    conn = sqlite3.connect(f"/home/qinzongyue/LLMSpeculativeSampling/spider/spider/database/{db}/{db}.sqlite")
    conn.text_factory = bytes
    cur = conn.cursor()
    try:
        gt_result = cur.execute(sql).fetchall()
    except Exception as e:
        return -1

    try:
        result = cur.execute(pred).fetchall()
    except Exception as e:
        return 0
    spider_acc = float(match(result, gt_result))
    return spider_acc


def execution_accuracy_references(predictions, references):
    em = 0
    exception = 0
    for pred, refer in zip(predictions, references):
        db = refer.split("[SQL]")[0]
        sql = refer.split("[SQL]")[1]
        acc = execution_accuracy(db, pred, sql)
        if acc >= 0:
            em += acc
        else:
            exception += 1
    return {"execution accuracy": 100.0 * em / (len(predictions)-exception), "exception": exception}

def get_seq_att_mask(input_cnt, all_input_idx, all_beam_idx, all_next_token, input_len, pad_token_id, device='cpu'):
    """
    Return sequences of shape input_cnt * max_seq_len
    Return tree attention mask of shape input_cnt * max_seq_len * max_seq_len

    """
    ret_seq = [[] for i in range(input_cnt)]
    ret_att_mask = [[] for i in range(input_cnt)]
    position_ids = [[] for i in range(input_cnt)]

    last_beam_att_mask = [[] for i in range(all_input_idx[0].numel())]
    max_seq_len = 0
    pos = [[i,-1] for i in range(input_cnt)]
    position = input_len
    for input_idx_list, beam_idx_list, next_token_list in zip(all_input_idx, all_beam_idx, all_next_token):
        cur_beam_att_mask = []
        for j in range(input_idx_list.numel()):
            input_idx = input_idx_list[j].item()
            next_token = next_token_list[j].item()
            beam_idx = beam_idx_list[j].item()

            cur_seq_len = len(ret_seq[input_idx])
            pos.append([input_idx, cur_seq_len])
            ret_seq[input_idx].append(next_token)
            position_ids[input_idx].append(position)

            if len(ret_seq[input_idx]) > max_seq_len:
              max_seq_len = len(ret_seq[input_idx])
            prev_seq_len = len(last_beam_att_mask[beam_idx])

            mask = last_beam_att_mask[beam_idx] + [False for k in range(cur_seq_len-prev_seq_len)]+[True]
            ret_att_mask[input_idx].append(mask)
            cur_beam_att_mask.append(mask)
        last_beam_att_mask = cur_beam_att_mask
        position += 1

    """ padding """
    for i in range(input_cnt):
      position_ids[i] += [0 for j in range(max_seq_len-len(ret_seq[i]))]
      ret_seq[i] += [pad_token_id for j in range(max_seq_len-len(ret_seq[i]))]

      for j in range(len(ret_att_mask[i])):
        ret_att_mask[i][j] += [False for k in range(max_seq_len-len(ret_att_mask[i][j]))]
      ret_att_mask[i] += [[False for k in range(max_seq_len)] for j in range(max_seq_len-len(ret_att_mask[i]))]

    ret_seq = torch.LongTensor(ret_seq).to(device)
    ret_att_mask = torch.Tensor(ret_att_mask)
    full_att_mask = torch.ones(input_cnt, max_seq_len, max_seq_len+input_len).bool()
    full_att_mask[:,:,input_len:] = ret_att_mask
    full_att_mask = full_att_mask.to(device)
    pos = torch.LongTensor(pos).to(device)
    position_ids = torch.LongTensor(position_ids).to(device)
    return ret_seq, full_att_mask, pos, position_ids


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
#    """ for debugging """
#    return logits

    if top_k is not None and top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p is not None and top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    #assert logits.isnan().any() == False
    ori_logits = logits
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = torch.log_softmax(logits, dim=1).exp()
#    sample(probs, logits = logits)
    #if probs.isnan().any() or probs.isinf().any() or (probs<0).any():
    #    probs = torch.softmax(ori_logits, dim=1)
    if probs.isnan().any() or probs.isinf().any() or (probs<0).any():
        print(torch.logical_not(logits.isinf()).any())
        print(logits[probs.isnan()])
        print(logits[probs.isinf()])
        raise RuntimeError('norm logits error')


    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):
    if torch.numel(probs.nonzero()) < num_samples:
        replacement = True
    else:
        replacement = False
#    """ for debugging """
#    replacement = True
    try:
        idx_next = torch.multinomial(probs, num_samples=num_samples, replacement = replacement)
    except:
        print((probs<0).any(), probs.isnan().any(), probs.isinf().any())
        raise RuntimeError('prob error')

    mask = torch.gather(probs, -1, idx_next) < 1e-3
    if mask.any():
        idx_next[mask] = torch.argmax(probs).item()
   
    #if (idx_next.item() == 0) and False:
    #    raise RuntimeError
    return idx_next % probs.size(-1)


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    if x.dim() > 1:
        x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    else:
        x_max_sum = torch.sum(x_max)
    return x_max / x_max_sum

def get_accept_prob(p,q): # p is large model, q is small model
  acc_p = p/(q+1e-6)
  acc_p = torch.where(acc_p>1, torch.ones_like(acc_p), acc_p)
  return torch.sum(acc_p * q)

def update_large_prob(p,q): # p is large model, q is small model
  new_p = p-q
  new_p = torch.where(new_p<0, torch.zeros_like(new_p), new_p)
  return new_p/(new_p.sum()+1e-6)

def get_first_accept_prob(p_list,q,m,idx):
  # given large model p, small model q, and m draft tokens
  # compute the probability of the first accepted token is token [idx]
  # idx value range is 1 to m
  assert idx <= m
  prob = 1.
#  cur_p = p.clone()
  for i in range(idx-1):
    cur_p = p_list[i]
    if cur_p.isnan().any():
        print('get_first_accept_prob, cur_p is nan')
    alpha_i = get_accept_prob(cur_p, q)
    if alpha_i.isnan():
        print('alpha_i is nan')
    prob *= (1-alpha_i)
#    cur_p = update_large_prob(cur_p, q)
  cur_p = p_list[idx-1]
  if cur_p.isnan().any():
      print('get_first_accept_prob, cur_p is nan')

  alpha_i = get_accept_prob(cur_p, q)
  if alpha_i.isnan():
      print('alpha_i is nan')
  prob *= alpha_i
  return prob

def get_all_reject_prob(p_list,q,m):
  prob = 1.
#  cur_p = p.clone()
  for i in range(m):
    cur_p = p_list[i]
    alpha_i = get_accept_prob(cur_p, q)
    prob *= (1-alpha_i)
 #   cur_p = update_large_prob(cur_p, q)
  return prob

def get_prob_for_accept_k_tokens(p_list,q,m,k,history):
  assert m >= 0

  if m < k:
    return 0
  if m == 0 and k == 0:
    return 1
  if history[m-1][k-1] >= 0:
    return history[m-1][k-1]
  if k == 0:
    return get_all_reject_prob(p_list,q,m)
  prob = 0
  for i in range(1,m+1):
    A = get_first_accept_prob(p_list,q,m,i)
    B = get_prob_for_accept_k_tokens(p_list,q,m-i,k-1,history)
    prob += A * B
    if prob < 0 or prob.isnan():
        print(prob)
        print(A)
        print(B)
        xxx = input('error')
  history[m-1][k-1] = prob
  return prob

def get_expect_cnt_by_thres(p_width, expect_thres, n):
    cum_p = p_width[n]
    while cum_p < expect_thres and n > 0:
        #print(n, cum_p)
        n -= 1
        cum_p += p_width[n]
    return int(n)

def get_num_acc_prob(p,q,m):
    beta_list = torch.zeros(m, dtype=p.dtype).to(p.device)
    P = torch.zeros(m+1, m+1, dtype=p.dtype).to(p.device)
    P[0,0] = 1
    acc_rej_prob = 1.

    cur_p = p.clone()
    for i in range(1,m+1):
        acc_p = cur_p/(q+1e-6)
        acc_p[acc_p>1] = 1.
        alpha = torch.sum(acc_p * q)

        beta_list[i-1] = acc_rej_prob * alpha
        P[i,0] = acc_rej_prob * (1-alpha)
        acc_rej_prob *= (1-alpha)

        new_line = torch.squeeze(beta_list[:i].flip(dims=[0]).view(1,-1) @ P[:i])
        P[i,1:] = new_line[:-1]

        cur_p = update_large_prob(cur_p, q)
        cur_p = cur_p-q
        cur_p[cur_p < 0] = 0
        cur_p = cur_p / (cur_p.sum() + 1e-6)

    return P[m], torch.sum(torch.arange(m+1).float().to(P.device) * P[m])

def extract_first_function(code_string):
    """
    Extract the first function definition from a Python code string.
    Handles imports, docstrings, and type hints.
    
    Args:
        code_string (str): String containing Python code
        
    Returns:
        str: The first function definition found, or empty string if none found
    """
    import re
    
    # Split code into lines while preserving empty lines
    lines = code_string.split('\n')
    
    # Initialize variables
    function_lines = []
    in_function = False
    base_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and imports before function
        if not in_function and (not stripped or stripped.startswith('from ') or stripped.startswith('import ')):
            if stripped.startswith(('from ', 'import ')):
                function_lines.append(line)
            i += 1
            continue
            
        # Check for function definition
        if not in_function and re.match(r'^\s*def\s+\w+.*:', line):
            in_function = True
            base_indent = len(line) - len(line.lstrip())
            function_lines.append(line)
            i += 1
            continue
            
        # If we're in a function
        if in_function:
            # Skip empty lines
            if not stripped:
                function_lines.append(line)
                i += 1
                continue
                
            current_indent = len(line) - len(line.lstrip())
            # If we hit a line with same or less indent (except empty lines), we're done
            if current_indent <= base_indent and stripped:
                break
                
            function_lines.append(line)
            i += 1
            continue
            
        i += 1
    
    return '\n'.join(function_lines)

