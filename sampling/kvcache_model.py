import torch
import os
from typing import Optional
import numpy as np

from sampling.utils import norm_logits, sample
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers import GenerationConfig, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, BeamSearchScorer, StoppingCriteriaList, BeamScorer
import copy
from typing import Union, List
import torch.nn as nn
from transformers.generation import BeamSampleDecoderOnlyOutput, BeamSampleEncoderDecoderOutput
from time import process_time_ns

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break

class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.beam_rollback_flag = False
        self.forward_time_dict = {'_model_time':0,
                                  'norm_prob_time':0,
                                  'prepare_cache_time':0
                                  }

    def forward_tree_attention(self, 
                               input_ids : torch.Tensor,
                               prefix: torch.Tensor,
                               extra_attention_mask: torch.Tensor,
                               position_ids: torch.Tensor,
                               gather_pos: torch.Tensor,
                               ) -> torch.Tensor:
        assert self._model.config.is_encoder_decoder == False
        prefix_len = prefix.size(-1)
        if self._past_key_values is None:
            tt = process_time_ns()
            if prefix.size(0) < input_ids.size(0):
                assert prefix.size(0) == 1
                input_ids = torch.concat((prefix.repeat(input_ids.size(0),1), input_ids), dim=1)
            else:
                input_ids = torch.concat((prefix, input_ids), dim=1)

            if position_ids.size(1) < input_ids.size(1):
                position_ids = torch.concat(
                                      (torch.arange(prefix_len).to(input_ids.device).view(1,-1).repeat(position_ids.size(0),1),
                                      position_ids),
                                      dim=1,
                                      )

#            print(input_ids)
#            print(extra_attention_mask)
#            print(position_ids)
            
            outputs = self._model(input_ids, extra_attention_mask = extra_attention_mask, position_ids = position_ids)

            self.forward_time_dict['_model_time'] += process_time_ns() - tt
            tt = process_time_ns()

            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
#                self._prob_history[:, i, :] = float('-inf')
#                self._prob_history[:, i, 1] = np.log(0.4)
#                self._prob_history[:, i, 12] = np.log(0.6)
#                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], 1, None, None)
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)

            self._past_key_values = outputs.past_key_values
            self.forward_time_dict['norm_prob_time'] += process_time_ns() - tt
        else:
            tt = process_time_ns()
            # return the last token's logits
            cached_len = self._past_key_values[0][0].shape[2]

            assert self._past_key_values[0][0].size(0) == input_ids.size(0)

            input_ids = torch.concat((prefix, input_ids), dim=1)
            last_input_id = input_ids[:, cached_len:]
            position_ids = torch.concat(
                                      (torch.arange(prefix_len).to(input_ids.device).view(1,-1).repeat(position_ids.size(0),1),
                                      position_ids),
                                      dim=1,
                                      )
            position_ids = position_ids[:, cached_len:]


            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            if position_ids.dim() == 1:
                position_ids = torch.unsqueeze(position_ids, 0)


            
            outputs = self._model(last_input_id, 
                                  past_key_values=self._past_key_values, 
                                  use_cache=True, 
                                  extra_attention_mask = extra_attention_mask,
                                  position_ids = position_ids,
                                  )
            self.forward_time_dict['_model_time'] += process_time_ns() - tt

            tt = process_time_ns()

            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                

            for i in range(not_cached_q.shape[-2]):   
#                not_cached_q[:, i, :] = float('-inf')
#                not_cached_q[:, i, 1] = np.log(0.4)
#                not_cached_q[:, i, 12] = np.log(0.6)
#                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], 1, None, None)    
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            self._past_key_values = outputs.past_key_values
            self.forward_time_dict['norm_prob_time'] += process_time_ns() - tt
       
        gather_pos[:,1] += prefix_len
        p = self._prob_history[gather_pos[:,0], gather_pos[:,1]]

        return p




    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = False,
            attention_mask = None,
            decoder_input_ids = None,
            copy_cache_index = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if copy_cache_index is not None:
            copy_cache_index = copy_cache_index.cpu()

        if self._past_key_values is None:
            #assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            tt = process_time_ns() 
            if self._model.config.is_encoder_decoder == False:
#                outputs = self._model(input_ids, attention_mask=attention_mask)
                outputs = self._model(input_ids)

            else:
#                outputs = self._model(input_ids, decoder_input_ids = decoder_input_ids, use_cache=True, decoder_attention_mask = attention_mask)
                #print(input_ids.size(), decoder_input_ids.size())
                outputs = self._model(input_ids, decoder_input_ids = decoder_input_ids)

            self.forward_time_dict['_model_time'] += process_time_ns() - tt
            tt = process_time_ns()

            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
            self.forward_time_dict['norm_prob_time'] += process_time_ns() - tt
        else:
            tt = process_time_ns()
            # return the last token's logits
            cached_len = self._past_key_values[0][0].shape[2]
#            for values in self._past_key_values:
#                cached_len = values[0].shape[2]
#                break #added by zongyue

            num_beams = input_ids.size(0)
#            if self._past_key_values[0][0].size(0) == 1 and num_beams > 1: # handle batch input with single past_key_values
            if self._past_key_values[0][0].size(0) < num_beams:
                #TODO handle multiple beams, need to record the beams each input correspond to
                if copy_cache_index is None:
                    repeats = int(num_beams/self._past_key_values[0][0].size(0))
                    past_key_values = []
                    for values in self._past_key_values:
                        repeat_values = []
                        for val in values:
                            repeat_values.append(val.repeat(num_beams, 1, 1, 1))
                        past_key_values.append(tuple(repeat_values))
                    self._past_key_values = past_key_values
                else:
                    past_key_values = []
                    for values in self._past_key_values:
                        repeat_values = []
                        for val in values:
                            repeat_values.append(val[copy_cache_index])
                        past_key_values.append(tuple(repeat_values))
                    self._past_key_values = past_key_values
            self.forward_time_dict['prepare_cache_time'] += process_time_ns() - tt
            tt = process_time_ns()

               
            if self._model.config.is_encoder_decoder == False:
                last_input_id = input_ids[:, cached_len:]
                if last_input_id.dim() == 1:
                    last_input_id = torch.unsqueeze(last_input_id, 0)
            
                if use_debug:
                    print(f"last_input_id shape {last_input_id.shape}")
                    _debug_show_kvcache(self._past_key_values)
            
                outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            else:
                last_input_id = decoder_input_ids[:, cached_len:]
                if last_input_id.dim() == 1:
                    last_input_id = torch.unsqueeze(last_input_id, 0)
            
                if use_debug:
                    print(f"last_input_id shape {last_input_id.shape}")
                    _debug_show_kvcache(self._past_key_values)
            
                outputs = self._model(input_ids, decoder_input_ids = last_input_id, past_key_values=self._past_key_values, use_cache=True)
            self.forward_time_dict['_model_time'] += process_time_ns() - tt

            tt = process_time_ns()

            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                

            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
#            if self._prob_history.size(0) == 1 and num_beams > 1:
            if self._prob_history.size(0) < num_beams:
                if copy_cache_index is None:
                    repeats = int(num_beams/self._prob_history.size(0))
                    self._prob_history = self._prob_history.repeat(repeats, 1, 1)
                else:
                    self._prob_history = self._prob_history[copy_cache_index]

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
            self.forward_time_dict['norm_prob_time'] += process_time_ns() - tt
       
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    decoder_input_ids = None,
                                    multi: int = 1,
                                    strategy: str = "beam",
                                    attention_mask = None,
                                    copy_cache_index = None,
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix # size (bs, seq_len)
        if strategy == "iid" and multi > 1:
            x = x.repeat(multi, 1)
            if self._model.config.is_encoder_decoder == True and decoder_input_ids.size:
                decoder_input_ids = decoder_input_ids.repeat(multi, 1)


        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug, decoder_input_ids = decoder_input_ids, 
                    attention_mask = attention_mask, copy_cache_index = copy_cache_index)
            if multi == 1:
                next_tok = sample(q)
            else:
                if strategy == "beam":
                    raise NotImplementedError 
                elif strategy == "iid":
                    next_tok = sample(q)
                else:
                    raise RuntimeError("Strategy Not Implemented "+strategy)
                
            if self._model.config.is_encoder_decoder == False:
                x = torch.cat((x, next_tok), dim=1)
                ret = x
                #print('next tok', next_tok, q.size())
            else:
                decoder_input_ids = torch.cat((decoder_input_ids, next_tok), dim=1)
                ret = decoder_input_ids
        return ret

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int,
                 decoder_input_ids = None,
                 attention_mask = None,
                 copy_cache_index = None,
                 multi: int = 1, strategy: str = "beam") -> torch.Tensor:
        if self._model.config.is_encoder_decoder == True and decoder_input_ids is None:
            raise RuntimeError("It is an encoder-decoder model, please set decoder_input_ids")
        output = self._generate_with_kvcache(input, gamma, decoder_input_ids = decoder_input_ids, multi=multi, strategy=strategy, 
                attention_mask = attention_mask, copy_cache_index = copy_cache_index)
        return output

    @torch.no_grad()
    def beam_rollback(self, beam_idx, choice):
        
        #self.beam_rollback_flag = False
        assert beam_idx >= 0
        if beam_idx == len(self.beam_past_key_values):
#            self._past_key_values = None
#            return
            self._past_key_values = self.beam_past_key_values[beam_idx-1] 
        else:
            self._past_key_values = self.beam_past_key_values[beam_idx]

        self.rollback(None, choice)

    @torch.no_grad()
    def rollback_tree_attention(self, input_idx, mask):
        assert isinstance(self._model, BloomForCausalLM) == False
        num_beams, num_head, _, emb_dim = self._past_key_values[0][0].size()
        width = input_idx.numel()
        past_key_values_trimmed = []
        
        device = self._past_key_values[0][0].device
        mask = mask.to(device)
        input_idx_cpu = input_idx.cpu()

        expand_mask = mask.unsqueeze(1).expand(-1, num_head, -1).cpu()
        for kv in self._past_key_values:
            k, v = kv
            k = k[input_idx_cpu][expand_mask].view(width, num_head, -1, emb_dim)
            v = v[input_idx_cpu][expand_mask].view(width, num_head, -1, emb_dim)
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)

        self._past_key_values = past_key_values_trimmed
        if self._prob_history is not None:
            vocab_size = self._prob_history.size(-1)
            self._prob_history = self._prob_history[input_idx_cpu][mask.cpu()].view(width, -1, vocab_size)

    @torch.no_grad()
    def rollback(self, end_pos, choice=None):
        if choice is not None and isinstance(choice, int) == False:
            choice = choice.cpu()
        past_key_values_trimmed = []
        assert self._past_key_values
        if self._model.config.is_encoder_decoder == False:
            for kv in self._past_key_values:
                k, v = kv
                # NOTE() the indexing is specific for bloom. This won't work for other models
                # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
                # Bloom is special one
                if choice is None:
                    if isinstance(self._model, BloomForCausalLM):
                        # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                        k = k[:, :, :end_pos]
                        v = v[:, :end_pos, :]
                        kv_trimmed = (k, v)
                        past_key_values_trimmed.append(kv_trimmed)
                    else:
                        # k, v (batch, head, seq, hidden_dim)
                        k = k[:, :, :end_pos, :]
                        v = v[:, :, :end_pos, :]
                        kv_trimmed = (k, v)
                        past_key_values_trimmed.append(kv_trimmed)
                else:
                    if isinstance(self._model, BloomForCausalLM):
                        raise NotImplementedError
                    else:
                        # k, v (batch, head, seq, hidden_dim)
                        if isinstance(choice, int):
                            k = k[choice:choice+1, :, :end_pos, :]
                            v = v[choice:choice+1, :, :end_pos, :]
                        else:
                            k = k[choice, :, :end_pos, :]
                            v = v[choice, :, :end_pos, :]
                        kv_trimmed = (k, v)
                        past_key_values_trimmed.append(kv_trimmed)
        else:
            for values in self._past_key_values:
                trimmed_values = []
                for i, val in enumerate(values):
                    if i >= 2:
                        if choice is None:
                            trimmed_values.append(val)
                        else:
                            if isinstance(choice, int):
                                val = val[choice:choice+1, :, :, :]
                            else:
                                val = val[choice, :, :, :]

                            trimmed_values.append(val)

                    else:
                        if choice is None:
                            val = val[:, :, :end_pos, :]
                            trimmed_values.append(val)
                        else:
                            if isinstance(choice, int):
                                val = val[choice:choice+1, :, :end_pos, :]
                            else:
                                val = val[choice, :, :end_pos, :]
                                #print(val.size())

                            trimmed_values.append(val)
                past_key_values_trimmed.append(tuple(trimmed_values))
        #xxx = input('after rollback')
        
        self._past_key_values = past_key_values_trimmed
        if self._prob_history is not None:
            if choice is None:
                self._prob_history = self._prob_history[:, :end_pos, :]
            else:
                if isinstance(choice, int):
                    self._prob_history = self._prob_history[choice:choice+1, :end_pos, :]
                else:
                    self._prob_history = self._prob_history[choice, :end_pos, :]


    @torch.no_grad()
    def beam_sample_with_kv_cache(
            self,
            prefix,
            gamma,
            num_beams,
            top_k = None,
            top_p = None,
            acc_rate_head = None,
            acc_rate_thres = 0.4,
            ret_seq_scores = False,
            return_intermediate_results = False,
            optimization = False,
            **kwargs
            ):
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """

        # initiation
        generation_config = None
        if generation_config is None and False:
           # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
           # two conditions must be met
           # 1) the generation config must have been created from the model config (`_from_model_config` field);
           # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self._model.generation_config._from_model_config and self._model.generation_config._original_object_hash == hash(
                 self._model.generation_config
               ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self._model.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                        )
                    self._model.generation_config = new_generation_config
        generation_config = self._model.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.num_beams = num_beams
        max_length = gamma + prefix.size(-1)
        generation_config.max_length = max_length
        generation_config.return_dict_in_generate = True
        generation_config.validate()
        self._model._validate_model_kwargs(model_kwargs.copy())
#        model_kwargs['cache_position'] = torch.arange(prefix.shape[1], dtype=torch.int64, device="cuda:0")

        logits_warper = LogitsProcessorList()
        
        if top_k is not None and top_k > 0:
            logits_warper.append(TopKLogitsWarper(top_k))
        if top_p is not None and top_p > 0:
            logits_warper.append(TopPLogitsWarper(top_p))
        
        # 12. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
               batch_size=1,
               num_beams=generation_config.num_beams,
               device=prefix.device,
               length_penalty=generation_config.length_penalty,
               do_early_stopping=generation_config.early_stopping,
               num_beam_hyps_to_keep=generation_config.num_return_sequences,
               max_length=generation_config.max_length)

        # 13. interleave input_ids with `num_beams` additional sequences per batch
        if prefix.size(0) == 1:
            #TODO check this
            input_ids, model_kwargs = self._model._expand_inputs_for_generation(
                                        input_ids=prefix,
                                        expand_size=generation_config.num_beams,
                                        is_encoder_decoder=self._model.config.is_encoder_decoder,
                                        **model_kwargs
                                   )
            past_key_values = []
            if self._past_key_values is not None:
                for values in self._past_key_values:
                    repeat_values = []
                    for val in values:
                        repeat_values.append(val.repeat(num_beams, 1, 1, 1))
                    past_key_values.append(tuple(repeat_values))
                self._past_key_values = past_key_values

        else:
            input_ids, model_kwargs = self._model._expand_inputs_for_generation(
                                        input_ids=prefix,
                                        expand_size=1,
                                        is_encoder_decoder=self._model.config.is_encoder_decoder,
                                        **model_kwargs
                                   )
            """
            if self._past_key_values is not None:
                for values in self._past_key_values:
                    repeat_values = []
                    for val in values:
                        print(val.size())
                    #past_key_values.append(tuple(repeat_values))
                #self._past_key_values = past_key_values
                xxx = input()
            """

            # interleave kv cache


        # call beam_sample with kv cache
        return self.beam_sample(input_ids,
                                beam_scorer,
                                gamma,
                                logits_warper=logits_warper,
                                pad_token_id=generation_config.pad_token_id,
                                eos_token_id=generation_config.eos_token_id,
                                output_scores=generation_config.output_scores,
                                return_dict_in_generate=generation_config.return_dict_in_generate,
                                synced_gpus=False,
                                acc_rate_head = acc_rate_head,
                                acc_rate_thres = acc_rate_thres,
                                ret_seq_scores = ret_seq_scores,
                                return_intermediate_results = return_intermediate_results,
                                optimization = optimization,
                                **model_kwargs,
                                )



    @torch.no_grad()
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        gamma: int,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        acc_rate_head = None,
        acc_rate_thres = 0.4,
        ret_seq_scores = False,
        return_intermediate_results = False,
        optimization = False,
        **model_kwargs,
    ): 
        self.beam_past_key_values = []
        """rewrite beam sample with kv cache"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self._model.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self._model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self._model.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self._model.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self._model.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self._model.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # init attention / hidden states / scores tuples
        #scores = () if (return_dict_in_generate and output_scores) else None
        scores = None
        seq_scores = None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self._model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        new_len = 0
        seq_acc_rate = torch.ones((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        all_intermediate_beams = []
        all_beam_indices = []
        all_next_tokens = []
        all_beam_scores = []
        all_input_indices = []
        all_sample_probs = []
        input_index = torch.arange(num_beams, dtype=torch.long, device=input_ids.device)
        all_input_indices.append(input_index)

#        print('start iteration')
#        xxx = input()

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break


            if self._past_key_values is None: # or True:
                model_inputs = self._model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                model_kwargs['use_cache'] = True

                try:

                    outputs = self._model(
                      **model_inputs,
                      return_dict=True,
                      output_attentions=output_attentions,
                      output_hidden_states=output_hidden_states,
                    )
                except Exception as e:
                    print(model_kwargs)
                    print(input_ids.size())
                    #print(model_inputs)
                    #print(model_inputs.keys())
                    for k,v in model_inputs['encoder_outputs'].items():
                        if isinstance(v, torch.Tensor):
                            print(k)
                            print(v.size())
                    xxx = input('error')
                    print(e)
                    os._exit(0)

                model_kwargs['past_key_values'] = outputs.past_key_values



            else:
                cached_len = self._past_key_values[0][0].shape[2]
                last_input_ids = input_ids[:, cached_len:]
                
                
                if self.beam_rollback_flag == True:
                    model_kwargs["past_key_values"] = None
                    model_kwargs['use_cache'] = True
                    model_inputs = self._model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    """
                    print(input_ids.size())
                    print(last_input_ids.size())
                    print(cached_len)
                    print(model_inputs.keys())
                    print(model_inputs['input_ids'].size())
                    print(model_inputs['past_key_values'])
                    debug_outputs = self._model(
                      **model_inputs,
                      return_dict=True,
                      output_attentions=output_attentions,
                      output_hidden_states=output_hidden_states,
                    )
                    """

                model_kwargs["past_key_values"] = self._past_key_values
                model_kwargs['use_cache'] = True
                model_inputs = self._model.prepare_inputs_for_generation(last_input_ids, 
                    **model_kwargs)

                if self._model.config.is_encoder_decoder == False:
                    model_inputs['input_ids'] = last_input_ids
                else:
                    model_inputs['decoder_input_ids'] = last_input_ids

                #if self.beam_rollback_flag == True:
                #    print(debug_outputs.past_key_values[0][0].size())
                #    print(model_inputs['input_ids'].size())
                #    print(model_inputs['past_key_values'][0][0].size())
              
                assert model_inputs['use_cache'] == True

                outputs = self._model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                """ 
                if self.beam_rollback_flag == True:
                    diff = 0
                    for values, debug_values in zip(outputs.past_key_values, debug_outputs.past_key_values):
                        repeat_values = []
                        for val, d_val in zip(values, debug_values):
                            print(val.size(), d_val.size())
                            diff += (val-d_val).abs().sum(0).sum(0).sum(1)
                            if (val-d_val).abs().sum() > 1e-6:
                                print((val-d_val).abs().nonzero())
                                print((val-d_val).abs().sum())
                                mask = (val-d_val).abs() > 1e-6
                                print(val[mask])
                                xxx = input('val')
                                print(d_val[mask])
                                xxx = input('d_val')
                    print(diff)
                """
                

            self.beam_past_key_values.append(outputs.past_key_values)



            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            tmp = next_token_scores

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )
            next_token_scores = logits_warper(input_ids, next_token_scores)
           
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                #if output_scores:
                #    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self._model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self._model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self._model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            """ for debug """
            """
            probs = probs.view(num_beams, vocab_size)
            probs[:] = 0
            probs[:,7] = 0.
            probs[:,1] = 0.25
            probs[:,12] = 0.75
            probs = probs.view(1,-1)
            probs = probs/probs.sum()
            """
            
            #  add alternative sort criterion based on predicted acceptance rate
            if optimization == True:
            #    next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
                next_tokens = sample(probs, 2*num_beams)
            else: 
#            next_tokens = torch.multinomial(probs, num_samples=num_beams, replacement=True)
                next_tokens = sample(probs, num_beams)

            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
            """
            if (next_token_scores==0).any():
                print('sample 0')
                print(next_token_scores)
                print(probs.nonzero().numel(), 2*num_beams)
                print(next_token_scores_processed.nonzero().numel())
                print(beam_scores)
                print(tmp.nonzero().numel())
                #xxx = input()
            """
            if optimization == True:
                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size
                next_token_scores = torch.clamp(next_token_scores, min=-1e10)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
                # stateless
                # I suspect beam_scorer warp the sampling probability, so I remove it to see the outcome
                beam_outputs = beam_scorer.process(
                  input_ids,
                  next_token_scores,
                  next_tokens,
                  next_indices,
                  pad_token_id=pad_token_id,
                  eos_token_id=eos_token_id,
                  beam_indices=beam_indices,
                )
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

            else:
                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size
                next_token_scores = torch.clamp(next_token_scores, min=-1e10)
                beam_scores = next_token_scores.squeeze()
                beam_next_tokens = next_tokens.squeeze()
                beam_idx = next_indices.squeeze()
            #print(beam_scores.size())
            #print(beam_next_tokens.size())
            #print(beam_idx.size())
            #xxx = input()


            if return_intermediate_results == True: 
                all_intermediate_beams.append(input_ids)
                all_beam_indices.append(beam_idx)
                all_next_tokens.append(beam_next_tokens)
                sample_id = beam_idx * vocab_size + beam_next_tokens
                sample_id = sample_id.view(batch_size, -1)
                score = torch.gather(probs, -1, sample_id).view(-1)
                all_beam_scores.append(score)
                all_sample_probs.append(probs)
                input_index = input_index[beam_idx]
                all_input_indices.append(input_index)
                
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self._model.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._model._reorder_cache(model_kwargs["past_key_values"], beam_idx)
#                if return_intermediate_results == False:
                self._past_key_values = model_kwargs["past_key_values"]

            else:
                print('past key values is none')

            if scores is not None:
                scores = torch.cat([scores, nn.functional.softmax(next_token_scores_processed, dim=-1)[:,None, :]], dim=1)
            else:
                scores = nn.functional.softmax(next_token_scores_processed, dim=-1)[:,None,:]
            scores = scores[beam_idx,:,:]


            cur_seq_scores = probs.view(batch_size*num_beams*vocab_size)
            cur_seq_scores = torch.take(cur_seq_scores, beam_idx * vocab_size + beam_next_tokens)[:,None] # shape [batch_size * num_beams, 1]
            if seq_scores is not None:
                seq_scores = torch.cat([seq_scores, cur_seq_scores], dim=1)
            else:
                seq_scores = cur_seq_scores

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1
            new_len = new_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores) or new_len == gamma:
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
            
            # additional stopping criterion with predicted acc value

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )
        self.beam_rollback_flag = False


        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if ret_seq_scores == True:
                ret_scores = (scores, seq_scores)
            else:
                ret_scores = scores


            if return_intermediate_results == False:
                if self._model.config.is_encoder_decoder:
                    return BeamSampleEncoderDecoderOutput(
                      sequences=sequence_outputs["sequences"][:, :cur_len],
                      sequences_scores=sequence_outputs["sequence_scores"],
                      scores=ret_scores,
                      beam_indices=sequence_outputs["beam_indices"],
                      encoder_attentions=encoder_attentions,
                      encoder_hidden_states=encoder_hidden_states,
                      decoder_attentions=decoder_attentions,
                      cross_attentions=cross_attentions,
                      decoder_hidden_states=decoder_hidden_states,
                    )

                else:
                    return BeamSampleDecoderOnlyOutput(
                      sequences=sequence_outputs["sequences"][:, :cur_len],
                      sequences_scores=sequence_outputs["sequence_scores"],
                      scores=ret_scores,
                      beam_indices=sequence_outputs["beam_indices"],
                      attentions=decoder_attentions,
                      hidden_states=decoder_hidden_states,
                    )
            else:
               # all_intermediate_beams.append(sequence_outputs['sequences'][:, :cur_len])
                all_intermediate_beams.append(input_ids)

                if self._model.config.is_encoder_decoder:
                    return (BeamSampleEncoderDecoderOutput(
                      sequences=sequence_outputs["sequences"][:, :cur_len],
                      sequences_scores=sequence_outputs["sequence_scores"],
                      scores=ret_scores,
                      beam_indices=sequence_outputs["beam_indices"],
                      encoder_attentions=encoder_attentions,
                      encoder_hidden_states=encoder_hidden_states,
                      decoder_attentions=decoder_attentions,
                      cross_attentions=cross_attentions,
                      decoder_hidden_states=decoder_hidden_states,
                  ),
                  all_intermediate_beams,
                  all_beam_indices,
                  all_next_tokens,
                  all_beam_scores,
                  all_sample_probs,
                  all_input_indices)


                else:
                    return (BeamSampleDecoderOnlyOutput(
                      sequences=sequence_outputs["sequences"][:, :cur_len],
                      sequences_scores=sequence_outputs["sequence_scores"],
                      scores=ret_scores,
                      beam_indices=sequence_outputs["beam_indices"],
                      attentions=decoder_attentions,
                      hidden_states=decoder_hidden_states,
                  ),
                  all_intermediate_beams,
                  all_beam_indices,
                  all_next_tokens,
                  all_beam_scores,
                  all_sample_probs,
                  all_input_indices)

        else:
            return sequence_outputs["sequences"]

