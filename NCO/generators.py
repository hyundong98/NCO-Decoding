from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList
from transformers.cache_utils import Cache

@torch.no_grad()
def constrained_generate(
    model,
    tokenizer,
    inputs,
    pad_token_id=None,
    num_beams: int = 1,
    do_sample: bool = False,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    max_new_tokens: int = 20,
    logits_processor=None
) -> torch.Tensor:
    # Setting default values
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    if pad_token_id is None:
        if tokenizer.pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = tokenizer.pad_token_id
            
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if num_beams > 1:
        # Decoding w/ beam search
        return _custom_beam_search_loop(
            model, tokenizer, input_ids, attention_mask, pad_token_id, 
            num_beams, do_sample, top_k, top_p, temperature, max_new_tokens, logits_processor
        )
    else:
        # Decoding w/o beam search
        return _custom_no_beam_loop(
            model, tokenizer, input_ids, attention_mask, pad_token_id, 
            num_beams, do_sample, top_k, top_p, temperature, max_new_tokens, logits_processor
        )

def _custom_no_beam_loop(
    model, tokenizer, input_ids, attention_mask, pad_token_id, 
    num_beams, do_sample, top_k, top_p, temperature, max_new_tokens, logits_processor
):
    batch_size = input_ids.shape[0]
    past_key_values = None # Caching

    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    pad_token_tensor = torch.tensor(pad_token_id, device=model.device)
    eos_token_id = tokenizer.eos_token_id
    finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=model.device) # Check bit for finished generation
    for _ in range(max_new_tokens):
        if past_key_values:
            model_inputs = input_ids[:, -1:]
            position_ids = (attention_mask.sum(axis=1)-1).unsqueeze(-1)
        else:
            model_inputs = input_ids

        outputs = model(
            model_inputs, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values, 
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Calculating original log probs
        original_log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Processing logits
        processed_logits = logits_processor(input_ids, next_token_logits)
        
        # Temperature based sampling
        if do_sample and temperature != 1.0:
            processed_logits = processed_logits / temperature

        # Top-k and Top-p sampling
        if do_sample:
            processed_logits = _apply_top_k_top_p(processed_logits, top_k, top_p)
        
        # Calculating processed log probs
        probs = F.softmax(processed_logits, dim=-1)
        
        # Choosing next token
        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)

        score = torch.gather(original_log_probs, 1, next_token)
        
        # If generation for a batch is ended, add a pad token.
        next_token = torch.where(
            finished_mask.unsqueeze(1),
            pad_token_tensor,
            next_token
        )
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Updating the attention mask
        new_mask = (~finished_mask).type(attention_mask.dtype).unsqueeze(1).to(model.device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
        
        # Updating the finished mask
        is_eos = (next_token.squeeze() == eos_token_id)
        finished_mask = finished_mask | is_eos
        
        # Stopping generation if all batches are finished
        if finished_mask.all():
            break
            
    return input_ids

def _custom_beam_search_loop(
    model, tokenizer, input_ids, attention_mask, pad_token_id, 
    num_beams, do_sample, top_k, top_p, temperature, max_new_tokens, logits_processor
):
    vocab_size = model.config.vocab_size
    batch_size = input_ids.shape[0]    

    # Expanding vectors according to the beam size
    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

    # Calculating position indices
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # Preparing vectors for beam scores
    beam_scores = torch.zeros((batch_size, num_beams), device=model.device)
    beam_scores[:, 1:] = -float("inf")
    beam_scores = beam_scores.view(-1)
    past_key_values = None # Caching
    beam_finished = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=model.device) # Check bit for finished generation

    for _ in range(max_new_tokens):
        if past_key_values:
            model_inputs = input_ids[:, -1:]
            position_ids = (attention_mask.sum(axis=1)-1).unsqueeze(-1)
        else:
            model_inputs = input_ids

        outputs = model(
            model_inputs, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_values=past_key_values, 
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Calculating original log probs
        original_log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Processing logits
        processed_logits = logits_processor(input_ids, next_token_logits)
        if do_sample and temperature != 1.0:
            processed_logits = processed_logits / temperature

        # Temperature based sampling
        if do_sample:
            # Top-k and Top-p sampling
            processed_logits = _apply_top_k_top_p(processed_logits, top_k, top_p)

        # Calculating processed log probs
        candidate_log_probs = F.log_softmax(processed_logits, dim=-1)

        # If generation for a batch is ended, add a pad token.
        if beam_finished.any():
            candidate_log_probs[beam_finished] = -float("inf")
            candidate_log_probs[beam_finished, pad_token_id] = 0.0

        # Calculating candiate scores
        candidate_scores = beam_scores.unsqueeze(-1) + candidate_log_probs
        candidate_scores = candidate_scores.view(batch_size, -1)
        
        # Selecting top beams
        topk_scores, topk_indices = torch.topk(candidate_scores, num_beams, dim=1)
        
        # Calculating indices
        local_beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size
        batch_offset = torch.arange(batch_size, device=model.device).unsqueeze(1) * num_beams
        global_beam_indices = (batch_offset + local_beam_indices).view(-1)
        token_indices = token_indices.view(-1)

        # Updatinng beam scores with original log probs
        selected_original_scores = original_log_probs[global_beam_indices, token_indices]
        prior_finished = beam_finished[global_beam_indices]
        selected_original_scores[prior_finished] = 0.0
        beam_scores = beam_scores[global_beam_indices] + selected_original_scores

        # Reordering the states in the logit processor
        for processor in logits_processor:
            if hasattr(processor, "reorder_state"):
                processor.reorder_state(global_beam_indices)

        # Reordering cache
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(global_beam_indices)
        else:
            reorder_func = None
            if hasattr(model, "_reorder_cache"):
                reorder_func = model._reorder_cache
            elif hasattr(model, "_orig_mod") and hasattr(model._orig_mod, "_reorder_cache"):
                reorder_func = model._orig_mod._reorder_cache
            if reorder_func is not None:
                past_key_values = reorder_func(past_key_values, global_beam_indices)
            else:
                try:
                    reordered_past = []
                    for layer_past in past_key_values:
                        reordered_layer = tuple(
                            past.index_select(0, global_beam_indices) for past in layer_past
                        )
                        reordered_past.append(reordered_layer)
                    past_key_values = tuple(reordered_past)
                except Exception as e:
                    print(f"Error occurred while reordering cache: {e}")
                    pass

        # Updating the generated text
        input_ids = torch.cat([input_ids[global_beam_indices], token_indices.unsqueeze(-1)], dim=-1)

        # Updating attention mask
        attention_mask = attention_mask[global_beam_indices]
        new_mask = torch.ones((batch_size * num_beams, 1), dtype=attention_mask.dtype, device=model.device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        # Updating the finished mask
        current_is_eos = (token_indices == tokenizer.eos_token_id)
        beam_finished = prior_finished | current_is_eos

        # Stopping generation if each top-1 beam of each batch is finished
        top_1_tokens = token_indices.view(batch_size, num_beams)[:, 0]
        is_eos = (top_1_tokens == tokenizer.eos_token_id)
        is_pad = (top_1_tokens == pad_token_id)
        if (is_eos | is_pad).all():
            break
    
    # Return the best beams
    final_output = input_ids.view(batch_size, num_beams, -1)
    return final_output[:, 0, :]

def _apply_top_k_top_p(logits, top_k, top_p):
    # Top-k sampling
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')

    # Top-p sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        
    return logits