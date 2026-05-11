import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from collections import deque
import numpy as np
import json
import re
import argparse
from pathlib import Path
from transformers import LogitsProcessorList
import os
import interegular
import warnings

NEG_INF_CONST = -1000000000
NEG_INF_THRESH = -1000000

def find_base_vocabs(merges, vocab):
    not_merged = set()
    merged = set()
    for idx, (prefix, suffix) in enumerate(merges):
        if not prefix in merged and not prefix in not_merged:
            not_merged.add(prefix)
        if not suffix in merged and not suffix in not_merged:
            not_merged.add(suffix)
        merged.add(prefix+suffix)
    not_merged -= merged
    base_vocabs = sorted(list(set(vocab)-set(merged)))
    for token in not_merged:
        assert token in base_vocabs
    assert len(base_vocabs)+len(merged) == len(vocab), "the number of tokens are not matched"
    return base_vocabs

def find_space_char(tokenizer):
    skip_candidates = {" ", "\u2581", "\u0120", "\u00A0"}
    space_char = " " 
    try:
        test_tokens = [tokenizer.encode(w, add_special_tokens=False) for w in [" the", " and"]]
        for ids in test_tokens:
            if len(ids) > 0:
                first_token = tokenizer.convert_ids_to_tokens(ids[0])
                if isinstance(first_token, bytes):
                    try:
                        first_token = first_token.decode('utf-8')
                    except:
                        continue
                if first_token and first_token[0] in skip_candidates:
                    space_char = first_token[0]
                    break
    except:
        pass
    return space_char

class ForbiddenACTrie:
    def __init__(self, forbidden_strings, tokenizer, soft_logit_to_add=-float("inf"), model_vocab_size=None, device="cuda"):
        self.device = device
        self.tokenizer = tokenizer
        self.model_vocab_size = model_vocab_size or getattr(tokenizer, "vocab_size", None)
        self.soft_logit_to_add = soft_logit_to_add
        assert soft_logit_to_add!=float("inf"), "The soft logit value to add cannot be inf."
        self.space_char = find_space_char(tokenizer)
        self._build_ac_trie(forbidden_strings)
        self._build_token_transitions()

    def _build_ac_trie(self, forbidden_strings):
        self.transitions = {0: {}}
        self.fail = {0: 0}
        self.forbidden_nodes = set()
        self.num_states = 1

        # Build Aho-Corasick trie
        for word in forbidden_strings:
            word = word.replace(" ", self.space_char)
            state = 0
            for ch in word:
                if ch not in self.transitions[state]:
                    self.transitions[state][ch] = self.num_states
                    self.transitions[self.num_states] = {}
                    self.num_states += 1
                state = self.transitions[state][ch]
            self.forbidden_nodes.add(state)

        # Build failure links
        q = deque()
        for ch, nxt in self.transitions[0].items():
            self.fail[nxt] = 0
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.transitions[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.transitions[f]:
                    f = self.fail[f]
                self.fail[s] = self.transitions[f].get(ch, 0)
                if self.fail[s] in self.forbidden_nodes:
                    self.forbidden_nodes.add(s)

        # Recording the forbidden nodes
        self.is_forbidden = np.zeros(self.num_states, dtype=bool)
        for s in self.forbidden_nodes:
            self.is_forbidden[s] = True

    def compute_vocab_transitions(self, vocab_map, vocab, transitions, forbidden_mask):
        for text in vocab:
            tok_id = vocab_map[text]
            for ini_state in range(self.num_states):
                state = ini_state
                for ch in text:
                    if self.is_forbidden[state]:
                        forbidden_mask[tok_id, ini_state] = True
                    while state and ch not in self.transitions[state]:
                        state = self.fail[state]
                    state = self.transitions[state].get(ch, 0)
                if self.is_forbidden[state]:
                    forbidden_mask[tok_id, ini_state] = True
                transitions[tok_id, ini_state] = state
    
    def _build_token_transitions(self):
        # Loading vocabulary
        vocab = self.tokenizer.get_vocab()
        vocab_size = self.model_vocab_size or len(self.tokenizer)
        transitions = np.zeros((vocab_size, self.num_states), dtype=np.int32)
        forbidden_mask = np.zeros((vocab_size, self.num_states), dtype=bool)

        # Check if the tokenizer is BPE tokenizer
        self.is_BPE = True
        try:
            merges = json.loads(self.tokenizer.backend_tokenizer.to_str())["model"]["merges"]
        except:
            self.is_BPE = False
        merges = sorted(merges, key=lambda x: len(x[0]+x[1]))

        # Precomputing vocab-level transitions
        if self.is_BPE:
            print("The model uses BPE.")
            base_vocabs = find_base_vocabs(merges, vocab)

            # Precomputing for base
            self.compute_vocab_transitions(vocab, base_vocabs, transitions, forbidden_mask)

            # Precomputing merged vocabs
            for prefix, suffix in merges:
                text = prefix+suffix
                tok_id = vocab[text]
                prefix_tok_id = vocab[prefix]
                suffix_tok_id = vocab[suffix]
                for state in range(self.num_states):
                    temp_state = transitions[prefix_tok_id, state]
                    transitions[tok_id, state] = transitions[suffix_tok_id, temp_state]
                    forbidden_mask[tok_id, state] = forbidden_mask[prefix_tok_id, state] or forbidden_mask[suffix_tok_id, temp_state]
        else:
            print("The model does not use BPE.")
            # Precomputing vocabs
            self.compute_vocab_transitions(vocab, vocab.keys(), transitions, forbidden_mask)

        forbidden_mask[self.tokenizer.eos_token_id,:] = False
        float_forbidden_mask = forbidden_mask.astype(np.float32)
        float_forbidden_mask[forbidden_mask] *= self.soft_logit_to_add
        
        self.transitions = torch.from_numpy(transitions).to(self.device)
        self.forbidden_mask = torch.from_numpy(float_forbidden_mask).T.to(self.device)
        self.num_states = self.transitions.shape[1]
        self.vocab_size = vocab_size

class ForbiddenDFA:
    def __init__(self, forbidden_regular_expression, tokenizer, soft_logit_to_add=NEG_INF_CONST, model_vocab_size=None, device="cuda"):
        self.device = device
        self.tokenizer = tokenizer
        self.model_vocab_size = model_vocab_size or getattr(tokenizer, "vocab_size", None)
        if soft_logit_to_add < NEG_INF_CONST:
            self.soft_logit_to_add = NEG_INF_CONST
        else:
            self.soft_logit_to_add = soft_logit_to_add
        assert soft_logit_to_add!=float("inf"), "The soft logit value to add cannot be inf."
        self.space_char = find_space_char(tokenizer)
        self.dfa = interegular.parse_pattern(forbidden_regular_expression.replace(" ", self.space_char)).to_fsm().reduce()
        assert all(state >= 0 for state in self.dfa.states)
        states = sorted(self.dfa.states)
        assert states == list(range(len(states)))
        self.num_states = len(self.dfa.states)
        self._build_token_transitions()

    def compute_vocab_transitions(self, vocab_map, vocab, transitions, suffix_reachable_from_initial, forbidden_mask):
        for text in vocab:
            tok_id = vocab_map[text]
            # Precomputation over state
            for ini_state in self.dfa.states:
                state = ini_state
                for ch in text:
                    if state in self.dfa.finals:
                        forbidden_mask[tok_id, ini_state] = True
                    ch_id = self.dfa.alphabet.get(ch, self.dfa.alphabet["anything_else"])
                    state = self.dfa.map[state].get(ch_id, -1)
                    if state == -1:
                        break
                if state in self.dfa.finals:
                    forbidden_mask[tok_id, ini_state] = True
                transitions[tok_id, ini_state] = state
            
            # Precomputation over suffixes
            ini_state = self.dfa.initial
            if transitions[tok_id, ini_state] != -1:
                suffix_reachable_from_initial[tok_id, transitions[tok_id, ini_state]] = True
            for i in range(1, len(text)):
                state = ini_state
                for j in range(i, len(text)):
                    ch = text[j]
                    if state in self.dfa.finals:
                        forbidden_mask[tok_id, :] = True
                    ch_id = self.dfa.alphabet.get(ch, self.dfa.alphabet["anything_else"])
                    state = self.dfa.map[state].get(ch_id, -1)
                    if state == -1:
                        break
                if state in self.dfa.finals:
                    forbidden_mask[tok_id, :] = True
                if state != -1:
                    suffix_reachable_from_initial[tok_id, state] = True

    def _build_token_transitions(self):
        # Loading vocabulary
        vocab = self.tokenizer.get_vocab()
        vocab_size = self.model_vocab_size or len(self.tokenizer)
        transitions = np.zeros((vocab_size, self.num_states), dtype=np.int32)
        forbidden_mask = np.zeros((vocab_size, self.num_states), dtype=bool)
        vocab_max_len = len(max(vocab.keys(), key = lambda x: len(x)))
        suffix_reachable_from_initial = np.zeros((vocab_size, self.num_states), dtype=bool)

        # Check if the tokenizer is BPE tokenizer
        self.is_BPE = True
        try:
            merges = json.loads(self.tokenizer.backend_tokenizer.to_str())["model"]["merges"]
        except:
            self.is_BPE = False
        merges = sorted(merges, key=lambda x: len(x[0]+x[1]))

        # Precomputing vocab-level transitions
        if self.is_BPE:
            print("The model uses BPE.")
            base_vocabs = find_base_vocabs(merges, vocab)
            
            # Precomputing for base
            self.compute_vocab_transitions(vocab, base_vocabs, transitions, suffix_reachable_from_initial, forbidden_mask)

            # Precomputing merged vocabs
            for prefix, suffix in merges:
                text = prefix+suffix
                tok_id = vocab[text]
                prefix_tok_id = vocab[prefix]
                suffix_tok_id = vocab[suffix]
                # Precomputation over state
                for state in range(self.num_states):
                    temp_state = transitions[prefix_tok_id, state]
                    if temp_state == -1:
                        transitions[tok_id, state] = -1
                        forbidden_mask[tok_id, state] = forbidden_mask[prefix_tok_id, state]
                    else:
                        transitions[tok_id, state] = transitions[suffix_tok_id, temp_state]
                        forbidden_mask[tok_id, state] = forbidden_mask[prefix_tok_id, state] or forbidden_mask[suffix_tok_id, temp_state]
                
                # Precomputation over suffixes
                prefix_states = suffix_reachable_from_initial[prefix_tok_id]
                suffix_states = np.nonzero(suffix_reachable_from_initial[suffix_tok_id])[0]
                total_states = np.unique(np.concatenate((transitions[suffix_tok_id, prefix_states], suffix_states)))
                np.put_along_axis(suffix_reachable_from_initial[tok_id], total_states[total_states!=-1], True, axis=0)
                if any(forbidden_mask[suffix_tok_id, prefix_states]) or forbidden_mask[suffix_tok_id, self.dfa.initial]:
                    forbidden_mask[tok_id,:] = True
        else:
            print("The model does not use BPE.")
            # Precomputing vocabs
            self.compute_vocab_transitions(vocab, vocab.keys(), transitions, suffix_reachable_from_initial, forbidden_mask)

        forbidden_mask[self.tokenizer.eos_token_id,:] = False
        float_forbidden_mask = forbidden_mask.astype(np.float32)
        float_forbidden_mask[forbidden_mask] *= self.soft_logit_to_add
        
        self.transitions = transitions
        self.suffix_reachable_from_initial = suffix_reachable_from_initial
        self.forbidden_mask = float_forbidden_mask.T
        self.num_states = self.transitions.shape[1]
        self.vocab_size = vocab_size

class ACTrieLogitsProcessor(LogitsProcessor):
    def __init__(self, 
                 ac_trie_constraint,  
                 tokenizer,
                 batch_size=1):

        self.batch_size = batch_size
        self.device = ac_trie_constraint.device
        self.ac_trie_constraint = ac_trie_constraint
        self.constraints_state = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.tokenizer = tokenizer
        self.prompt_length = -1

    @torch.no_grad()
    def __call__(self, input_ids, scores):
        batch_size, ctx_length = input_ids.shape
        vocab_size = scores.shape[1]
        if self.prompt_length == -1:
            if ctx_length > 0:
                self.prompt_length = ctx_length
            else:
                return scores
        generated_length = ctx_length - self.prompt_length
        prev_len = ctx_length - 1

        # Update the state with the last token
        if generated_length > 0:
            last_token = input_ids[:, -1]
            self.constraints_state = self.ac_trie_constraint.transitions[last_token, self.constraints_state]
        # Masking based on the states
        scores += self.ac_trie_constraint.forbidden_mask[self.constraints_state, :vocab_size]
        return scores

    def reorder_state(self, beam_indices):
        # Reordering for beam search
        self.constraints_state = self.constraints_state[beam_indices]

    def reset_state(self, batch_size=None):
        # Resetting states
        if batch_size == None:
            batch_size = self.batch_size
        self.prompt_length = -1
        self.constraints_state = torch.zeros(batch_size, dtype=torch.long, device=self.device)

class DFALogitsProcessor(LogitsProcessor):
    def __init__(self,
                 dfa_constraints,
                 tokenizer,
                 batch_size=1):

        self.batch_size = batch_size
        self.device = dfa_constraints[0].device
        
        self.suffix_reachable_from_initial = torch.from_numpy(np.concatenate([i.suffix_reachable_from_initial for i in dfa_constraints], axis=1)).to(self.device)
        self.forbidden_mask = torch.from_numpy(np.concatenate([i.forbidden_mask for i in dfa_constraints], axis=0)).to(self.device)
        
        self.initials = torch.zeros(sum([i.num_states for i in dfa_constraints]), dtype=bool, device=self.device)
        transitions_np = np.concatenate([i.transitions for i in dfa_constraints], axis=1)
        prev_num_states = 0
        for constraint in dfa_constraints:
            self.initials[prev_num_states + constraint.dfa.initial] = True
            next_num_states = prev_num_states + constraint.num_states
            block = transitions_np[:, prev_num_states:next_num_states]
            valid = block != -1
            block[valid] += prev_num_states
            prev_num_states = next_num_states
        self.total_num_states = prev_num_states
        self.transitions = torch.from_numpy(transitions_np).to(self.device)
        
        self.tokenizer = tokenizer
        self.prompt_length = -1
        
        self.constraints_states = torch.zeros((batch_size, self.total_num_states), dtype=bool, device=self.device)
        
    @torch.no_grad()
    def __call__(self, input_ids, scores):
        batch_size, ctx_length = input_ids.shape
        vocab_size = scores.shape[1]
        if self.prompt_length == -1:
            if ctx_length > 0:
                self.prompt_length = ctx_length
            else:
                return scores
        generated_length = ctx_length - self.prompt_length
        prev_len = ctx_length - 1

        # Update the state with the last token
        if generated_length > 0:
            last_tokens = input_ids[:, -1]
            batch_idx, state_idx = torch.where(self.constraints_states)
            self.constraints_states[:] = self.suffix_reachable_from_initial[last_tokens]
            if batch_idx.numel() > 0:
                batched_last_tokens = last_tokens[batch_idx]
                next_state_idx = self.transitions[batched_last_tokens, state_idx]
                valid_transition = next_state_idx != -1
                if valid_transition.any():
                    valid_batch = batch_idx[valid_transition]
                    valid_next_state = next_state_idx[valid_transition]
                    self.constraints_states[valid_batch, valid_next_state] = True
            self.constraints_states[:, self.initials] = True

        # Masking based on several dfa constraints
        scores += torch.matmul(self.constraints_states.float(), self.forbidden_mask)
        scores[scores<NEG_INF_THRESH] = -float("inf")
        return scores

    def reorder_state(self, beam_indices):
        # Reordering for beam search
        self.constraints_states = self.constraints_states[beam_indices]

    def reset_state(self, batch_size=None):
        # Resetting states
        if batch_size == None:
            batch_size = self.batch_size
        self.prompt_length = -1
        self.constraints_states = torch.zeros((batch_size, self.total_num_states), dtype=bool, device=self.device)
        self.constraints_states[:, self.initials] = True