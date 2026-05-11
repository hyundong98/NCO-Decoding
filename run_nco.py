import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from collections import deque
import numpy as np
import json
import re
from datasets import load_dataset
from tqdm import tqdm
import argparse
from pathlib import Path
from transformers import LogitsProcessorList
import os
import interegular
import time
from NCO import *
import random


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def whitespace_chars():
    return [
        " ",
        "\u2581",
        "\u0120",
        "\u00A0",
        "\u202F",
        "\u2009",
        "\u200B",
        "\u2060",
        "\t", "\n", "\r",
    ]


def make_char_class(chars):
    out = []
    for c in chars:
        if c in ["\\", "-", "]", "^"]:
            out.append("\\" + c)
        else:
            out.append(c)
    return "[" + "".join(out) + "]"

WHITESPACE_CLASS = make_char_class(whitespace_chars())


def load_model_and_tokenizer(model_pretrained, max_answer_tokens, num_beams, do_sample, temperature, top_p, top_k):
    # Loading model and tokenizer
    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_pretrained, padding_side="left")
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
    model = AutoModelForCausalLM.from_pretrained(model_pretrained, device_map="auto")
    model = torch.compile(model)

    # Setting generation config
    model.generation_config.max_new_tokens = max_answer_tokens
    model.generation_config.num_beams = num_beams
    if tokenizer.pad_token_id:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Setting config for sampling
    model.generation_config.do_sample = do_sample
    if do_sample:
        if temperature != None:
            model.generation_config.temperature = temperature
        if top_p != None:
            model.generation_config.top_p = top_p
        if top_k != None:
            model.generation_config.top_k = top_k

    return model, tokenizer


def load_data(dataset_name):
    # Loading dataset
    print("Loading dataset")
    if dataset_name == "Enron":
        dataset = load_dataset("LLM-PBE/enron-email", split="train")
        regex_patterns = [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
            r"[0-9]{3}-[0-9]{2}-[0-9]{4}",                      # SSN-like
            r"[0-9]{3}-[0-9]{3}-[0-9]{4}",                      # Phone-like
            r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}"              # Credit Card-like
        ]
        combined_regex = re.compile("|".join(regex_patterns))

        def contains_pii(data):
            text = data.get("text", "")
            if text:
                match = re.search(combined_regex, text)
                if match:
                    start_idx = match.start()
                    prompt = text[:start_idx]
                    if len(prompt) >= 20 and len(prompt)<=2000:
                        return True
            return False
        dataset = dataset.filter(contains_pii, num_proc=4)

    elif dataset_name == "RTP":
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        with open("./predefined_refusal_keywords/refusal_toxic_words_en.json", 'r') as f:
            toxic_vocab = json.load(f)
        def contains_toxic_words(data):
            prompt = data.get("prompt", "").get("text", "")
            answer = data.get("continuation", "").get("text", "")
            for i in toxic_vocab:
                if " "+i.lower()+" " in answer.lower():
                    if len(prompt) >= 20 and len(prompt)<=2000:
                        return True
            return False
        dataset = dataset.filter(contains_toxic_words, num_proc=4)

    else:
        print(f"Dataset {dataset_name} is not implemented yet.")
        exit(1)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(dataset), size=min(500, len(dataset)), replace=False)
    dataset = dataset.select(idx.tolist())
    return dataset


def create_dfa_logit_processor(regexes, tokenizer, model_vocab_size, device, batch_size):
    # Constructing DFAs for the forbidden regexes
    print("Constructing DFAs")
    start_time = time.time()
    dfa_list = []
    for regex in regexes:
        dfa_list.append(ForbiddenDFA(regex, tokenizer, model_vocab_size=model_vocab_size, device=device))
    print("Done in {:.2f}s".format(time.time()-start_time))

    # Creating a logit processor
    dfa_processor = DFALogitsProcessor(
        dfa_constraints=dfa_list,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    return dfa_processor


def load_forbidden_strings_from_files(forbidden_strings_files):
    # Loading forbidden strings
    print(f"Loading forbidden strings from: {forbidden_strings_files}")
    forbidden_strings = set()
    for forbidden_strings_file in forbidden_strings_files:
        path = Path(forbidden_strings_file)
        if not path.exists():
            print(f"Warning: File not found: {forbidden_strings_files}")
        else:
            # Loading multiple forbidden strings
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    forbidden_strings.update(set(data))
                elif isinstance(data, dict):
                    print("Not implemented yet.")
                    continue
    return forbidden_strings


def create_trie_logit_processor(forbidden_strings, tokenizer, model_vocab_size, device, batch_size):
    # Constructing Aho-Corasick trie for the forbidden strings
    print("Building Aho-Corasick trie")
    start_time = time.time()
    ac_trie = ForbiddenACTrie(forbidden_strings, tokenizer, model_vocab_size=model_vocab_size, device=device)
    print("Done in {:.2f}s".format(time.time()-start_time))

    # Creating a logit processor
    trie_processor = ACTrieLogitsProcessor(
        ac_trie_constraint=ac_trie,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    return trie_processor


def create_constraints(dataset_name, model, tokenizer, batch_size, prefix_forcing):
    model_vocab_size = model.config.vocab_size
    device = model.device

    if dataset_name == "Enron": # PII suppression
        # Constructing DFAs for the forbidden regexes
        dfa_processor = create_dfa_logit_processor([
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
            r"[0-9]{3}-[0-9]{2}-[0-9]{4}",                      # SSN-like
            r"[0-9]{3}-[0-9]{3}-[0-9]{4}",                      # Phone-like
            r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}"              # Credit Card-like
        ], tokenizer, model_vocab_size, device, batch_size)

        # Creating a list of logits processors
        processors = LogitsProcessorList([dfa_processor])

        # Predefined forced prefix
        forced_prefix = ""

    elif dataset_name == "RTP": # Toxic text suppression
        # Constructing actrie for the forbidden strings
        forbidden_strings = load_forbidden_strings_from_files(["./predefined_refusal_keywords/refusal_toxic_words_en.json"])

        # Creating a list of logits processors
        trie_processor = create_trie_logit_processor(forbidden_strings, tokenizer, model_vocab_size, device, batch_size)

        # Creating a list of logits processors
        processors = LogitsProcessorList([trie_processor])

        # Predefined forced prefix
        forced_prefix = ""

    else:
        print(f"Dataset {dataset_name} is not implemented yet.")
        exit(1)

    return processors, forced_prefix


def get_input_texts(data, tokenizer, dataset_name, forced_prefix):
    # Getting qeury
    if dataset_name == "Enron":
        regex_patterns = [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
            r"[0-9]{3}-[0-9]{2}-[0-9]{4}",                      # SSN-like
            r"[0-9]{3}-[0-9]{3}-[0-9]{4}",                      # Phone-like
            r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}"              # Credit Card-like
        ]
        combined_regex = re.compile("|".join(regex_patterns))

        queries = []
        for text in data["text"]:
            match = re.search(combined_regex, text)
            start_idx = match.start()
            queries.append(text[:start_idx])
    elif dataset_name == "RTP":
        queries = [i["text"] for i in data["prompt"]]
    else:
        print(f"Dataset {dataset_name} is not implemented yet.")
        exit(1)

    if dataset_name in ["Enron", "RTP"]:
        input_texts = queries
    else:
        print(f"Dataset {dataset_name} is not implemented yet.")
        exit(1)

    return input_texts


def main(dataset_name, model_pretrained, batch_size, output_dir, seed, max_answer_tokens, num_beams, do_sample, temperature, top_p, top_k, prefix_forcing):
    set_seed(seed)
    torch.set_float32_matmul_precision('high')

    # Loading model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_pretrained, max_answer_tokens, num_beams, do_sample, temperature, top_p, top_k)

    # Loading dataset
    dataset = load_data(dataset_name)
    dataset_size = len(dataset)

    # Loading constraints
    processors, forced_prefix = create_constraints(dataset_name, model, tokenizer, batch_size, prefix_forcing)

    # Preparing to save outputs
    os.makedirs(output_dir,exist_ok=True)
    output_file_name = "ours_" + dataset_name + "_"+ model_pretrained.split("/")[1].split("-Instruct")[0].replace("-", "_") + "_" + str(batch_size) + "_" + str(seed) + ".jsonl"
    output_path = output_dir / output_file_name

    # Warm-up
    with torch.no_grad():
        dummy_input = tokenizer(
            ["warmup"] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        _ = constrained_generate(
            model,
            tokenizer,
            dummy_input,
            pad_token_id=model.generation_config.pad_token_id,
            num_beams=model.generation_config.num_beams,
            do_sample=model.generation_config.do_sample,
            top_k=model.generation_config.top_k,
            top_p=model.generation_config.top_p,
            temperature=model.generation_config.temperature,
            max_new_tokens=model.generation_config.max_new_tokens,
            logits_processor=processors
        )
    torch.cuda.synchronize()

    # Start generation
    print("Generating text")
    total_gen_time = 0
    total_gen_tokens = 0
    with open(output_path, "w", encoding='utf-8') as f:
        for i in tqdm(range(0, dataset_size, batch_size)):
            # Batching inputs
            batch_data = dataset[i : i + batch_size]
            input_texts = get_input_texts(batch_data, tokenizer, dataset_name, forced_prefix)

            # Applying chat templates and tokenizing
            inputs = tokenizer(
                input_texts,
                return_tensors="pt", 
                padding=True,
                add_special_tokens=False
            ).to(model.device)

            # Resetting the states
            for processor in processors:
                processor.reset_state(len(input_texts)*model.generation_config.num_beams)

            # Generating outputs
            start_time = time.time()

            outputs = constrained_generate(
                model,
                tokenizer,
                inputs,
                pad_token_id=model.generation_config.pad_token_id,
                num_beams=model.generation_config.num_beams,
                do_sample=model.generation_config.do_sample,
                top_k=model.generation_config.top_k,
                top_p=model.generation_config.top_p,
                temperature=model.generation_config.temperature,
                max_new_tokens=model.generation_config.max_new_tokens,
                logits_processor=processors
            )

            end_time = time.time()
            batch_time = end_time - start_time

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            for idx, text in enumerate(decoded_outputs):
                json.dump({
                    "idx":i+idx,
                    "prompt": input_texts[idx],
                    "output":text
                }, f, ensure_ascii=False)
                f.write("\n")

            num_input_tokens = inputs.input_ids.numel()
            num_total_tokens = outputs.numel()
            num_new_tokens = num_total_tokens - num_input_tokens

            total_gen_time += batch_time
            total_gen_tokens += num_new_tokens

    print("{} tokens, {}s, {:.2f} tokens/s".format(total_gen_tokens, total_gen_time, total_gen_tokens/total_gen_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", choices=["Enron", "RTP"], required=True, type=str)
    parser.add_argument("--model-pretrained", default="meta-llama/Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--output-dir", type=Path, default="./result")
    parser.add_argument("--seed", default="42", type=int)
    parser.add_argument("--max-answer-tokens", default=256, type=int)
    parser.add_argument("--num-beams", default=1, type=int)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--top-p", default=None, type=float)
    parser.add_argument("--top-k", default=None, type=float)
    parser.add_argument("--prefix-forcing", action="store_true", default=False)
    args = parser.parse_args()
    
    main(args.dataset_name, args.model_pretrained, args.batch_size, args.output_dir, args.seed, args.max_answer_tokens, args.num_beams, args.do_sample, args.temperature, args.top_p, args.top_k, args.prefix_forcing)
