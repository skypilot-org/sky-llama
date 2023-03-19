# Copied load() and setup_model_parallel() functions from https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/example.py
# Copied sample_top_p() function from https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/generation.py


import fire
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Set, Tuple

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import torch

from llama import ModelArgs
from llama import Tokenizer
from llama import Transformer


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class ChatLLaMA:

    MAX_SEQ_LEN = 2048

    # FIXME
    USER_PREFIX = 'User:'
    CHATBOT_PREFIX = 'Chatbot:'
    INIT_PROMPT = (
        'The following is a conversation between a human and a chatbot. '
        'The chatbot is responsible for answering the user questions, providing information, '
        'and having discussions with users in a conversational manner.\n'
        f'{USER_PREFIX} Hi, what is your name?\n'
        f'{CHATBOT_PREFIX} Hi, my name is LLaMA.\n'
        f'{USER_PREFIX} What is the largest city in the US?\n'
        f'{CHATBOT_PREFIX} The largest city in the US by population is New York City, with over 8 million people.\n'
        f'{USER_PREFIX} What is the highest mountain in the world?\n'
        f'{CHATBOT_PREFIX} The highest mountain in the world is Mount Everest, '
        'which is located in the Himalayas on the border between Nepal and Tibet.\n'
        f'{USER_PREFIX} Explain cloud computing in 1 sentence.\n'
        f'{CHATBOT_PREFIX} Cloud computing is the delivery of computing services, '
        'including servers, storage, databases, networking, software, analytics, and intelligence, over the Internet.\n'
    )

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        model: Transformer,
        tokenizer: Tokenizer,
        temperature: float,
        top_p: float,
        stop_token_ids: Set[int] = set(),
        max_gen_len: int = 256,
    ) -> None:
        self.local_rank = local_rank
        self.world_size = world_size
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.stop_token_ids = stop_token_ids
        self.max_gen_len = max_gen_len

        self.cur_pos = 0
        self.stop_token_ids.add(self.tokenizer.eos_id)

    def forward_step(
        self,
        input_tokens: torch.Tensor,
        input_len: int,
    ) -> torch.Tensor:
        input_tokens = input_tokens[:, :input_len]
        logits = self.model.forward(input_tokens, self.cur_pos)
        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            next_token = sample_top_p(probs, self.top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        self.cur_pos += input_len
        return next_token

    def chat(self) -> None:
        if self.local_rank == 0:
            print('Sampling parameters:')
            print(f'  temperature: {self.temperature}')
            print(f'  top p: {self.top_p}')
            print(f'  maximum response length: {self.max_gen_len}')

        while True:
            is_first_input = self.cur_pos == 0
            if is_first_input:
                print('Initializing the chatbot...')

            if self.local_rank == 0:
                if is_first_input:
                    input_str = self.INIT_PROMPT
                else:
                    user_input = input(self.USER_PREFIX + ' ')
                    if user_input:
                        input_str = f'{self.USER_PREFIX} {user_input}\n'
                        input_str += f'{self.CHATBOT_PREFIX}'
                    else:
                        input_str = ''

                input_tensor = torch.full(
                    (1, self.MAX_SEQ_LEN), self.tokenizer.pad_id).cuda().long()
                if input_str:
                    input_tokens = self.tokenizer.encode(
                        input_str, bos=is_first_input, eos=False)
                    input_tensor[0, :len(input_tokens)] = torch.tensor(input_tokens).long()
            else:
                input_tensor = torch.empty((1, self.MAX_SEQ_LEN)).cuda().long()
            torch.distributed.broadcast(input_tensor, src=0)

            input_len = int((input_tensor != self.tokenizer.pad_id).sum())
            if input_len == 0:
                if self.local_rank == 0:
                    print('Finishing the chat. Wait a few seconds...')
                exit()
            if self.cur_pos + input_len > self.MAX_SEQ_LEN:
                if self.local_rank == 0:
                    print('The conversation length has exceeded the maximum, which is 2048.')
                exit()

            if is_first_input:
                _ = self.forward_step(input_tensor, input_len)
                continue

            output_tokens: List[int] = []
            for _ in range(self.max_gen_len):
                next_token = self.forward_step(input_tensor, input_len)
                input_tensor = next_token
                input_len = 1

                next_token = int(next_token)
                if next_token in self.stop_token_ids:
                    break

                temp_output_tokens = output_tokens + [next_token]
                temp_output = self.tokenizer.decode(temp_output_tokens)
                if temp_output.endswith(self.USER_PREFIX):
                    self.cur_pos -= 1
                    output_tokens.pop()
                    break
                elif temp_output.endswith(self.CHATBOT_PREFIX):
                    self.cur_pos -= 3
                    for _ in range(3):
                        output_tokens.pop()
                    break
                else:
                    output_tokens.append(next_token)

            if self.local_rank == 0:
                output = self.tokenizer.decode(output_tokens)
                print(f'{self.CHATBOT_PREFIX} {output.rstrip()}')


def setup_model_parallel(
    seed: int,
) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(seed)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
) -> Tuple[Transformer, Tokenizer]:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading the model weights. This may take several minutes.")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


@torch.inference_mode()
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.99,
    seed: int = 42,
) -> None:
    local_rank, world_size = setup_model_parallel(seed)
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    model, tokenizer = load(ckpt_dir, tokenizer_path, local_rank, world_size)
    chat_llama = ChatLLaMA(
        local_rank=local_rank,
        world_size=world_size,
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=top_p,
    )
    chat_llama.chat()


if __name__ == "__main__":
    fire.Fire(main)
