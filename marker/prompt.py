'''
This file assembles prompts.
'''

import numpy as np
import random
from marker import markers, probs

intro_text = '''You are invited to participate in an experiment. 
Your task is to choose between option K and option U for the following lotteries.
'''

continue_text = """
Here is lottery {i}
"""

lot_temp = """
For option K:
- {x}% probability to win $100

For option U:
- It's {marker} to win $100.

Please DO NOT REASON and DIRECTLY output your choice, by ONLY returning one of the following two labels:
"[U]","[K]"

The answer is:"""

def format_lottery(x, marker, index):
    result = ""
    # if index == 0:
    #     result += intro_text
    result += continue_text.format(i=index+1)

    result += lot_temp.format(x=x, marker=marker)
    return result

def generate_ran_seq():
    seq = probs.copy()
    random.shuffle(seq)
    return seq

def gen_prompt_from_seq(marker, seqs):
    result = [intro_text]
    for i, seq in enumerate(seqs):
        result.append(format_lottery(x=seq, marker=marker, index=i))
    return result
                
def gen_batch_prompt(marker, batchsize=8):
    full_seq = [generate_ran_seq() for _ in range(batchsize)]
    full_prompt = [gen_prompt_from_seq(marker, full_seq[i]) for i in range(len(full_seq))]
    return full_seq, full_prompt

if __name__ == "__main__":
    print("batch prompt")
    print(gen_batch_prompt(markers[0]))

    # seqs = generate_ran_seq()
    # for marker in markers:
    #     print(f"===Marker '{marker}'===")
    #     for i, seq in enumerate(seqs):
    #         print(format_lottery(x=seq, marker=marker, index=i))
    #         print("-" * 80)



