'''
This file assembles prompts.
'''

from values_probs import values, probs
import re
import random

intro_text = '''You are invited to participate in an experiment.
Your task is to choose between option K and option U for each of the following lotteries. 
'''

continue_text = '''
Here is lottery {i}:
'''

lot_temp_win = """
For option U:  
- {u_prob_more}% probability to win ${u_more}.
- {u_prob_less}% probability to win ${u_less}.

For option K:  
- {k_prob_more}% probability to win ${k_more}.
- {k_prob_less}% probability to win ${k_less}.

Please DO NOT REASON and DIRECTLY output your choice, by ONLY returning one of the following two labels:
"[U]","[K]"

The answer is: """

lot_temp_mix = """
For option U:  
- 50% probability to win ${u_win}.
- 50% probability to lose ${u_lose}.

For option K:  
- 50% probability to win ${k_win}.
- 50% probability to lose ${k_lose}.

Please DO NOT REASON and DIRECTLY output your choice, by ONLY returning one of the following two labels:
"[U]","[K]"

The answer is: """

# lot_intex = range(0, 36), randomized_seq = random sequence of [series, index]
def format_lottery(lot_index, randomized_seq):
    result = ""

    # result += continue_text.format(i=lot_index+1)

    lottery = randomized_seq[lot_index]

    value = values[lottery[0]][lottery[1]] # e.g. [40, 30, 60, 5]
    prob = probs[lottery[0]] # e.g. [30, 70, 10, 90]

    if lottery[0] != 2:
        result += lot_temp_win.format(
            k_more=value[0],
            k_less=value[1],
            u_more=value[2],
            u_less=value[3],
            k_prob_more=prob[0],
            k_prob_less=prob[1],
            u_prob_more=prob[2],
            u_prob_less=prob[3]
        )
    elif lottery[0] == 2:
        result += lot_temp_mix.format(
            k_win=value[0],
            k_lose=-value[1],
            u_win=value[2],
            u_lose=-value[3]
        )
    return result

# generate random [series, index]
def generate_ran_seq():
    seq = []
    for series in [0, 1, 2]:
        length = 14 if series != 2 else 7
        for index in range(length):
            seq.append([series, index])
    random.shuffle(seq)
    
    return seq

def gen_prompt_from_seq(seqs):
    result = [intro_text]
    for i, seq in enumerate(seqs):
        result.append(format_lottery(lot_index=i, randomized_seq=seqs))
    return result

def gen_batch_prompt(batchsize=8):
    full_seq = [generate_ran_seq() for _ in range(batchsize)]
    full_prompt = [gen_prompt_from_seq(full_seq[i]) for i in range(len(full_seq))]
    return full_seq, full_prompt

if __name__ == "__main__":
    # Example usage
    print("batch prompt")
    seq, prompt = gen_batch_prompt()
    print(prompt)
