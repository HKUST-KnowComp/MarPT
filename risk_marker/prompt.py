'''
This file assembles prompts. It also replace probabilities with epistemic markers.
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

def substitute(text, probs, expressions, option=None, only_one=False):
    if option is None:
        for prob, expression in zip(probs, expressions):
            pattern = re.escape(f"{prob}% probability")
            if only_one:
                text = re.sub(pattern, expression, text, count=1)  # 仅替换第一个匹配项
            else:
                text = re.sub(pattern, expression, text)  # 替换所有匹配项
    else:
        lines = text.split('\n')
        for prob, expression in zip(probs, expressions):
            # print(prob, expression)
            # print('\n'.join(lines))
            for i, line in enumerate(lines):
                if f"For option {option}:" in line:
                    for j in range(i + 1, i + 3):
                        if f"{prob}% probability" in lines[j]:
                            lines[j] = lines[j].replace(
                                f"{prob}% probability", 
                                "It's " + expression
                                # 1 if only_one else -1  # 替换 1 次（仅第一个）或全部
                            )
                            if only_one:
                                break  # 替换一次后跳出循环
        text = '\n'.join(lines)
    return text

def safe_sub(text, model_name, mix=False):
    if "Qwen" in model_name and "7B" in model_name:
        probs = [30, 70, 10, 90]
        expressions = ["uncertain", "almost certain", "somewhat likely", "highly likely"]
        if mix:
            probs.append(50)
            expressions.append("very likely")
        text = substitute(text, probs, expressions, option='K')
        return text, [30, 70, 12, 88]
    elif "Qwen" in model_name and "14B" in model_name:
        probs = [30, 70, 10, 90]
        expressions = ["somewhat unlikely", "highly likely", "very unlikely", "almost certain"]
        if mix:
            probs.append(50)
            expressions.append("likely")
        text = substitute(text, probs, expressions, option='K')
        return text, [32, 68, 10, 90]
    elif "Qwen" in model_name and "32B" in model_name:
        probs = [30, 70, 10, 90]
        expressions = ["somewhat unlikely", "probable", "somewhat likely", "almost certain"]
        if mix:
            probs.append(50)
            expressions.append("probable")
        text = substitute(text, probs, expressions, option='K')
        return text, [29, 71, 18, 82]
    elif "Llama" in model_name and "8B" in model_name:
        probs = [30, 70, 10, 90]
        expressions = ["likely", "almost certain", "very unlikely", "almost certain"]
        if mix:
            probs.append(50)
            expressions.append("highly likely")
        text = substitute(text, probs, expressions, option='K')
        return text, [32, 68, 26, 74]
    elif "Mistral" in model_name and "7B" in model_name:
        probs = [30, 70, 10, 90]
        expressions = ["very unlikely", "highly likely", "highly unlikely", "almost certain"]
        if mix:
            probs.append(50)
            expressions.append("somewhat likely")
        text = substitute(text, probs, expressions, option='K')
        return text, [30, 70, 22, 78]

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

def to_text(batch_prompt, model_name):
    for j in range(len(batch_prompt)):
        for i in range(len(batch_prompt[j])):
            batch_prompt[j][i], new_probs = safe_sub(batch_prompt[j][i], model_name=model_name)
    return batch_prompt

if __name__ == "__main__":
    # Example usage
    print("batch prompt")
    seq, prompt = gen_batch_prompt(batchsize=1)
    prompt = to_text(prompt, model_name="Qwen/Qwen2.5-32B-Instruct")
    print(prompt[:4])
