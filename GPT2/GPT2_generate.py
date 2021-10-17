import torch
import numpy as np
import json
import argparse
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import warnings
warnings.filterwarnings('ignore')

### 예시 사용법 ###
# 터미널에서      python3 GPT_generate.py --input_sent "성실"

### ### ### ### ###


parser = argparse.ArgumentParser()
parser.add_argument('--input_sent', type=str, default="성실",
                    help="글의 시작 문장입니다.")
parser.add_argument('--temperature', type=float, default=1.0,
                    help="temperature 를 통해서 글의 창의성을 조절합니다.")
parser.add_argument('--top_p', type=float, default=1.0,
                    help="top_p 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--top_k', type=int, default=50,
                    help="top_k 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--max_length', type=int, default=50,
                    help="결과물의 길이를 조정합니다.")
parser.add_argument('--num_beams', type= int, default=3,
                    help="Beam search 파라미터를 조정합니다.")
parser.add_argument('--num_return_sequences', type=int, default=3,
                    help="몇 개의 문장을 생성할지 선택합니다..")
parser.add_argument('--model_name', type=str, default="model128_46450",
                    help="불러올 모델의 이름을 입력합니다: (1) model128_46450,\
                    (2) model128_32000, (3) base_model")

args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "./models/model128_32000",
    bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

model = GPT2LMHeadModel.from_pretrained("./models/model128_32000")

def main(input_sent="성실", temperature=1.0, top_p=1.0, top_k=50, max_length=50, num_beams=3, num_return_sequences=2):
    beam_outputs = model.generate(torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0).to(device),
                                  do_sample=True,
                                  temperature=temperature,
                                  top_p=top_p,
                                  top_k=top_k,
                                  max_length=max_length,
                                  num_beams=num_beams,
                                  num_return_sequences=num_return_sequences,
                                  use_cache=True
                                  )
    out = dict()
    for i, beam_output in enumerate(beam_outputs):
        out[i] = tokenizer.decode(beam_output, skip_special_tokens=True)
    # 위에서 나오는 out은 대부분 max length만큼 문장이 생성되기 때문에 대부분 끝 문장이 중간에 잘려있다.

    text = [i for i in out.values()]

    # out은 raw output dictionary,   text는 sentence split시키고 
    return formatted(text)

def formatted(text):
    print()
    print('-'*70)
    for i, txt in enumerate(text):
        print(f'{i+1}: {txt}')
        print()

if __name__ == "__main__":
    # execute only if run as a script
    main(input_sent=args.input_sent, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_length=args.max_length, num_beams=args.num_beams, num_return_sequences=args.num_return_sequences)


