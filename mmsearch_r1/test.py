import pickle
import torch

print(torch._C._GLIBCXX_USE_CXX11_ABI)

with open("./prompts/round_1_user_prompt_qwenvl.pkl", "rb") as f:
    prompts = pickle.load(f)
    print(prompts)
