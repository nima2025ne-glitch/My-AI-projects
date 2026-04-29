import os
os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir/"
from transformers import pipeline
classifier = pipeline(
    task="text-generation",
    model="google/gemma-3n-E2B-it",
    device=0   
)
classifier = pipeline(
    task="text-generation",
    model="google/gemma-3-270m-it",
    device=0   
)
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModel.from_pretrained("google/gemma-3-270m-it")

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_text = input("How can i assist you today ?")

inputs = tokenizer(input_text, return_tensors="pt").to(device)

pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

generation_kwargs = dict(
    input_ids=inputs["input_ids"],
    max_new_tokens=900,
    temperature=0.8,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=pad_token_id,
    do_sample=True,
    streamer=streamer  
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("مدل در حال تولید پاسخ...")

# چاپ توکن‌ها به محض آماده شدن
for new_text in streamer:
    print(new_text, end="", flush=True)

print("\n--- پایان تولید ---")

