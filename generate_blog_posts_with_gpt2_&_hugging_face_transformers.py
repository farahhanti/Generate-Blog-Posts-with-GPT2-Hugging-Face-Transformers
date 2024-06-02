\# ***1/ Install and Import Dependencies***

!pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer

"""# ***2/ Load Model***"""

tokenizer.decode(tokenizer.eos_token_id)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained("gpt2-large",pad_token_id=tokenizer.eos_token_id)

"""# ***3/Tokenize Sentences***"""

sentence = " Introducing LLaMA: A foundational, 65-billion-parameter large language model "
input_ids = tokenizer.encode(sentence, return_tensors='pt')

input_ids[0][0]

tokenizer.decode(input_ids[0])

"""# ***4/ Generate and decode text***"""

output = model.generate (input_ids, max_length = 500, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

len(output[0])

tokenizer.decode(output[0], skip_special_tokens=True)

"""## ***5/ Output Result ***bold text***"""

text = tokenizer.decode(output[0], skip_special_tokens=True)

with open('LammaPaper.txt','w') as f:
  f.write(text)

