import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate(sms):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2-large", pad_token_id=tokenizer.eos_token_id)

    # encodes the data in a way that helps the model "understand it"
    input_ids = tokenizer.encode(sms, return_tensors='pt')

    # generate text until the output length (which includes the context length) reaches 50
    output = model.generate(input_ids, max_length=100, num_beams=5,
                            no_repeat_ngram_size=2, early_stopping=True)

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text
