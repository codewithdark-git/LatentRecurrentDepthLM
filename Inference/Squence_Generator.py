from transformers import AutoTokenizer, AutoModel
# you can also user AutoModelForCausalLM class 

def generator(prompt, num_interations=3, max_length=50, temperature=0.7):
    tokenizer = AutoTokenizer.from_pretrained("codewithdark/latent-recurrent-depth-lm")
    model = AutoModel.from_pretrained("codewithdark/latent-recurrent-depth-lm")

    # prompt = "In the realm of language modeling"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=max_length, num_iterations=num_interations, temperature=temperature, top_k=50)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

