from transformers import AutoModelForCausalLM, AutoTokenizer

#   python3 -m garak --model_name GroNLP/gpt2-small-italian --model_type huggingface.Pipeline --probes promptinject --generator_options '{"max_length":150}' --generations 10

#   python3 -m garak --model_name GroNLP/gpt2-small-italian --model_type huggingface.Pipeline --probes packagehallucination --generator_options '{"max_length":150}' --generations 10

def test_gpt2_italian():
    """Utilizza GPT-2 in italiano direttamente da Hugging Face."""
    print("Caricando GPT-2 italiano da Hugging Face...")

    # Carica il tokenizzatore e il modello per l'italiano
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    model = AutoModelForCausalLM.from_pretrained("GroNLP/gpt2-small-italian")

    # Prompt per generare testo
    prompt = "Qual è il futuro dell'intelligenza artificiale in Italia?"
    print(f"Prompt: {prompt}")

    # Tokenizza il prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Genera una risposta
    print("Generando la risposta...")
    outputs = model.generate(
        **inputs,
        max_length=150,   # Lunghezza massima della risposta
        temperature=0.7,  # Livello di creatività
        top_p=0.9,        # Nucleus sampling
        do_sample=True,   # Abilita il sampling
        pad_token_id=tokenizer.eos_token_id
    )

    # Decodifica e stampa la risposta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Risultato generato dal modello:")
    print(response)

if __name__ == "__main__":
    test_gpt2_italian()
