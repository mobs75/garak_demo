from transformers import AutoModelForCausalLM, AutoTokenizer

# Nome del modello da scaricare
model_name = "EleutherAI/gpt-neo-2.7B"  # Sostituisci con "gpt2" se hai meno risorse
local_model_path = "./models/gpt-neo-2.7B"  # Percorso locale per salvare il modello

def download_model():
    """Scarica il modello e il tokenizer da Hugging Face"""
    print(f"Scaricando il modello '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print(f"Modello salvato in '{local_model_path}'")

def test_model():
    """Carica il modello locale e genera una risposta"""
    print(f"Caricando il modello da '{local_model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)

    # Prompt per la demo
    prompt = "Cosa puoi dirmi sul futuro dell'intelligenza artificiale?"
    print(f"Prompt: {prompt}")

    # Tokenizza il prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Genera una risposta
    print("Generando la risposta...")
    outputs = model.generate(
        **inputs,
        max_length=300,  # Lunghezza massima della risposta
        temperature=0.8,  # Creatività
        top_p=0.9,  # Probabilità cumulativa
        do_sample=True,  # Abilita il sampling
        pad_token_id=tokenizer.eos_token_id
    )

    # Decodifica e stampa il risultato
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Risultato generato dal modello:")
    print(response)

if __name__ == "__main__":
    download_model()
    test_model()



