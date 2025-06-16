from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LLaMA2 (can use a smaller model like TinyLLaMA or LLaMA2-7B if memory allows)
model_name = "NousResearch/Llama-2-7b-chat-hf"  # Or "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def generate_story(caption, tone="creative", mode="story"):
    prompt = f"Write a {tone} {mode} based on this image description: {caption}"
    response = text_gen(prompt, max_new_tokens=200, do_sample=True, temperature=0.9)
    generated = response[0]["generated_text"]

    # Remove prompt from output
    if prompt in generated:
        generated = generated.replace(prompt, "").strip()

    return generated or "⚠️ No output generated."
