from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)  # web:7
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)  # web:7

prefix = "items: "
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95,
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {"<sep>": "--", "<section>": "\n"}  # web:7


def skip_special_tokens(text: str, special_tokens_list):
    for token in special_tokens_list:
        text = text.replace(token, "")
    return text


def target_postprocessing(texts, special_tokens_list):
    if not isinstance(texts, list):
        texts = [texts]

    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens_list)
        for k, v in tokens_map.items():
            text = text.replace(k, v)
        new_texts.append(text)
    return new_texts


def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]

    encodings = tokenizer(
        inputs,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt",  # PyTorch tensors
    )

    output_ids = model.generate(
        input_ids=encodings.input_ids,
        attention_mask=encodings.attention_mask,
        **generation_kwargs,
    )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    generated_recipe = target_postprocessing(decoded, special_tokens)
    return generated_recipe


def make_recipe(ingredients: str) -> str:
    """
    Gradio-facing function: takes one ingredients string, returns formatted recipe.
    """
    if not ingredients.strip():
        return "Please enter some ingredients first."

    generated_list = generation_function(ingredients)
    text = generated_list[0]

    sections = text.split("\n")
    headline = ""
    lines = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if section.startswith("title:"):
            section = section.replace("title:", "")
            headline = "TITLE"
        elif section.startswith("ingredients:"):
            section = section.replace("ingredients:", "")
            headline = "INGREDIENTS"
        elif section.startswith("directions:"):
            section = section.replace("directions:", "")
            headline = "DIRECTIONS"

        if not headline:
            continue

        if headline == "TITLE":
            lines.append(f"[{headline}]: {section.strip().capitalize()}")
        else:
            section_info = [
                f"  - {i+1}: {info.strip().capitalize()}"
                for i, info in enumerate(section.split("--"))
                if info.strip()
            ]
            lines.append(f"[{headline}]:")
            lines.extend(section_info)

    if not lines:
        return "No recipe could be generated. Please try different ingredients."

    return "\n".join(lines)


with gr.Blocks(title="AI Recipe Generator") as demo:
    gr.Markdown("## Enter your ingredients to generate a recipe")

    with gr.Row():
        ingredients_box = gr.Textbox(
            label="Ingredients",
            placeholder="e.g. macaroni, butter, salt, bacon, milk, flour, pepper, cream corn",
            lines=8,
        )
        recipe_box = gr.Textbox(
            label="Generated Recipe",
            lines=12,
            interactive=False,
        )

    generate_btn = gr.Button("Generate recipe")

    generate_btn.click(
        fn=make_recipe,
        inputs=ingredients_box,
        outputs=recipe_box,
    )

if __name__ == "__main__":
    demo.launch()
    