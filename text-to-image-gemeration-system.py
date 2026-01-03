# Generated from: text-to-image-gemeration-system.ipynb
# Converted at: 2026-01-03T19:21:12.407Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # **Import Libraries**


!pip install -q diffusers==0.31.0 transformers accelerate safetensors gradio datasets

# # For CUDA 11.x
# !pip install xformers

# # If that fails, try the nightly version:
# # pip install triton==2.0.0.dev20221202
# # pip install xformers

# # **Import Libraries**


import os, torch, gc, warnings, json
warnings.filterwarnings("ignore")

from diffusers import StableDiffusionPipeline
from datetime import datetime
from PIL import Image
import gradio as gr
from difflib import SequenceMatcher

# # **DEVICE + PIPELINE**


# =====================================================
# DEVICE + PIPELINE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing()


# # **HELPERS**


# =====================================================
# HELPERS
# =====================================================
def autocast_device():
    return torch.autocast("cuda") if device == "cuda" else torch.no_grad()

def safe_cleanup():
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

# # **FEEDBACK MEMORY**


# =====================================================
# FEEDBACK MEMORY
# =====================================================
FEEDBACK_FILE = "feedback_memory.json"
feedback_memory = (
    json.load(open(FEEDBACK_FILE)) if os.path.exists(FEEDBACK_FILE) else {}
)

def save_feedback_memory():
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_memory, f, indent=4)


def suggest_improvement(prompt):
    best = ""
    max_ratio = 0
    for old_prompt, info in feedback_memory.items():
        r = SequenceMatcher(None, prompt, old_prompt).ratio()
        if r > 0.5 and info["score"] >= max_ratio:
            best, max_ratio = info["improved"], info["score"]
    return best


# # **HISTORY STORAGE**


# =====================================================
# HISTORY STORAGE
# =====================================================
HISTORY_FILE = "image_history.json"
os.makedirs("history_images", exist_ok=True)

image_history = (
    json.load(open(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else []
)

def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(image_history, f, indent=4)

def add_to_history(img, prompt, tag):
    if img is None:
        return
    filename = f"history_images/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}.png"
    img.save(filename)
    image_history.append({
        "prompt": prompt,
        "file": filename,
        "type": tag,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_history()

def get_gallery_items():
    return [entry["file"] for entry in image_history]

# # **IMAGE GENERATION**


# =====================================================
# IMAGE GENERATION
# =====================================================
def generate_preview(prompt, steps, guidance):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt.", ""
    try:
        with autocast_device():
            img = pipe(
                prompt,
                height=512, width=512,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance)
            ).images[0]

        suggestion = suggest_improvement(prompt)
        add_to_history(img, prompt, "preview")
        return img, suggestion, prompt

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            safe_cleanup()
            pipe.to("cpu")
            return None, "⚠️ Switched to CPU due to low VRAM.", prompt
        raise


def generate_highres(prompt, steps, guidance, width, height):
    if not prompt or not prompt.strip():
        return None, None
    try:
        with autocast_device():
            img = pipe(
                prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance)
            ).images[0]

        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_HD.png"
        img.save(path)

        add_to_history(img, prompt, "highres")
        return img, path

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            safe_cleanup()
            pipe.to("cpu")
            return None, None
        raise


def improve_image(last_prompt, feedback, steps, guidance, preview_img):
    if not last_prompt or not last_prompt.strip():
        return preview_img, last_prompt, "⚠️ Generate a preview first.", ""

    if not feedback or not feedback.strip():
        return preview_img, last_prompt, "⚠️ Provide feedback first.", ""

    if last_prompt in feedback_memory:
        improved = f"{feedback_memory[last_prompt]['improved']}. Refine: {feedback}"
        feedback_memory[last_prompt]["score"] += 1
    else:
        improved = f"{last_prompt}. Improve with: {feedback}"
        feedback_memory[last_prompt] = {"improved": improved, "score": 1}

    save_feedback_memory()

    new_preview, suggestion, _ = generate_preview(improved, steps, guidance)
    add_to_history(new_preview, improved, "refined")

    return new_preview, improved, "✅ Improved preview generated.", suggestion


# # **UI**


# =====================================================
# UI
# =====================================================
with gr.Blocks() as app:
    gr.Markdown("<h1 style='text-align:center;color:#eb4509;'> Text to Image Generation System</h1>")
    gr.Markdown("<p style='text-align:center;color:gray;'>Fast Preview • High-Resolution • Feedback Memory • History</p>")

    with gr.Row():
        # -------- Sidebar --------
        with gr.Column(scale=1, min_width=320):
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="A beautiful fantasy castle on a mountain at sunrise, cinematic, ultra-detailed"
            )

            steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
            guidance = gr.Slider(5.0, 12.0, value=7.5, step=0.1, label="Guidance Scale")

            generate_btn = gr.Button("Generate Image", variant="primary")

            gr.Markdown("### 💬 Improve Image")
            feedback = gr.Textbox(
                label="Feedback",
                lines=2,
                value="Make lighting softer and colors warmer"
            )

            improve_btn = gr.Button("Improve Image", variant="primary")

            gr.Markdown("### ⚡ High-Res Options")
            width_slider = gr.Slider(512, 2048, value=1024, step=64, label="Width")
            height_slider = gr.Slider(512, 2048, value=1024, step=64, label="Height")

            generate_hd_btn = gr.Button("Generate High-Res Image", variant="primary")

            status = gr.Markdown("")
            suggested_box = gr.Textbox(label="Suggested Prompt", interactive=False)

        # -------- Main Panel --------
        with gr.Column(scale=3):
            output_preview = gr.Image(label="Preview", interactive=False)
            output_hd = gr.Image(label="High-Res Output", interactive=False)
            saved_path = gr.Textbox(label="Saved Path", visible=False)
            active_prompt = gr.Textbox(label="Current Prompt", visible=False)

            gr.Markdown("### 🕘 Image History")
            history_gallery = gr.Gallery(columns=4, height=300, allow_preview=True)


    # -------- Actions --------
    
    generate_btn.click(
        fn=generate_preview,
        inputs=[prompt, steps, guidance],
        outputs=[output_preview, suggested_box, active_prompt]
    ).then(get_gallery_items, outputs=history_gallery)

    improve_btn.click(
        fn=improve_image,
        inputs=[active_prompt, feedback, steps, guidance, output_preview],
        outputs=[output_preview, active_prompt, status, suggested_box]
    ).then(get_gallery_items, outputs=history_gallery)

    generate_hd_btn.click(
        fn=generate_highres,
        inputs=[active_prompt, steps, guidance, width_slider, height_slider],
        outputs=[output_hd, saved_path]
    ).then(get_gallery_items, outputs=history_gallery)

app.launch(share=True)




# 1. Realistic photo of a majestic wolf howling at the moon, eerie and powerful.
# 2. Realistic photo of a curious fox in a snowy forest, bright eyes and bushy tail.
# 3. Hyper-realistic painting of a majestic lion in the savannah, detailed fur and lifelike eyes.
# 4. Realistic photo of a graceful deer in a misty forest, soft morning light.
# 5. Realistic photo of a playful puppy in a flower field, bright and cheerful.
# 6. Detailed sketch of a sleek black panther in the moonlight, mysterious and elegant.
# 7. Realistic photo of a curious raccoon in a suburban backyard, playful and mischievous.
# 8. Detailed sketch of a majestic horse running through an open field, powerful and free.
# 9. Realistic photo of a sleepy cat lounging in a sunbeam, cozy and content.
#