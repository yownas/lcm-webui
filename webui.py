import gradio
import argparse
import gradio as gr
from diffusers import DiffusionPipeline
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
    parser.add_argument(
        "--share", action="store_true", help="Set whether to share on Gradio."
    )
    parser.add_argument("--naughty", action="store_true", help="...or nice.")
    parser.add_argument(
        "--listen",
        type=str,
        default=None,
        metavar="IP",
        nargs="?",
        const="0.0.0.0",
        help="Set the listen interface.",
    )
    return parser.parse_args()
args = parse_args()

def launch(args, gradio_root):
    gradio_root.queue(concurrency_count=4)
    gradio_root.launch(
        inbrowser=False,
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
    )

def check(image, device, dtype):
    return image, None

pipe = DiffusionPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    custom_pipeline="latent_consistency_txt2img",
    custom_revision="main"
)

pipe.enable_xformers_memory_efficient_attention()
# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
pipe.to(torch_device="cuda", torch_dtype=torch.float32)
# Slow, low vram
#pipe.enable_sequential_cpu_offload()
if args.naughty:
    pipe.run_safety_checker = check

def generate(prompt, steps, cfg, size, image_count):
    (width, height) = size.split('x')
    images = pipe(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        lcm_origin_steps=50,
        output_type="pil",
        width=int(width),
        height=int(height)
    ).images

    return {
        image: gr.update(value=images[0]),
        gallery: gr.update(),
    }

gradio_root = gr.Blocks(
    title="LCM webui",
    analytics_enabled=False,
).queue()

with gradio_root as block:
    with gr.Row():
        image = gr.Image(
            height=680,
            type="filepath",
            visible=True,
            show_label=False,
            image_mode="RGBA"
        )
    with gr.Row():
        gallery = gr.Gallery(
            height=60,
            object_fit="scale_down",
            show_label=False,
            allow_preview=True,
            preview=True,
            visible=True,
        )
    with gr.Group(), gr.Row():
        prompt = gr.Textbox(
            show_label=False,
            placeholder="Type prompt here.",
            container=False,
            autofocus=True,
            elem_classes="type_row",
            lines=4,
            scale=9,
        )
        submit = gr.Button(
            label="Generate",
            value="Generate",
            elem_id="generate",
            scale=1,
        )

    with gr.Row():
        steps = gr.Slider(
            label="Steps (4-8 is recommended)",
            minimum=1,
            maximum=50,
            step=1,
            value=4,
        )
        cfg = gr.Slider(
            label="CFG",
            minimum=0.0,
            maximum=20.0,
            step=0.1,
            value=7.5,
        )
        size = gr.Dropdown(
            label="Size",
            choices=["512x512", "768x512", "512x768", "768x768", "1024x768", "768x1024"],
            value="512x512",
        )
        image_count = gr.Slider(
            label="Image number",
            minimum=1,
            maximum=50,
            step=1,
            value=1,
        )

    submit.click(
        fn=generate,
        inputs=[prompt, steps, cfg, size, image_count],
        outputs=[image, gallery],
    )


launch(args, gradio_root)

