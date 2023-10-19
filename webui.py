import datetime
import math
import os
import gradio
import argparse
import gradio as gr
from diffusers import DiffusionPipeline
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowvram", action="store_true", help="Try to use less VRAM.")
    parser.add_argument("--naughty", action="store_true", help="...or nice.")
    parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
    parser.add_argument(
        "--share", action="store_true", help="Set whether to share on Gradio."
    )
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

def or_nice(image, device, dtype):
    return image, None

pipe = DiffusionPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    custom_pipeline="latent_consistency_txt2img",
    custom_revision="main"
)

pipe.enable_xformers_memory_efficient_attention()
pipe.to(torch_device="cuda", torch_dtype=torch.float32)
if args.lowvram:
    pipe.enable_sequential_cpu_offload()
if args.naughty:
    pipe.run_safety_checker = or_nice

def generate_temp_filename(index=1, folder="./outputs/", extension="png"):
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{time_string}_{index}.{extension}"
    result = os.path.join(folder, date_string, filename)
    return os.path.abspath(os.path.realpath(result))

def generate(prompt, steps, cfg, size, image_count):
    (width, height) = size.split('x')
    width = int(width)
    height = int(height)
    result = []
    filename = ""
    preview_name = "./outputs/preview.jpg"

    # Preview
    grid_xsize = math.ceil(math.sqrt(image_count))
    grid_ysize = math.ceil(image_count / grid_xsize)
    grid_max = max(grid_xsize, grid_ysize)
    pwidth = int(width * grid_xsize / grid_max)
    pheight = int(height * grid_ysize / grid_max)
    preview_grid = Image.new("RGB", (pwidth, pheight))
    preview_grid.save(preview_name, optimize=True, quality=35)
    yield {image: gr.update(value=preview_name, shape=[width, height]), gallery: gr.update(value=None)}

    for i in range(image_count):
        filename = generate_temp_filename(index=i+1)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        images = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            lcm_origin_steps=50,
            output_type="pil",
            width=width,
            height=height
        ).images[0]

        # Preview
        grid_xpos = int((i % grid_xsize) * (pwidth / grid_xsize))
        grid_ypos = int(math.floor(i / grid_xsize) * (pheight / grid_ysize))
        preview = images.resize((int(width / grid_max), int(height / grid_max)))
        preview_grid.paste(preview, (grid_xpos, grid_ypos))
        preview_grid.save(preview_name, optimize=True, quality=35)

        # Save
        metadata = PngInfo()
        metadata.add_text("parameters", f"prompt: {prompt}\n\nsteps: {steps}\ncfg: {cfg}\nwidth: {width} height: {height}")
        images.save(filename, pnginfo=metadata)
        result.append(filename)
        yield {image: gr.update(value=preview_name)}

    if image_count > 1:
        result.insert(0, preview_name)

    yield {
        image: gr.update(value=preview_name if image_count > 1 else filename),
        gallery: gr.update(value=result),
    }

scripts = """
function generate_shortcut(){
  document.addEventListener('keydown', (e) => {
    let handled = false;
    if (e.key !== undefined) {
      if ((e.key === 'Enter' && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    } else if (e.keyCode !== undefined) {
      if ((e.keyCode === 13 && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    }
    if (handled) {
      const button = document.getElementById('generate');
      if (button) button.click();
      e.preventDefault();
    }
  });
}
"""

gradio_root = gr.Blocks(
    title="LCM webui",
    theme=None,
    analytics_enabled=False,
).queue()

with gradio_root as block:
    block.load(_js=scripts)
    with gr.Row():
        gr.HTML()
        image = gr.Image(
            type="filepath",
            visible=True,
            show_label=False,
        )
        gr.HTML()
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

    @gallery.select(
        inputs=[gallery], outputs=[image], show_progress="hidden"
    )
    def gallery_change(files, sd: gr.SelectData):
        return files[sd.index]["name"]

    submit.click(
        fn=generate,
        inputs=[prompt, steps, cfg, size, image_count],
        outputs=[image, gallery],
    )


launch(args, gradio_root)

