import datetime
import time
import math
import os
import gradio
import argparse
import gradio as gr
from diffusers import DiffusionPipeline, AutoencoderTiny
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import threading
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu or mps (MacOS).")
    parser.add_argument("--dtype", default="32", help="Use 16 or 32(default) bit float.")
    parser.add_argument("--offload", action="store_true", help="Offload to CPU to use less VRAM.")
    parser.add_argument("--xformers", action="store_true", help="Use xformers.")
    parser.add_argument("--naughty", action="store_true", help="Naughty or nice.")
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
    gradio_root.queue()
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
)

if args.xformers:
    pipe.enable_xformers_memory_efficient_attention()

match args.dtype:
    case "16":
        dtype = torch.float16
    case _:
        dtype = torch.float32
pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", torch_dtype=dtype, use_safetensors=True
)
pipe.to(device=args.device, dtype=dtype).to(args.device)
if args.offload:
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

queue = []
results = []

def generate_worker():
    global queue, results
    while len(queue) > 0:
        request = queue.pop(0)

        seed = int(request["seed"])
        if seed == -1:
            seed = random.randint(0, 2**32)
        torch.manual_seed(seed)

        images = pipe(
            prompt=request["prompt"],
            num_inference_steps=request["steps"],
            guidance_scale=request["cfg"],
            lcm_origin_steps=50,
            output_type="pil",
            width=request["width"],
            height=request["height"],
        ).images[0]
        results.append(images)
    results.append(None)

def generate(prompt, steps, cfg, size, seed, image_count):
    global queue, results
    (width, height) = size.split('x')
    width = int(width)
    height = int(height)
    result = []
    filename = ""
    preview_name = "./outputs/preview.jpg"

    start_time = time.time()

    # Create queue
    for i in range(image_count):
        queue.append({
            "prompt": prompt,
            "steps": int(steps),
            "cfg": float(cfg),
            "width": width,
            "height": height,
            "seed": seed,
        })
        if not seed == -1:
            seed += 1

    # Start worker
    threading.Thread(target=generate_worker, daemon=True).start()

    # Preview
    grid_xsize = math.ceil(math.sqrt(image_count))
    grid_ysize = math.ceil(image_count / grid_xsize)
    grid_max = max(grid_xsize, grid_ysize)
    pwidth = int(width * grid_xsize / grid_max)
    pheight = int(height * grid_ysize / grid_max)
    preview_grid = Image.new("RGB", (pwidth, pheight))
    preview_grid.save(preview_name, optimize=True, quality=35)
    yield {image: gr.update(value=preview_name, min_width=width, height=height), gallery: gr.update(value=None)}

    i = 0
    generating = True
    while generating:
        # Wait for data
        while len(results) == 0:
            time.sleep(0.1)
        images = results.pop(0)
        if images is None:
            generating = False
            continue
        # Preview
        grid_xpos = int((i % grid_xsize) * (pwidth / grid_xsize))
        grid_ypos = int(math.floor(i / grid_xsize) * (pheight / grid_ysize))
        preview = images.resize((int(width / grid_max), int(height / grid_max)))
        preview_grid.paste(preview, (grid_xpos, grid_ypos))
        preview_grid.save(preview_name, optimize=True, quality=35)

        # Save
        filename = generate_temp_filename(index=i+1)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        metadata = PngInfo()
        metadata.add_text("parameters", f"prompt: {prompt}\n\nsteps: {steps}\ncfg: {cfg}\nwidth: {width} height: {height}")
        images.save(filename, pnginfo=metadata)
        result.append(filename)
        i+=1
        yield {image: gr.update(value=preview_name)}

    if image_count > 1:
        result.insert(0, preview_name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[91mTime taken: {elapsed_time:0.2f} seconds\033[0m")

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
    block.load(js=scripts)
    with gr.Row():
        gr.HTML()
        image = gr.Image(
            min_width=512,
            height=512,
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
        seed = gr.Number(
            label="Seed (-1 is random)",
            precision=0,
            value=-1,
        )
        image_count = gr.Slider(
            label="Image number",
            minimum=1,
            maximum=50,
            step=1,
            value=1,
        )

    def gallery_change(evt: gr.SelectData):
        return evt.value["image"]["path"]

    gallery.select(gallery_change, None, image)

    submit.click(
        fn=generate,
        inputs=[prompt, steps, cfg, size, seed, image_count],
        outputs=[image, gallery],
    )


launch(args, gradio_root)

