import base64
import cv2
import os

import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan.realesrgan import RealESRGANer
from realesrgan.realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer


def prepare_model(
    model_name,
    model_path,
    denoise_strength,
    tile_pad,
    tile,
    pre_pad,
    fp32,
    gpu_id,
    outscale,
):
    model_name = model_name.split(".")[0]
    if model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
    elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    elif model_name == "realesr-animevideov3":  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        ]
    elif model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]

    if model_path is not None:
        model_path = model_path
    else:
        model_path = os.path.join("weights", model_name + ".pth")
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url=url,
                    model_dir=os.path.join(ROOT_DIR, "weights"),
                    progress=True,
                    file_name=None,
                )

    dni_weight = None
    if model_name == "realesr-general-x4v3" and denoise_strength != 1:
        wdn_model_path = model_path.replace(
            "realesr-general-x4v3", "realesr-general-wdn-x4v3"
        )
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id,
    )

    face_enhancer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=outscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )

    return upsampler, face_enhancer


def perform_inference(
    upsampler,
    face_enhancer,
    base64_img,
    outscale=4,
    face_enhance=False,
    ext="auto",
):
    # Convert base64 to numpy array
    img_data = base64.b64decode(base64_img)
    img_np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = "RGBA"
    else:
        img_mode = None

    try:
        if face_enhance and face_enhancer is not None:
            _, _, output_img = face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output_img, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print("Error", error)
        print(
            "If you encounter CUDA out of memory, try to set --tile with a smaller number."
        )
    else:
        if ext == "auto":
            extension = "png" if img_mode == "RGBA" else "jpg"
        else:
            extension = ext

        # Convert the output image to base64
        _, buffer = cv2.imencode(f".{extension}", output_img)
        output_base64 = base64.b64encode(buffer).decode("utf-8")

    return output_base64


def main():
    requiredArgs = {
        "image": "base64 encoded image",
    }

    args = {
        "input": "example_folder",
        "model_name": "RealESRGAN_x4plus",
        "output": "results_folder",
        "denoise_strength": 0.7,
        "outscale": 3,
        "model_path": "path/to/model",
        "suffix": "restored",
        "tile": 5,
        "tile_pad": 15,
        "pre_pad": 5,
        "face_enhance": True,
        "fp32": False,
        "alpha_upsampler": "bicubic",
        "ext": "png",
        "gpu_id": 1,
    }

    upsampler, face_enhancer = prepare_model(
        args["model_name"],
        args["model_path"],
        args["denoise_strength"],
        args["tile_pad"],
        args["tile"],
        args["pre_pad"],
        args["fp32"],
        args["gpu_id"],
        args["outscale"],
    )

    # image in base 64
    image = perform_inference(
        upsampler,
        face_enhancer,
        args["image"],
        args["outscale"],
        args["face_enhance"],
        args["ext"],
    )


if __name__ == "__main__":
    main()
