from inference_realesrgan_client import perform_inference, prepare_model
from potassium import Potassium, Request, Response
from transformers import pipeline
import torch

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    args = {
        "model_name": "RealESRGAN_x4plus",
        "model_path": None,
        "denoise_strength": 0.7,
        "tile": 5,
        "tile_pad": 15,
        "pre_pad": 5,
        "fp32": False,
        "gpu_id": 0 if torch.cuda.is_available() else -1,
        "outscale": 3,
    }

    upsampler, face_enhancer = prepare_model(
        args["model_name"],
        args["model_path"],
        args["denoise_strength"],
        args["tile"],
        args["tile_pad"],
        args["pre_pad"],
        args["fp32"],
        args["gpu_id"],
        args["outscale"],
    )

    context = {
        "upsampler": upsampler,
        "face_enhancer": face_enhancer,
        "outscale": args["outscale"],
        "face_enhance": True,  # can be set as per requirement
        "ext": "png",  # can be set as per requirement
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    image_b64 = request.json.get("image")  # assuming base64 image comes as 'image' key
    face_enhance = request.json.get("face_enhance")
    outscale = request.json.get("outscale")

    # if image is not provided return a 400 response
    if not image_b64:
        return Response(json={"error": "Image not provided"}, status=400)

    # Run the model and generate the output
    image_out = perform_inference(
        context["upsampler"],
        context["face_enhancer"],
        image_b64,
        outscale if outscale else context["outscale"],
        face_enhance if face_enhance else context["face_enhance"],
        context["ext"],
    )

    return Response(json={"output_image": image_out}, status=200)


if __name__ == "__main__":
    app.serve()
