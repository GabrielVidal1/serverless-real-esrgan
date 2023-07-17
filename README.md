# Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) aims to provide a practical algorithm for general image/video restoration by extending the powerful ESRGAN with pure synthetic data. This model is useful for enhancing the resolution and quality of general images as well as for specific use-cases such as face restoration. The provided serverless GPU model makes use of the Real-ESRGAN algorithm for an easy-to-use image enhancement tool.

## Inputs

```json
{
  "image": "<base64_encoded_image>",
  "face_enhance": true,
  "outscale": 4
}
```

## Outputs

The output from the model is the enhanced image (in base64 format).

Example:

```json
{
  "output_image": "<base64_encoded_enhanced_image>"
}
```
