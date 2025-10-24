import gc

import torch

from vggt.models.vggt import VGGT
from evaluation.quarot.args_utils import get_config
from evaluation.quarot.utils import set_ignore_quantize, quantize_linear, load_qs_parameters, after_resume_qs, model_reparameterize
from vggt.utils.load_fn import load_and_preprocess_images


MODEL_PATH = "vggt_weights/model.pt"
QPARAMS_PATH = "vggt_weights/a44_quant_model_tracker_fixed_e20.pt_sym/"


def load_quantized_model(device):
    model = VGGT()
    model.load_state_dict(torch.load(MODEL_PATH)) #FIXME. Don't store the weights
    model.eval()
    model = model.to(device)

    config = get_config()
    config.update_from_args(
        wbit=4,
        abit=4,
        not_smooth=False,
        not_rot=False,
        lwc=True, #Clip weights
        lac=True,
        rv=True, #Rotation + Smooth ?
        model_id=MODEL_PATH,
    )

    #1. Ignore modules that wont be quantized. Only weights and activations from attention layers will be quantized.
    set_ignore_quantize(model)

    #2. Replace all modules which doesnt include the ignore qunatization flag with a VGGTQuantizedLinear Module
    quantize_linear(model, args=config)
    
    #3. Load quantization parameters of the attention blocks. (frame attention and global attention: alternating attention)
    model = load_qs_parameters(config, model, path=QPARAMS_PATH)

    for param in model.parameters(): #Did I freeze original VGGT? 
        param.requires_grad = False

    #4. Reparameterize the model: compute new fake-quantized weights using params:
    #   - Hadamard rotation
    #   - Smoothing
    #   - Weight clipping
    #   - Param quantization
    # However, all values are stored in the original format. So, an extra step is required. Not provided in their code.
    # Also, if there's no special hardware for int4 operations, weights are stored either packed and casted to
    # another format. The model's memory footprint could be shorter, but memory usage would not reflect useful gains.
    after_resume_qs(model) 

    return model



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("Loading model")
    torch.cuda.memory._record_memory_history(max_entries=100000)

    model = load_quantized_model(device)
    gc.collect()

    image_paths = ["fotos/zacatecas/calderon_1.jpeg"]
    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(images)
    
    gc.collect()

    torch.cuda.memory._dump_snapshot("profile.pkl") #Way more memory than vainilla
    torch.cuda.memory._record_memory_history(enabled=None)

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            print(name, module.weight.dtype, module.weight.shape)
            break

    print(predictions['images'].shape)


if __name__ == '__main__':
    main()
