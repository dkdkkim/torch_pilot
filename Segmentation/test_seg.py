import torch, pathlib
import numpy as np
from transformations import normalize_uint8
from model_detection import Model
from visual import fig_3views_with_lbl
import matplotlib.pyplot as plt

def predict(img, model, preprocess, postprocess):
    model.eval()
    img = preprocess(img)
    x = torch.from_numpy(img).cuda()
    with torch.no_grad():
        out = model(x)

    out_softmax = torch.softmax(out, dim=1)
    result = postprocess(out_softmax)

    return result

def preprocess(img: np.ndarray):
    img = normalize_uint8(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)
    img = img.cpu().numpy()
    img = np.squeeze(img)
    return img


root = pathlib.Path('/data/dk/datasets_CROPS/crops_fixed_scale_uint8/')

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext+'/**/*0.5_0.npy') if file.is_file()]
    return filenames

images_names = get_filenames_of_path(root / 'valid/TP')
targets_names = [pathlib.Path(str(inp).replace('crops_fixed_scale_uint8','crops_lbl_fixed_rev')) for inp in images_names]

images = [np.load(img_name) for img_name in images_names]
targets = [np.load(tar_name) for tar_name in targets_names]

model = Model().cuda()

model_name = f"/data/dk/exp/torch_seg_test/model_test.pt"
model_weights = torch.load(pathlib.Path.cwd() / model_name)

model.load_state_dict((model_weights))

output = [predict(img, model, preprocess, postprocess) for img in images]

for inp, out in zip(images,output):
    fig_3views_with_lbl(inp,out,True)
    plt.show()