import fire
import torch
import tarfile
import numpy as np
from pathlib import Path

from tr_rosetta_pytorch.tr_rosetta_pytorch import trRosettaNetwork
from tr_rosetta_pytorch.utils import preprocess, d

# paths

CURRENT_PATH = Path(__file__).parent
DEFAULT_MODEL_PATH = CURRENT_PATH / 'models'
MODEL_PATH =  DEFAULT_MODEL_PATH / 'models.tar.gz'
MODEL_FILES = [*Path(DEFAULT_MODEL_PATH).glob('*.pt')]

# extract model files if not extracted

if len(MODEL_FILES) == 0:
    tar = tarfile.open(str(MODEL_PATH))
    tar.extractall(DEFAULT_MODEL_PATH)
    tar.close()

# prediction function

@torch.no_grad()
def get_ensembled_predictions(input_file, output_file=None, model_dir=DEFAULT_MODEL_PATH):
    net = trRosettaNetwork()
    i = preprocess(input_file)

    if output_file is None:
        input_path = Path(input_file)
        output_file = f'{input_path.parents[0] / input_path.stem}.npz'

    outputs = []
    model_files = [*Path(model_dir).glob('*.pt')]

    if len(model_files) == 0:
        raise 'No model files can be found'

    for model_file in model_files:
        net.load_state_dict(torch.load(model_file, map_location=torch.device(d())))
        net.to(d()).eval()
        output = net(i)
        outputs.append(output)

    averaged_outputs = [torch.stack(model_output).mean(dim=0).cpu().numpy().squeeze(0).transpose(1,2,0) for model_output in zip(*outputs)]
    # prob_theta, prob_phi, prob_distance, prob_omega
    output_dict = dict(zip(['theta', 'phi', 'dist', 'omega'], averaged_outputs))
    np.savez_compressed(output_file, **output_dict)
    print(f'predictions for {input_file} saved to {output_file}')

def predict():
    fire.Fire(get_ensembled_predictions)
