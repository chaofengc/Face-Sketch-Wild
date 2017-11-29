import numpy as np

from torch import nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.onnx

import onnx
import onnx_caffe2.backend

from models.sketch_net_v1 import SketchNetV1

pth_model = SketchNetV1(in_channels=3, out_channels=3)
# Load pretrained model weights
model_weight_path = './weight/pix2pix-face_sketch_data-AtoB-SketchNetV1-IN-lr0.0010-layers00111-loss_4-weight-1.0e+00-1.0e-01-1.0e-04-epoch60-_coloraug-sizeaug/epochs-059.pth' 
batch_size = 1    # just a random number
# Initialize model with the pretrained weights
torch_model.load_state_dict(model_zoo.load_url(model_weight_path))
# set the train mode to false since we will only run the forward pass.
torch_model.train(False)

# Input to the model
x = Variable(torch.randn(batch_size, 1, 224, 224), requires_grad=True)

# Export the model
torch_out = torch.onnx._export(torch_model,                 # model being run 
                                x,                          # model input (or a tuple for multiple inputs) 
                                "./caffe2_model/face_sketch_test.onnx",    # where to save the model (can be a file or file-like object)
                                export_params=True)         # store the trained parameter weights inside the model file

onnx_model = onnx.load("./caffe2_model/face_sketch_test.onnx")
prepared_backend = onnx_caffe2.backend.prepare(model, device="CUDA:0")

# run the model in Caffe2

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input
W = {model.graph.input[0].name: x.data.numpy()}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# Verify the numerical correctness upto 3 decimal places
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

print("Exported model has been executed on Caffe2 backend, and the result looks good!")
