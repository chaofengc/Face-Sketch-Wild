import torch
from torchvision import transforms
import numpy as np
import itertools
from PIL import Image, ImageDraw, ImageFilter
import dlib
import matplotlib.pyplot as plt

from pthutils import tensorToVar
#  from load_data import CalDoG, Rescale, AddXY 

def read_img_var(img_path, color=1, size=None):
    """
    Read image and convert it to Variable in 0~255.
    Args:
        img_path: str, test image path
        size: tuple, output size (1, C, W, H)
    """
    if color:
        img = Image.open(img_path).convert('RGB') 
    else:
        img = Image.open(img_path).convert('L') 
    if size is not None:
        img = transforms.functional.resize(img, size)
    return tensorToVar(transforms.functional.to_tensor(img)).unsqueeze(0) * 255  

def read_sketch_var(img_path, color=1, size=None, addxy=1, DoG=1):
    """
    Read image and convert it to Variable.
    Args:
        img_path: str, test image path
        size: tuple, output size (W, H)
    """
    img = Image.open(img_path).convert('L') 
    sample = {}
    sample['sketch'] = transforms.functional.resize(img, size)
    trans_list = [CalDoG(), AddXY()]
    trans_filter = [DoG, addxy]
    all_trans = list(itertools.compress(trans_list, trans_filter))
    if len(all_trans):
        trans = transforms.Compose(all_trans)
        face_img = trans(sample)['sketch']
    else:
        face_img = sample['sketch']
    return tensorToVar(transforms.functional.to_tensor(face_img)) * 255

def read_imgAB_var(img_path, AB=0, size=None):
    """
    Read RGB image, resize to given size and convert to Variable.
    Args:
        img_path: str, test image path
        AB: read image A or image B
        size: tuple, output size(W, H)
    """
    img = Image.open(img_path).convert('RGB')
    
    width, height = img.size
    img1 = img.crop((0, 0, np.floor(width / 2).astype('int'), height))
    img2 = img.crop((np.ceil(width / 2).astype('int'), 0, width, height))
    img = img2 if AB else img1 

    if size:
        img = transforms.functional.resize(img, size)
    img_tensor = transforms.functional.to_tensor(img)
    return tensorToVar(img_tensor) * 255
   
def save_var_img(var, save_path=None, size=None):
    """
    Post processing output Variable.
    Args:
        var: Variable, (1, C, H, W)
    """
    out = var.squeeze().data.cpu().numpy()
    out[out>255] = 255  
    out[out<0]   = 0
    if len(out.shape) > 2:
        out = out.transpose(1, 2, 0)
    #  out = transforms.to_pil_image(out, mode='L')
    out = Image.fromarray(out.astype(np.uint8)).convert('RGB')
    if size:
        out = transforms.functional.resize(out, size)
    if save_path:
        out.save(save_path)
    return out


def subtract_imagenet_mean_batch(batch):
    """
    Convert image batch to BGR and subtract imagenet mean
    Batch Size: (B, C, H, W), RGB
    """
    batch = batch[:, [2, 1, 0], :, :]
    batch[:, 0] = batch[:, 0] - 103.939
    batch[:, 1] = batch[:, 1] - 116.779 
    batch[:, 2] = batch[:, 2] - 123.680 
    return batch

def subtract_mean_batch(batch, type):
    """
    Convert image batch to BGR and subtract imagenet mean
    Batch Size: (B, C, H, W), RGB
    """
    #  vggface_mean_bgr = np.array([129.1863, 104.7624, 93.5940]) 
    vgg_mean_bgr = np.array([103.939, 116.779, 123.680]) 
    #  sketch_mean = np.array([191.7165, 191.5756, 191.8129])
    sketch_mean = np.array([np.mean(vgg_mean_bgr)]*3)
    if type == 'face':
        mean_bgr = vgg_mean_bgr
    elif type == 'sketch':
        mean_bgr = sketch_mean
    elif type == 'face_gray':
        mean_bgr = np.mean(vgg_mean_bgr).repeat(3)

    batch = batch[:, [2, 1, 0], :, :]
    batch = batch - tensorToVar(torch.Tensor(mean_bgr)).view(1, 3, 1, 1) 
    return batch

def draw_landmark_mask(img, face_predictor_path='../scripts/shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    img_array = np.array(img)
    dets = detector(img_array, 1)
    if len(dets) < 1:
        return [] 
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img_array, d)
    landmarks = []
    for i in range(68):
        landmarks.append([shape.part(i).x, shape.part(i).y])
    landmarks = np.array(landmarks)
    landmark_parts = [landmarks[0:17],  # cheek
                      landmarks[17:22], # left eyebrow
                      landmarks[22:27], # right eyebrow
                      landmarks[np.r_[27:36, 30]], # nose 
                     landmarks[np.r_[36:42, 36]], # left eye
                      landmarks[np.r_[42:48, 42]], # right eye
                      landmarks[np.r_[48:61, 48]], # outer mouth
                      landmarks[np.r_[61:68, 61]], # inner mouth
                    ]

    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    for i in landmark_parts:
        draw.line(map(tuple, i), fill=255, width=1)
    mask = mask.filter(ImageFilter.GaussianBlur())
    mask = mask.point(lambda p: p > np.median(np.array(mask)) and 255)
    return mask

def save_images_grid(images, filename, cols = 1):
    """Save a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    n_images = len(images)
    fig = plt.figure()
    plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1, wspace=0, hspace=0)
    for n, image in enumerate(images):
        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        if len(image.size) == 2:
            plt.gray()
        a.set_axis_off()
        a.set_frame_on(False)
        plt.imshow(image)
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    #  img = Image.open('../small_data/photos/00.png')
    #  img = cv.imread('../small_data/photos/00.png')
    #  draw_landmark_mask(img)
    dataset_img_list_path = '../data/dataset_img_list.txt'
    sketch_data = []
    for i in open(dataset_img_list_path).readlines():
        sketch_path = i.strip().replace('train_photos', 'train_sketches')
        sketch_data.append(np.array(Image.open(sketch_path).convert('RGB')))
    sketch_data = np.array(sketch_data)
    print(sketch_data.shape, np.mean(sketch_data, axis=(0, 1, 2)))





