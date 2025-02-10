import os.path
import cv2
import numpy as np
import torch
from scipy.io import loadmat
import utils_sisr
from models.ScaleResidualNet import Net as denoiser_net
from models.DeepResidualNet import Net as parameter_estimation_net

def main():
    iter_num = 6
    sf = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    denoiser_model_path = './model_zoo/denoiser.pth'
    parameter_estimation_model_path = './model_zoo/parameterEstimation.pth'

    L_folder = './input/'
    E_folder = './output/'

    # --------------------------------
    # load model
    # --------------------------------

    denoiser_model = denoiser_net()
    denoiser_model.load_state_dict(torch.load(denoiser_model_path), strict=True)
    denoiser_model.eval()
    for _, v in denoiser_model.named_parameters():
        v.requires_grad = False
    denoiser_model = denoiser_model.to(device)

    parameter_estimation_model = parameter_estimation_net()
    parameter_estimation_model.load_state_dict(torch.load(parameter_estimation_model_path), strict=True)
    parameter_estimation_model.eval()
    for _, v in parameter_estimation_model.named_parameters():
        v.requires_grad = False
    parameter_estimation_model = parameter_estimation_model.to(device)

    # --------------------------------
    # load kernel
    # --------------------------------

    k = loadmat(os.path.join('kernels', 'psf.mat'))['kernel']
    k = k.astype(np.float32)
    k /= np.sum(k)  

    # --------------------------------
    # Start
    # --------------------------------
    for im in os.listdir(L_folder):
        if im.endswith('.jpg') or im.endswith('.bmp') or im.endswith('.png'):
            img_name, ext = os.path.splitext(im)
            img_L = cv2.imread(os.path.join(L_folder, im), 0)
            img_L = np.expand_dims(img_L, axis=2)
            img_L = np.float32(img_L/255.)

            np.random.seed(seed=0)

            x = cv2.resize(img_L, (img_L.shape[1]*sf, img_L.shape[0]*sf), interpolation=cv2.INTER_CUBIC)
            if np.ndim(x)==2:
                x = x[..., None]
            x = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0).to(device)
            k_tensor = torch.from_numpy(np.ascontiguousarray(np.expand_dims(k, 2))).to(device)
            img_L_tensor = torch.from_numpy(np.ascontiguousarray(img_L)).to(device)

            FB, FBC, F2B, FBFy = utils_sisr.pre_calculate(img_L_tensor, k_tensor, sf)

            sigma = float(parameter_estimation_model(img_L_tensor))

            for i in range(iter_num):
                if i == 0:
                    tau = 0.30*(sigma**2)/(49**2).float().repeat(1, 1, 1, 1)
                else:
                    sigma_i = float(parameter_estimation_model(x))
                    tau = 0.30*(sigma**2)/(sigma_i**2).float().repeat(1, 1, 1, 1)
                x = utils_sisr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, sf)
                sigma = float(parameter_estimation_model(x))
                x = torch.cat((x, sigma.float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                x = denoiser_model(x)
        img_E = x.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        img_E = np.uint8(img_E/np.max(img_E)*255)
        cv2.imwrite(os.path.join(E_folder, img_name+ext), img_E)     

if __name__ == '__main__':
    main()