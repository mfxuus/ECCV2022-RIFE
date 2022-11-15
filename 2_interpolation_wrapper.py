import os
import cv2
import torch
import time
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------
INTERMEDIATE_DIR = 'E:\\4_GithubProjects\\fps-webcam-demo\\intermediate_images'
if not os.path.exists(INTERMEDIATE_DIR):
    os.mkdir(INTERMEDIATE_DIR)


CAM_STREAM_DIR = 'E:\\4_GithubProjects\\fps-webcam-demo\\cam_stream'
ECCV_DIR = 'E:\\4_GithubProjects\\fps-webcam-demo\\ECCV2022-RIFE'

# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model('train_log', -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model('train_log', -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model('train_log', -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model('train_log', -1)
    print("Loaded ArXiv-RIFE model")
model.eval()
model.device()


# ----------------------------------

def read_first_image():
    padding = None
    while padding is None:
        first_img_path = os.path.join(CAM_STREAM_DIR, '0.png')
        if os.path.exists(first_img_path):
            img0 = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
            # precalculate these
            n, c, h, w = img0.shape
            ph = ((h - 1) // 32 + 1) * 32
            pw = ((w - 1) // 32 + 1) * 32
            padding = (0, pw - w, 0, ph - h)
        else:
            time.sleep(1)
            print('-- waiting for first image --')
    return h, w, padding


def main():
    h, w, padding = read_first_image()

    def get_image_data(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img = F.pad(img, padding)
        return img

    def get_next_img_data(ind):
        img_path = os.path.join(CAM_STREAM_DIR, f"{ind}.png")
        while True:
            try:
                img = get_image_data(img_path)
                break
            except:
                pass
        return img

    start_ind = 0
    end_ind = 1

    img0 = None
    img1 = None

    img0_path = os.path.join(CAM_STREAM_DIR, f"{start_ind}.png")
    img0 = get_image_data(img0_path)

    # while True
    while start_ind <= 10:
        img1 = get_next_img_data(end_ind)
        img_list = [img0, img1]
        mid = model.inference(img_list[0], img_list[1])
        cv2.imwrite(
            f'{INTERMEDIATE_DIR}\\{start_ind + 0.5}.png',
            (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        )

        start_ind = end_ind
        end_ind += 1
        img0 = img1


if __name__ == '__main__':
    main()
