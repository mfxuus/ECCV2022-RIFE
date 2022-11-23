import os
import cv2
import sys
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

print(os.getcwd())
os.chdir(ECCV_DIR)
sys.path.append(ECCV_DIR)

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

# def read_first_image():
#     padding = None
#     while padding is None:
#         first_img_path = os.path.join(CAM_STREAM_DIR, '0.png')
#         if os.path.exists(first_img_path):
#             img0 = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
#             img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
#             # precalculate these
#             n, c, h, w = img0.shape
#             ph = ((h - 1) // 32 + 1) * 32
#             pw = ((w - 1) // 32 + 1) * 32
#             padding = (0, pw - w, 0, ph - h)
#         else:
#             time.sleep(1)
#             print('-- waiting for first image --')
#     return h, w, padding


def interpolate_wrapper(input_q, output_q):
    # h, w, padding = read_first_image(input_q)
    # specify the input dimensions
    h = 480
    w = 640
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)

    def get_image_data(img):
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img = F.pad(img, padding)
        return img

    # start_ind = 0
    # end_ind = 1

    # img0 = None
    # img1 = None

    # img0_path = os.path.join(CAM_STREAM_DIR, f"{start_ind}.png")
    img0 = input_q.get()
    img0 = get_image_data(img0)

    # while True
    start_time = time.time()
    i = 0
    while time.time() - start_time <= 10:
        print(i)
        i += 1
        img1 = input_q.get()
        img1 = get_image_data(img1)
        img_list = [img0, img1]
        mid = model.inference(img_list[0], img_list[1])

        img0_output = (img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        mid_output = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        output_q.put(img0_output)
        output_q.put(mid_output)
        print('===')
        print(output_q.qsize())

        img0 = img1


# if __name__ == '__main__':
#     main()
