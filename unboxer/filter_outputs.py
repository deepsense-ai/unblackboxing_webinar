import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse

from keras.applications.vgg16 import VGG16
from quiver_engine import server

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', 
                        type=str,
                        help='directory with your images')

    parser.add_argument('--port', 
                        type=int,
                        help='port to serve your quiver')

    args = parser.parse_args()
    
    print(os.listdir(args.img_folder))
    
    model = VGG16(include_top=True)
    tmp_dir = os.path.join(args.img_folder,'.tmp')
    if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
    server.launch(model, 
                  input_folder=args.img_folder,
                  temp_folder=tmp_dir,
                  port=args.port)