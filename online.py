'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''
import os
import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")

import shutil
import numpy as np
from subprocess import call

from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj

DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/weights.npy'


def cmd_exists(cmd):
    return shutil.which(cmd) is not None

def compressImage(srcPath,dstPath):  
    for filename in os.listdir(srcPath):  
        if not os.path.exists(dstPath):
                os.makedirs(dstPath)        
        srcFile=os.path.join(srcPath,filename)
        dstFile=os.path.join(dstPath,filename)
        print(srcFile)

        if srcFile.endswith(".png") or srcFile.endswith(".JPG") or srcFile.endswith(".jpg"):     
            sImg=Image.open(srcFile)  
            w,h=sImg.size  
            print(w,h)
            dImg=sImg.resize((127,127),Image.ANTIALIAS)  
            dImg.save(dstFile) 
            print(dstFile+" compressed succeeded")
        '''
        if os.path.isdir(srcFile):
            compressImage(srcFile,dstFile)
        '''

def load_demo_images():
    ims = []
    compressImage("online/","online/resized")
    for (dirpath, dirnames, filenames) in os.walk('online/resized'):
        for file in sorted(filenames):
            if file.endswith(".png") or file.endswith(".JPG") or file.endswith(".jpg"):
                print(os.path.join(dirpath, file))
                im = Image.open(os.path.join(dirpath, file))
                ims.append([np.array(im).transpose(
                    (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction_online.obj'

    # load images
    demo_imgs = load_demo_images()


    # Use the default network model
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass(compute_grad=False)  # instantiate a network
    net.load(DEFAULT_WEIGHTS)                        # load downloaded weights
    solver = Solver(net)                # instantiate a solver

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)

    # Save the prediction to an OBJ file (mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)

    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    if cmd_exists('meshlab'):
        call(['meshlab', pred_file_name])
    else:
        print('Meshlab not found: please use visualization of your choice to view %s' %
              pred_file_name)


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    main()
