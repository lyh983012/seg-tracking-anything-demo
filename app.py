import gradio as gr
import sys
sys.path.append('./xmem')
from argparse import ArgumentParser
import os
import torch
import numpy as np
import time
import cv2
from model.network import XMem
from inference.interact.resource_manager import ResourceManager
from inference.inference_core import InferenceCore
from demoutils.tools import *
#need install
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator,SamPredictor
torch.set_grad_enabled(False)
#mask2former 模型
sys.path.append('./Mask2Former')
import tempfile
from pathlib import Path
import cog
# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
# import Mask2Former project
from mask2former import add_maskformer2_config
from demoutils.tools import _create_text_labels

parser = ArgumentParser([])
parser.add_argument('--model', default='../data/saves_demo/XMem.pth')
parser.add_argument('--s2m_model', default='../data/saves_demo/s2m.pth')
parser.add_argument('--fbrs_model', default='../data/saves_demo/fbrs.pth')

parser.add_argument('--images', help='Folders containing input images.', default=None)
parser.add_argument('--video', help='Video file readable by OpenCV.', default="../data/valvedio/WIN_20230619_14_31_58_Pro.mp4")
parser.add_argument('--workspace', help='directory for storing buffered images (if needed) and output masks', default="../data/workspace/")

parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)
parser.add_argument('--num_objects', type=int, default=1)

# Long-memory options
# Defaults. Some can be changed in the GUI.
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                    type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128) 

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', type=int, default=10)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
parser.add_argument('--size', default=480, type=int, 
            help='Resize the shorter side to this size. -1 to use original resolution. ')
args = parser.parse_args(args=['--size', '480'])


args.workspace += args.video.split('.')[-2].split('/')[-1]

nowList = []

def upload_file(files):
    file_paths = [file.name for file in files]
    print(file_paths)
    nowList = file_paths
    return file_paths

with gr.Blocks() as demo:
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video"], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)

def infer():
    for v in nowList:
        args = parser.parse_args(args=['--size', '480','--video','../data/valvedio/'+file_paths])
        name = args.video.split('.')[-2].split('/')[-1]
        args.workspace += name
        print(name,args.video,args.workspace)
        if not os.path.exists(args.workspace):
            os.mkdir(args.workspace)
        config = vars(args)
        config['dummyName'] = './results/' + name + '_seg.mp4'
        config['enable_long_term'] = True
        config['enable_long_term_count_usage'] = True
        config['segPerson'] = True
        config['segAll'] = False
        vos = vedioSeg(config)            
        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            vos.segFromFirst()
    
demo.launch(server_name="0.0.0.0", share=True, server_port=38328)