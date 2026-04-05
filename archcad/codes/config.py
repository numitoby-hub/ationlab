import os
import torch

BASE_DIR      = r"C:\Users\user\Documents\ysu\ArchCAD"
DATA_DIR      = os.path.join(BASE_DIR, "data")
CACHE_DIR     = os.path.join(BASE_DIR, "mask_cache")
JSON_DIR      = os.path.join(DATA_DIR, "json", "json")
PNG_DIR       = os.path.join(DATA_DIR, "png", "png")
SEGFORMER_OUT = os.path.join(BASE_DIR, "segformer_output")
GATV2_OUT     = os.path.join(BASE_DIR, "gatv2_output")
MODEL_OUT     = os.path.join(BASE_DIR, "pancadnet_v2_output")
GRAPH_DIR     = os.path.join(BASE_DIR, "graph_cache_v3")

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE      = 700
SCALE_ORG     = 980.0
FEAT_SIZE     = 128
NUM_GNN_CLS   = 31
NUM_SEG_CLS   = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TYPE_MAP      = {"LINE": 0, "ARC": 1, "CIRCLE": 2}
NUM_TYPES     = 4  # LINE, ARC, CIRCLE, other

CLASS_NAMES = ['Axis & Grid','Single Door','Double Door','Parent-Child Door','Other Door','Elevator','Staircase','Sink','Urinal','Toilet','Bathtub','Squat Toilet','Other Fixtures','Drain','Table','Chair','Bed','Sofa','Hole','Glass','Wall','Concrete Column','Steel Column','Concrete Beam','Steel Beam','Parking Space','Foundation','Pile','Rebar','Fire Hydrant','Others']

RAW_TO_NEW_CLASS = {i: i for i in range(30)}
RAW_TO_NEW_CLASS[100] = 30

THING_CLASSES = [1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,21,22,25,27,29]
STUFF_CLASSES = [0,12,13,19,20,23,24,26,28,30]

SEGFORMER_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
LINE_SAMPLE_N  = 16
LINE_SAMPLE_M  = 4
GEO_FEAT_DIM   = 16  # 8 geo + 4 type_onehot + 3 rgb + 1 line_width
EDGE_FEAT_DIM  = 4
GAT_HIDDEN     = 128
GAT_HEADS      = 4
GAT_DROPOUT    = 0.2
GRAPH_K_LIST   = [3, 8, 16]
NUM_QUERIES    = 100
DEC_HIDDEN     = 256
DEC_HEADS      = 8
DEC_LAYERS     = 3
LR             = 2e-4
WEIGHT_DECAY   = 0.01
EPOCHS         = 50
LAMBDA_CLS     = 2.0
LAMBDA_BCE     = 5.0
LAMBDA_DICE    = 5.0
