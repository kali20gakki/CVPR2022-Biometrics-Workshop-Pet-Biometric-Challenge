from glob import glob
from pyexpat import model
from re import S
import pandas as pd
import os, cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
import yaml
import numpy as np
from tqdm import tqdm
from models import *
from datasets import *

data = pd.read_csv(r'dataset/pet_biometric_challenge_2022/test/test_data.csv', encoding='utf-8')
images_pair = [(imageA_name, imageB_name) for imageA_name, imageB_name in zip(data['imageA'], data['imageB'])]


config_dir_list = ['commit_config', 'commit_config2'] # 配置文件夹
weights_dir = 'runs'     # 权重文件夹
output_dir = 'test_results' # 输出csv文件夹

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


if __name__ == '__main__':
    bs = 128
    
    for config_dir in config_dir_list:
        for cfg_file in tqdm(os.listdir(config_dir)):
            # 打开yaml
            cfg_path = os.path.join(config_dir, cfg_file)
            print(f'open {cfg_path}')
            with open(cfg_path, 'r', encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            test_dataset = BiometricsTestDataset('dataset/pet_biometric_challenge_2022/test/test_data.json',
                                                'dataset/pet_biometric_challenge_2022/test/test', imgsz=cfg['imgsz'])
            test_loader = Data.DataLoader(
                dataset=test_dataset,  
                batch_size=bs,       
                shuffle=False,     
                num_workers=4,
                drop_last=False 
            )
            
            # 加载模型
            model_name = cfg['model_name']
            if model_name == 'MultiStageEmbeddingNetWithBNNeck':
                model = MultiStageEmbeddingNetWithBNNeck(stride = cfg['stride'], 
                        encoder_name=cfg['encoder_name'], embedding_size=cfg['embedding_size'], pretrained=False)
            elif model_name == 'EmbeddingNetWithBNNeck':
                model = EmbeddingNetWithBNNeck(stride = cfg['stride'], 
                        encoder_name=cfg['encoder_name'], embedding_size=cfg['embedding_size'], pretrained=False)
            elif model_name == 'MultiStageEmbeddingNetWithBNNeckV2':
                model = MultiStageEmbeddingNetWithBNNeckV2(stride = cfg['stride'], 
                        encoder_name=cfg['encoder_name'], embedding_size=cfg['embedding_size'], pretrained=False)
            elif model_name == 'EmbeddingNetLocalkWithBNNeck':
                model = EmbeddingNetLocalkWithBNNeck(stride = cfg['stride'],
                        encoder_name=cfg['encoder_name'], embedding_size=cfg['embedding_size'], pretrained=False)
            elif model_name == 'MultiStageEmbeddingNetV2':
                model = MultiStageEmbeddingNetV2(stride = cfg['stride'],
                        encoder_name=cfg['encoder_name'], embedding_size=cfg['embedding_size'], pretrained=False)
            print(model_name)

            # predict
            for n in cfg['predict_ckpt']:
                weight_path = os.path.join(weights_dir, cfg['logname'], f'ckpt_{n}.pth')
                save_csv_path = os.path.join(output_dir, 'test_' + cfg['logname'] + f'_{n}.csv')
                print(f'load {weight_path}...')

                state_dict = torch.load(weight_path)
                model.load_state_dict(state_dict, strict=True)
                model.cuda()
                model.eval()

                pred_list = []
                for imgA, imgB, imgA_name, imgB_name in tqdm(test_loader):
                    with torch.no_grad():
                        imgA = imgA.cuda()
                        imgB = imgB.cuda()
                        
                        if 'face' in save_csv_path or \
                            'ArcFace' in save_csv_path or \
                            'Face' in save_csv_path:
                            out1, out2 = model.get_pair_encoder_embedding(imgA, imgB) 
                        else:
                            out1, out2 = model.get_pair_embedding(imgA, imgB)

                        similarity = torch.cosine_similarity(out1, out2)
                        pred_list += similarity.cpu().numpy().tolist()

                d = {'imageA': data['imageA'], 'imageB': data['imageB'], 'prediction': pred_list}
                df = pd.DataFrame(d)
                df.to_csv(save_csv_path, index=None)
                print(f'save {save_csv_path} done')

