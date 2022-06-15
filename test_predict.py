import pandas as pd
import os, cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
import numpy as np
from tqdm import tqdm
from models import *
from datasets import *

data = pd.read_csv(r'dataset/pet_biometric_challenge_2022/test/test_data.csv', encoding='utf-8')
images_pair = [(imageA_name, imageB_name) for imageA_name, imageB_name in zip(data['imageA'], data['imageB'])]


if __name__ == '__main__':
    imgsz = 224
    bs = 128

    #model = EmbeddingNet(encoder_name='resnetv2_101x1_bitm_in21k', pretrained=False)
    #model = MultiStageEmbeddingNetWithBNNeckV2(stride=1, encoder_name='resnetv2_101x1_bitm', embedding_size = 2048, pretrained=False)
    #model = EmbeddingNetWithBNNeck(stride=1, encoder_name='resnetv2_101x1_bitm', pretrained=False)
    #model = MultiStageEmbeddingNetV2(stride=1, encoder_name='resnetv2_101x1_bitm', pretrained=False)
    #model = EmbeddingNetLocalkWithBNNeck(stride=1, encoder_name='resnetv2_101x1_bitm', pretrained=False)
    model = MultiStageEmbeddingNetWithBNNeck(stride = 1, encoder_name='resnetv2_101x1_bitm', pretrained=False)
    #model = SingleStageEmbeddingNetWithBNNeck(encoder_name='swin_base_patch4_window7_224_in22k', embedding_size = 1024, pretrained=False)


    #ckpt_range = [170, 180, 190, 200] # for 200e
    #ckpt_range = [180, 190, 200, 210] # for 200e
    ckpt_range = [180,190,200] # for 200e
    #ckpt_range = [200] # for 200e
    weights_dir = 'runs/final_resnetv2_101x1_in1k_mstage_ArcFace_b150_k6_224_pseudo_val700_adamW_cosine_stride1_wcut_200e'
    log_name = weights_dir.split('/')[-1]
    print()

    for point in ckpt_range:
        weight_path = os.path.join(weights_dir, f'ckpt_{point}.pth')
        save_name = log_name + f'_{point}.csv'

        save_csv_path = os.path.join('test_results_new', 'test_' + save_name)
        state_dict = torch.load(weight_path)

        # for key in state_dict:
        #     print(key, ' --> ', state_dict[key].shape)


        print(f'load weight {weight_path}...')
        model.load_state_dict(state_dict, strict=True)
        model.cuda()
        model.eval()
        
        test_dataset = BiometricsTestDataset('dataset/pet_biometric_challenge_2022/test/test_data.json',
                                            'dataset/pet_biometric_challenge_2022/test/test', imgsz=imgsz)
        test_loader = Data.DataLoader(
            dataset=test_dataset,  
            batch_size=bs,       
            shuffle=False,     
            num_workers=3,
            drop_last=False 
        )

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

                # out1 = F.normalize(out1, p=2, dim=1)
                # out2 = F.normalize(out2, p=2, dim=1)

                similarity = torch.cosine_similarity(out1, out2)
                pred_list += similarity.cpu().numpy().tolist()

        d = {'imageA': data['imageA'], 'imageB': data['imageB'], 'prediction': pred_list}
        df = pd.DataFrame(d)
        df.to_csv(save_csv_path, index=None)