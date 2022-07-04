import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset


def create_loader(x,y=None):
# https://stackoverflow.com/a/44475689
    tensor_x = torch.Tensor(x).unsqueeze(0) # transform to torch tensor
    if y is None:
        tensor_y = torch.Tensor(np.zeros(tensor_x.shape[0]))
    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset) # create your dataloader

    return my_dataloader

def get_prediction(model,loader, device, tries = 2):
    for i in range(tries):
        try:
            model.eval()
            with torch.no_grad():
                for x,y in loader:
                    try:
                        score = model(x)
                        # print(score.shape)
                        # predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in score]).to(device='cuda')
                        prediction = torch.tensor(torch.argmax(score,dim=1)).to(device=device)
                    except RuntimeError as e:
                        prediction =  torch.ones(y.shape, dtype=torch.float64)*-1
                        score =  torch.ones(y.shape, dtype=torch.float64)*-1
        except RuntimeError as e:
            prediction = torch.ones([1])*-1
            score = torch.ones([1])*-1

           
    return prediction, score

def save_results(prediction:int, scores:list, dt_center:datetime, area_file, model_type = 'trigger'):
    """saves prediction and scores around a particular date time to a csv file.

    Args:
        prediction (_type_): model prediction. 0 or 1
        scores (_type_): Probability of each prediction.
        dt_center (_type_): date time center for prediction
        area_file (_type_): are specific file to store results
        model_type (str, optional): Denotes the model used to get the results.
            Options: 'trigger' or 'detector'. Defaults to 'trigger'.

    Returns:
        str: file path where results where saved.
    """
    scores = scores.numpy()[0]
    if area_file.is_file() is False:
        results = pd.DataFrame(columns = ['date',f'prediction_{model_type}',f'score_0_{model_type}',f'score_1_{model_type}'])
        # results.iloc[dt_center.strftime('%Y%m%d-%H'),:]= prediction, scores
        # results[dt_center.strftime('%Y%m%d-%H')]= scores
    else:
        results = pd.read_csv(area_file)
    results = results.append(pd.DataFrame({'date':[dt_center.strftime('%Y%m%d-%H')],
                        f'prediction_{model_type}':[prediction],
                        f'score_0_{model_type}':[scores[0]],
                        f'score_1_{model_type}':[scores[1]]}), ignore_index = True)

    results = results.drop_duplicates(subset=('date'), keep = 'last')
    results = results.sort_values(by='date')
    results.to_csv(area_file,index = False)
    return area_file
    

def save_prediction(prediction,dt_center,area_file):
    # Read JSON file
    if area_file.is_file() is False:
        with open(area_file, 'w') as json_file:
            json.dump({'predictions':{dt_center.strftime('%Y%m%d-%H'):prediction}}, 
                                json_file, 
                                indent=4,  
                                separators=(',',': '))
    with open(area_file) as json_file:
        predictions_dic = json.load(json_file)
    
    predictions_dic['predictions'].update({dt_center.strftime('%Y%m%d-%H'):prediction})

    with open(area_file, 'w') as json_file:
        json.dump(predictions_dic, json_file, 
                            indent=4,  
                            separators=(',',': '))
    return area_file
