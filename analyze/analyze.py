import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

############ read arguments #######################
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(
    label_file,
    patch_file,
    trained_folder,
    checkpoint):

    root_folder = Path().absolute().parent
    trained_folder = Path(trained_folder)
    out_folder = trained_folder.relative_to(root_folder)
    out_file = Path(checkpoint).stem + '_' + Path(label_file).name
    
    print(f'outfolder: {str(out_folder)}')
    
    if (out_folder / out_file).exists():
        print('output file already exists')
        exit()
    
    if not out_folder.exists():
        out_folder.mkdir(parents=True)
    
    #### read arguments pretrained model ##############
    
    with open(trained_folder / 'par.yml', 'r') as ff:
        par = ff.read()
        trained_args = yaml.safe_load(par)
    breaks = trained_args['data']['general']['breaks']
    trained_args['model']['size']['output_size'] = len(breaks) - 1
    trained_args['model']['size']['cat_size'] = \
        len(trained_args['data']['general']['cat_input'])
    
    sys.path.append(str(trained_folder))
    print(trained_folder)
    
    import sequence as seq
    import model as mdl
    
    net = mdl.Model(trained_args['model'])
    net.to(device)
    net.load_state_dict(
        torch.load(
            str(trained_folder / checkpoint),
            map_location=device
        )
    )
    
    seq_args = {
        'label_file': label_file,
        'weight_dict': {0: 1.0, 1: 1.0},
        'training': False
        
    }
    
    if patch_file:
        trained_args['data']['general']['patch_file'] = \
            patch_file
    
    test_args = {**trained_args['data']['general'], **seq_args}
    print('using args:')
    print(test_args)
    testset = seq.CustomDataset(test_args, with_label=False)
    
    result_df = pd.DataFrame(
        columns=['identifier'] + [str(ii) for ii in breaks[1:]]
    )
    
    net.eval()
    
    with torch.no_grad():
        for idx, pid in enumerate(testset.identifiers):
            input_dict = testset.get_data(idx)
            ct = torch.from_numpy(input_dict['ct'][np.newaxis])
            pt = torch.from_numpy(input_dict['pt'][np.newaxis])
            cat = torch.from_numpy(input_dict['cat'][np.newaxis])
            pred = net(ct, pt, cat).numpy()
            print(pid, pred)
            
            result_df.loc[idx] = [pid, *pred[0]]
    
    label_df = pd.read_csv(
        label_file
    )
    
    result_df = result_df.merge(label_df, on='identifier', how='left')
    result_df.to_csv(out_folder / out_file, index=False)

if __name__ == '__main__':
    try:
        par_file = str(sys.argv[1])
    except:
        print('no parameter file specified')
        exit()
    
    print('using parameter file: ' + par_file)
    
    with open(par_file, 'r') as ff:
        par = ff.read()
        data_args = yaml.safe_load(par)

    label_file = data_args['label_file']
    trained_folder = data_args['trained_folder']

    patch_file = data_args['patch_file']
    checkpoint = data_args['checkpoint']
    
    predict(
        label_file=label_file,
        patch_file=patch_file,
        trained_folder=trained_folder,
        checkpoint=checkpoint
    )
