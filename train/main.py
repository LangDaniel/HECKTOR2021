import model as mdl
import sequence as seq

import argparse
import sys
from pathlib import Path

from lifelines.utils import concordance_index
from sklearn import metrics
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

def prepare_output(folder, sub, copy_dict={}):
    ''' generate a subfolder run_# in args['folder'] for the output
    and copy all files there '''

    folder = Path(folder)

    if not folder.exists():
        folder.mkdir(parents=True)
        count = 0 
    else:
        all_folder = [sf.name for sf in folder.iterdir() if
                        sf.is_dir() and sub in sf.name]
        all_counts = [int(sf[len(sub) + 1: ]) for sf in all_folder]
        count = max(all_counts) + 1 

    print('run # {}'.format(count))

    subfolder = folder / (sub + '_' + str(count))
    subfolder.mkdir()

    for key, value in copy_dict.items():
        path = subfolder / key 
        with open(path, 'w') as ff: 
            ff.write(value)
        path.chmod(0o444)

    return subfolder

def get_param_files(path):
    '''
    returns a list of all the parameter files
    with could be found at 'path'
    '''
    path = Path(path)
    if path.is_dir():
        files = sorted([ff for ff in path.iterdir() if ff.suffix == '.yml'])
    elif path.suffix == '.yml':
        files = [path]
    else:
        raise ValueError('non valid parameter files (must be folder or .yml)')
    return files

def train(args, out_folder, device):
    ''' main training function '''

    # fix the seed for reproducibility
    seed = 42 
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_args = {**args['data']['general'], **args['data']['train']}
    valid_args = {**args['data']['general'], **args['data']['valid']}

    trainset = seq.CustomDataset(train_args)
    validset = seq.CustomDataset(valid_args)

    breaks = trainset.get_breaks().astype(float)

    args['model']['size']['output_size'] = \
        len(breaks) - 1
    args['model']['size']['cat_size'] = \
        len(args['data']['general']['cat_input'])
    net = mdl.Model(args['model'])
    
    optimizer = net.get_optimizer()
    criterion = net.get_loss()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_args['batch_size'],
        shuffle=True,
        num_workers=args['training']['workers']
    )
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=valid_args['batch_size'],
        shuffle=False,
        num_workers=args['training']['workers']
    )

    net.to(device)

    train_writer = SummaryWriter(str(Path(out_folder) / 'train'))
    valid_writer = SummaryWriter(str(Path(out_folder) / 'valid'))

    def get_surv_time(breaks, surv_prob):
        breaks = np.array(breaks).astype(float)
        surv_prob = np.array(surv_prob).astype(float)
    
        mu = 0 
        for ii in range(1, len(breaks)):
            mu += np.prod(surv_prob[:ii]) * breaks[ii]
    
        return mu

    best_loss = False
    best_cindex = False

    for epoch in range(args['training']['epochs']):
        print(f'epoch: {epoch}', end='  ')
    
        ############################ training ##########################
        net.train()
        running_loss = 0.0
        epoch_score = []
        epoch_time = []
        epoch_event = []

        for data in trainloader:
            
            ct = data['ct'].to(device)
            pt = data['pt'].to(device)
            cat = data['cat'].to(device)
            label = data['label'].to(device)
            raw_label = data['raw_label'].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(ct, pt, cat).squeeze()
            loss = criterion(y_pred=outputs, y_true=label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            for out in outputs.detach().cpu().numpy():
                epoch_score += [get_surv_time(breaks, out)]
            epoch_time += list(raw_label[:, 0])
            epoch_event += list(raw_label[:, 1])
    
        train_loss = running_loss/len(trainloader)
        train_cindex = concordance_index(epoch_time, epoch_score, epoch_event)

        train_writer.add_scalar('epoch_loss', train_loss, epoch)
        train_writer.add_scalar('epoch_cindex', train_cindex, epoch)
    
        print(f'train loss: {train_loss:.4} train cindex: {train_cindex:.4}', end='  ')
        
        ############################ validation ########################
        
        net.eval()
        running_loss = 0.0
        epoch_score = []
        epoch_time = []
        epoch_event = []

        with torch.no_grad():
            for data in validloader:
                
                ct = data['ct'].to(device)
                pt = data['pt'].to(device)
                cat = data['cat'].to(device)
                label = data['label'].to(device)
                raw_label = data['raw_label'].to(device)

                outputs = net(ct, pt, cat)
                running_loss += torch.mean(criterion(y_pred=outputs.squeeze(), y_true=label)).item()

                for out in outputs.detach().cpu().numpy():
                    epoch_score += [get_surv_time(breaks, out)]
                epoch_time += list(raw_label[:, 0])
                epoch_event += list(raw_label[:, 1])
    
        val_loss = running_loss / len(validloader)
        val_cindex = concordance_index(epoch_time, epoch_score, epoch_event)

        valid_writer.add_scalar('epoch_loss', val_loss, epoch)
        valid_writer.add_scalar('epoch_cindex', val_cindex, epoch)

        print(f'val loss: {val_loss:.4}, val cindex {val_cindex:.4}', end='\n')
        
        ## save best model ##
        if not epoch or val_loss < best_loss:
            torch.save(net.state_dict(), str(Path(out_folder) / 'best_loss.pth'))
            best_loss = val_loss
        if not epoch or val_cindex > best_cindex:
            torch.save(net.state_dict(), str(Path(out_folder) / 'best_cindex.pth'))
            best_cindex = val_cindex
    
    train_writer.close()
    valid_writer.close()
    print('Finished Training')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('pdir', help='parameter directory')
    parser.add_argument('--gpu', help='choose gpu [0, 1]')
    pargs = parser.parse_args()
    
    if pargs.gpu:
        gpu = 'cuda:'+pargs.gpu 
    else:
        gpu = 'cuda:0'

    par_path = pargs.pdir

    copy_dict = {}
    for ff in [__file__, seq.__file__, mdl.__file__]:
        path = Path(ff)
        copy_dict[path.name] = path.read_text()

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    for pf in get_param_files(par_path):
        print('processing: '+str(pf))

        with open(str(pf), 'r') as ff:
            par = ff.read()
            args = yaml.safe_load(par)

        copy_dict['par.yml'] = par

        out_folder = prepare_output(
            args['output']['folder'],
            args['output']['sub'],
            copy_dict
        ) 

        try:
            train(args, out_folder, device)
        except (Exception, ArithmeticError) as e:
            print(f'failed {pf}: {e}')
