import torch
import torchani
from ase.io import read, write
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchani
import torch
import math
import tqdm
from aimnet2calc import AIMNet2Calculator
import numpy as np
import os
from sys import argv
import pickle

train_file = 'train-sd_sal_sel3.hdf5'
max_epochs = 500
valsplit = 0.8
name = f'paramscan-2-testtrain2-{"-".join(argv[1:])}'
best_model_checkpoint = f'best-{"-".join(argv[1:])}.pt'
latest_checkpoint = f'latest-{"-".join(argv[1:])}.pt'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tensorboard = SummaryWriter('runs/{}_{}'.format(name, timestamp))


# use the following to decide which layers to freeze and which to train

update_params = [int(i) for i in argv[1:]]
print('update_params are ', update_params)
force_coefficient = 1.0
charge_coefficient = 0.0
batch_size = 1
                                     

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_file = '/u/jdm/models/aimnet2_b973c_0.jpt'
calc = AIMNet2Calculator(model_file)
model = calc.model


species_order = [b'Si',b'O',b'N',b'C',b'H']
num_species = len(species_order)

isol_es = {1: -12.47161347161026,
 6: -1027.3222172266535,
 7: -1480.8934922979006,
 8: -2038.8725875656698,
 14: -7871.001063761595}

training, validation = torchani.data.load(train_file, additional_properties=('forces','natoms', 'charges'))\
                                        .species_to_indices(species_order)\
                                        .shuffle()\
                                        .split(valsplit, None)

training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()


model.train()
params = list(model.parameters())
for i in range(len(params)):
    params[i].requires_grad = False
for i in update_params:
    params[i].requires_grad = True
    
# initialise weights for specific layer (use this to randomise particular layers before fine-tuning)
# for i in update_params:
#     stdv = 1. / math.sqrt(params[i].size(-1))
#     print(f'stdv is {stdv}')
#     torch.nn.init.uniform_(params[i], -stdv, stdv)
    
AdamW = torch.optim.AdamW([params[i] for i in update_params], lr=1e-3, weight_decay=1e-3) # accept mostly defaults for now
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
    
def validate(train=False):
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    total_me = 0.0
    total_mse_f = 0.0
    total_mse_c = 0.0
    count = 0
    count_f = 0
    count_c = 0
    model.train(False)
    if train:
        prop_list = training
    else:
        prop_list = validation
    for properties in prop_list:
        numbers = properties['numbers'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).float()
        true_forces = properties['forces'].to(device).float()
        true_charges = properties['charges'].to(device).float()
        inp = {'coord': coordinates, 
                'numbers': numbers, 
                'charge': 0}
                # 'mol_idx': np.hstack([np.ones(val)*i for i, val in enumerate(natoms)])}

        data = calc.prepare_input(inp)
        calc.set_grad_tensors(data, forces=True)
        data = model(data)
        predicted_energies = data['energy']
        predicted_charges = data['charges'][:-1].reshape((batch_size, numbers.shape[1], 1))
        pe = predicted_energies
        predicted_forces = -torch.autograd.grad(predicted_energies.sum(), data['coord'],
                                        create_graph=True, retain_graph=True)[0][:-1].reshape((batch_size, numbers.shape[1], 3))
        total_mse_f += mse_sum(predicted_forces, true_forces).item()
        total_mse_c += mse_sum(predicted_charges, true_charges).item()
        total_mse += mse_sum(pe, true_energies).item()
        total_me += (pe - true_energies).mean().item()
        count += predicted_energies.shape[0]
        count_f += np.prod(predicted_forces.shape)
        count_c += np.prod(predicted_charges.shape)
    model.train(True)
    return (math.sqrt(total_mse / count)), (math.sqrt(total_mse_f /count_f)), (math.sqrt(total_mse_c /count_c)), (total_me/count)


mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
early_stopping_learning_rate = 1.0E-5

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse, rmse_f, rmse_c, me = validate()
    rmse_t, rmse_f_t, rmse_c_t, me_t = validate(train=True)
    print('E RMSE (val):', rmse, 'F RMSE:', rmse_f, 'Mean E error:', me, 'C RMSE:', rmse_c, 'at epoch', AdamW_scheduler.last_epoch + 1)
    print('E RMSE (train):', rmse_t, 'F RMSE:', rmse_f_t, 'Mean E error:', me_t, 'C RMSE:', rmse_c_t)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(model.state_dict(), best_model_checkpoint)
        
    AdamW_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse_E', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('validation_rmse_F', rmse_f, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('validation_rmse_C', rmse_c, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse_E', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('training_rmse_E', rmse_t, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('training_rmse_F', rmse_f_t, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('training_rmse_C', rmse_c_t, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse_E', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)
    
    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        numbers = properties['numbers'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        coordinates.requires_grad = True
        true_energies = properties['energies'].to(device).float()
        true_forces = properties['forces'].to(device).float()
        true_charges= properties['charges'].to(device).float()
        num_atoms = (numbers >= 0).sum(dim=1, dtype=true_energies.dtype)
        data = calc.prepare_input({'coord': coordinates, 'numbers': numbers, 'charge': 0})
        calc.set_grad_tensors(data, forces=True)
        data = model(data)
        predicted_energies = data['energy'].to(device).float()
        predicted_charges = data['charges'].to(device).float()[:-1].reshape((batch_size, numbers.shape[1], 1))
        # TODO: fix this janky reshaping
        forces = -torch.autograd.grad(predicted_energies.sum(), data["coord"], create_graph=True, retain_graph=True)[0][:-1].reshape((batch_size, numbers.shape[1], 3))
        
        energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
        charge_loss = (mse(true_charges, predicted_charges ) / num_atoms).mean()
        loss = energy_loss + force_coefficient * force_loss + charge_coefficient * charge_loss

        AdamW.zero_grad()
        loss.backward()
        AdamW.step()
        
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': model.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
    }, latest_checkpoint)
    
    if (_%20 == 0 or _ == 10 or _ == 5 or _ == 1):
        torch.save({
            'nn': model.state_dict(),
            'AdamW': AdamW.state_dict(),
            'AdamW_scheduler': AdamW_scheduler.state_dict(),
        }, f'epoch-{_}--{"-".join(argv[1:])}.pt')