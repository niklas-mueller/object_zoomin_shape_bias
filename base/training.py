import time
import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.default_collate(batch)


def _get_device(model):
    if hasattr(model, 'src_device_obj'):
        device = model.src_device_obj
    elif hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device

def evaluate_objects(loader, model:nn.DataParallel, criterion, device=None, verbose:bool=False):
    num_correct = 0
    num_samples = 0
    losses = {}
    model.eval()

    if device is None:
        device = _get_device(model=model)
    
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), desc='0/0') as t:
            for i, (item) in t:
                x, y = item[0], item[1]
                # if len(item) > 2:
                #     z = item[2]
                x = x.to(device=device)
                if type(y) is list and len(y) == 2:
                    y = (y[0].to(device), y[1].to(device))
                else:
                    y = y.to(device)
                # y = y.to(device=device)
                
                scores = model(x)
                loss = criterion(scores, y)
                
                if (type(y) is tuple or type(y) is list) and len(y) == 2:
                    y = y[0]
                
                losses[i] = loss.cpu().tolist()
                
                if (type(scores) is tuple or type(scores) is list) and len(scores) == 2:
                    scores = scores[0]
                
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                t.set_description(f'{num_correct}/{num_samples} correct ')
        
        accuracy = float(num_correct)/float(num_samples)
        if verbose:
            print(f'Got {num_correct} \t/ {num_samples} correct -> accuracy {accuracy*100:.2f} %') 
    
    model.train()

    return {'accuracy': accuracy, 'batch_losses': losses}


def train_objects(model, trainloader, valloader, loss_fn, optimizer, device, 
            n_epochs:'int|list', lr_scheduler=None, plateau_lr_scheduler=None, 
            result_manager=None, verbose:bool=True, 
            testloader=None, results:dict={}, current_time:str=time.ctime().replace(' ', '_').replace(':', ''), save_per_epoch:bool=False):
    

    model.train(True)
    loss = None
    best_model_state_dict = model.state_dict()
    
    for key in ['training_losses', 'validation_losses', 'epoch_times']:
        if key not in list(results.keys()):
            results[key] = []

    best_valid_loss = np.inf if len(results['validation_losses']) == 0 else min(results['validation_losses'])
    best_training_loss = np.inf if len(results['training_losses']) == 0 else min(results['training_losses'])
    
    epochs = range(n_epochs) if type(n_epochs) is int else n_epochs

    for epoch in tqdm(epochs):
        if verbose:
            print(f"Running epoch {epoch}")

        start_time_epoch = time.time()

        # Training loop
        for item in tqdm(trainloader):
            inputs, labels = item[0], item[1]
            if len(item) > 2:
                item_names = item[2]
            # Move data to device
            inputs = inputs.to(device) #, dtype=torch.float32
            if type(labels) is list and len(labels) == 2:
                labels = (labels[0].to(device), labels[1].to(device))
            else:
                labels = labels.to(device) #, dtype=torch.float32

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            
            loss.backward()

            # Adjust learning weights
            optimizer.step()


        # Get time needed for epoch
        end_time_epoch = time.time()
        results['epoch_times'].append(end_time_epoch - start_time_epoch)
        
        results['training_losses'].append(loss.item())
        
        # Check if overall training loss has improve and if then save the model
        if loss.item() < best_training_loss:
            best_training_loss = loss.item()
            if verbose:
                print(f"Found new best model: Saving model in epoch {epoch} with loss {loss.item()}.")
                if np.isnan(loss.item()):
                    print('Exiting because loss is NaN')
                    return

            # Save model
            if result_manager is not None:
                result_manager.save_model(model, filename=f'best_model_{current_time}.pth', overwrite=True)

            best_model_state_dict = model.state_dict()
        if verbose:
            print(f"Loss after epoch {epoch}: {loss.item()}")

        # Evaluate on validation set
        eval = evaluate(loader=valloader, model=model, criterion=loss_fn, device=device, verbose=False)

        # Compute average over batches
        validation_loss = np.mean([batch_losses for _, batch_losses in eval['batch_losses'].items()])
        if verbose:
            print(f"Validation loss after epoch {epoch}: {float(validation_loss)}")

        results['validation_losses'].append(float(validation_loss))
        results[f'validation_during_training_epoch-{epoch}'] = eval

        # if plateau learning rate scheduler is given, make step depending on average validation loss
        if plateau_lr_scheduler is not None:
            plateau_lr_scheduler.step(float(validation_loss))

        # Check if validation loss has improved and if then store evaluation results
        if validation_loss < best_valid_loss:
            best_valid_loss = validation_loss
            

        # If learning rate scheduler is given, make step
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save losses, epochs times and current model at the end of each epoch
        if result_manager is not None:
            result_manager.save_result(results, filename=f'training_results_{current_time}.yml', overwrite=True)
            result_manager.save_model(model, filename=f'final_model_{current_time}.pth', overwrite=True)
            print(f'Saved Results and Model after epoch {epoch}')

            if save_per_epoch:
                result_manager.save_result(results, filename=f'training_results_{current_time}_epoch_{epoch}.yml', overwrite=True)
                result_manager.save_model(model, filename=f'final_model_{current_time}_epoch_{epoch}.pth', overwrite=True)


    model.train(False)

    # Load best model
    model.load_state_dict(best_model_state_dict)
    
    
    if verbose:
        print("\n\n-----------------------------\n\n")
        if loss is not None:
            print(f'Finished Training with loss: {loss.item()}')

        # Output average time per epoch
        print(f'Average time per epoch for {n_epochs} epochs: {np.mean(results["epoch_times"])}')

        # Evaluate on training model to get first indication whether training works
        training_eval = evaluate(loader=trainloader, model=model, criterion=loss_fn, device=device, verbose=True)
        results['eval_trained_traindata'] = training_eval

        print(f"Evaluated TRAINING data: Accuracy: {training_eval['accuracy']}")

        # Evaluate on validation set
        # responses, val_eval = record_responses(loader=valloader, models=[(f'best_model_{current_time}', model)], device=device, criterion=loss_fn, verbose=True)
        # result_manager.save_result(responses, filename=f'validation_responses_{current_time}.pkl')
        val_eval = evaluate(loader=valloader, model=model, criterion=loss_fn, device=device, verbose=True)
        results['eval_trained_valdata'] = val_eval# [f'best_model_{current_time}']

        print(f"Evaluated VALIDATION data: Accuracy: {val_eval}") #[f'best_model_{current_time}']['accuracy']}")

        # result_manager.save_result(res_per_model[f'best_model_{current_time}'], filename=f'validation_responses_{current_time}.pkl')


    # Evaluate on test dataset
    if testloader is not None:
        eval = evaluate(loader=testloader, model=model, criterion=loss_fn, device=device, verbose=True)
        # responses, eval = record_responses(loader=testloader, models=[(f'best_model_{current_time}', model)], device=device, criterion=loss_fn, verbose=True)
        # result_manager.save_result(responses, filename=f'test_responses_{current_time}.pkl')
        results['eval_trained_testdata'] = eval #[f'best_model_{current_time}']

        if verbose:
            print(f"Evaluated TEST data: Accuracy: {eval}") #[f'best_model_{current_time}']['accuracy']}")

    # Save all results that have been accumulated
    if result_manager is not None:
        result_manager.save_result(results, filename=f'training_results_{current_time}.yml', overwrite=True)

        if verbose:
            print(f"Saved results and model state.")

    if verbose:
        print(f"Done!")

    return results

def train_coco_scenes(model, trainloader, 
           args,
           optimizer, device, 
            n_epochs:'int|list', lr_scheduler=None, plateau_lr_scheduler=None, result_manager=None, verbose:bool=True,
            results:dict={}, current_time:str=time.ctime().replace(' ', '_'), save_per_epoch:bool=False):
    
    
    model = model.to(device)
    rank = 0

    model.train(True)
    loss = None
    best_model_state_dict = model.state_dict()
    
    for key in ['training_losses', 'validation_losses', 'epoch_times']:
        if key not in list(results.keys()):
            results[key] = []

    best_valid_loss = np.inf if len(results['validation_losses']) == 0 else min(results['validation_losses'])
    best_training_loss = np.inf if len(results['training_losses']) == 0 else min(results['training_losses'])
    
    epochs = range(n_epochs) if type(n_epochs) is int else n_epochs

    for epoch in tqdm(epochs):

        trainloader.sampler.set_epoch(epoch)

        if verbose:
            print(f"Running epoch {epoch}")

        start_time_epoch = time.time()

        # Training loop
        for imgs, annotations in tqdm(trainloader):
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            loss_dict = model(imgs, annotations)
            loss = sum(loss for loss in loss_dict.values())

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

        if rank == 0:
            # Get time needed for epoch
            end_time_epoch = time.time()
            results['epoch_times'].append(end_time_epoch - start_time_epoch)
            
            results['training_losses'].append(loss.item())
            
            # Check if overall training loss has improve and if then save the model
            if loss.item() < best_training_loss:
                best_training_loss = loss.item()
                if verbose:
                    print(f"Found new best model: Saving model in epoch {epoch} with loss {loss.item()}.")
                    if np.isnan(loss.item()):
                        print('Exiting because loss is NaN')
                        return

                # Save model
                if result_manager is not None:
                    result_manager.save_model(model, filename=f'best_model_{current_time}.pth', overwrite=True)

                best_model_state_dict = model.state_dict()
            if verbose:
                print(f"Loss after epoch {epoch}: {loss.item()}")

            # # Evaluate on validation set
            # eval = _evaluate(loader=valloader, model=model, criterion=loss_fn, device=device, verbose=False)

            # # Compute average over batches
            # validation_loss = np.mean([batch_losses for _, batch_losses in eval['batch_losses'].items()])
            # if verbose:
            #     print(f"Validation loss after epoch {epoch}: {float(validation_loss)}")

            # results['validation_losses'].append(float(validation_loss))
            # results[f'validation_during_training_epoch-{epoch}'] = eval

            # # if plateau learning rate scheduler is given, make step depending on average validation loss
            # if plateau_lr_scheduler is not None:
            #     plateau_lr_scheduler.step(float(validation_loss))

            # # Check if validation loss has improved and if then store evaluation results
            # if validation_loss < best_valid_loss:
            #     best_valid_loss = validation_loss
                

            # If learning rate scheduler is given, make step
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Save losses, epochs times and current model at the end of each epoch
            if result_manager is not None:
                result_manager.save_result(results, filename=f'training_results_{current_time}.yml', overwrite=True)
                result_manager.save_model(model, filename=f'final_model_{current_time}.pth', overwrite=True)
                print(f'Saved Results and Model after epoch {epoch}')

                if save_per_epoch:
                    result_manager.save_result(results, filename=f'training_results_{current_time}_epoch_{epoch}.yml', overwrite=True)
                    result_manager.save_model(model, filename=f'final_model_{current_time}_epoch_{epoch}.pth', overwrite=True)

    if rank == 0:
        model.train(False)

        # Load best model
        model.load_state_dict(best_model_state_dict)
        
        
        if verbose:
            print("\n\n-----------------------------\n\n")
            if loss is not None:
                print(f'Finished Training with loss: {loss.item()}')

            # Output average time per epoch
            print(f'Average time per epoch for {n_epochs} epochs: {np.mean(results["epoch_times"])}')

            # # Evaluate on training model to get first indication whether training works
            # training_eval = _evaluate(loader=trainloader, model=model, criterion=loss_fn, device=device, verbose=True)
            # results['eval_trained_traindata'] = training_eval

            # print(f"Evaluated TRAINING data: Accuracy: {training_eval['accuracy']}")

            # # # Evaluate on validation set
            # # responses, val_eval = record_responses(loader=valloader, models=[(f'best_model_{current_time}', model)], device=device, criterion=loss_fn, verbose=True)
            # # result_manager.save_result(responses, filename=f'validation_responses_{current_time}.pkl')
            # val_eval = _evaluate(loader=valloader, model=model, criterion=loss_fn, device=device, verbose=True)
            # results['eval_trained_valdata'] = val_eval[f'best_model_{current_time}']

            # print(f"Evaluated VALIDATION data: Accuracy: {val_eval[f'best_model_{current_time}']['accuracy']}")

            # # result_manager.save_result(res_per_model[f'best_model_{current_time}'], filename=f'validation_responses_{current_time}.pkl')


        # # Evaluate on test dataset
        # if testloader is not None:
        #     # eval = _evaluate(loader=testloader, model=model, criterion=loss_fn, device=device, verbose=True)
        #     # responses, eval = record_responses(loader=testloader, models=[(f'best_model_{current_time}', model)], device=device, criterion=loss_fn, verbose=True)
        #     # result_manager.save_result(responses, filename=f'test_responses_{current_time}.pkl')
        #     results['eval_trained_testdata'] = eval[f'best_model_{current_time}']

        #     if verbose:
        #         print(f"Evaluated TEST data: Accuracy: {eval[f'best_model_{current_time}']['accuracy']}")

        # Save all results that have been accumulated
        if result_manager is not None:
            result_manager.save_result(results, filename=f'training_results_{current_time}.yml', overwrite=True)

            if verbose:
                print(f"Saved results and model state.")

        if verbose:
            print(f"Done!")

        return results

