import torch
import torch.nn as nn

import numpy as np
import matplotlib.image as mpimg
import pandas as pd


def train_model(model, n_epochs, train_loader, valid_loader,
                device, optimizer, criterion, clip_grad=True,
                best_model_score=None):
    
    model.to(device)
    model.train()
    
    print_every_n = 10
    if best_model_score is None:
        best_model_score = 10 ** 10
    best_model_dict = model.state_dict()
    
    print("Starting  Model training")
    print("_"*50 + "\n")
    
    for epoch in range(n_epochs):
        
        batch_loss = 0
        train_epoch_loss = 0
        
        for batch_i, data in enumerate(train_loader, 1):
            # get the input images and their corresponding labels
            images = data["image"]
            key_pts = data["keypoints"]
            
             # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)
            
            # forward pass to get outputs
            output_pts = model(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # clipping gradients
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            # update the weights    
            optimizer.step()
            
            # Calculalatin batch and epoch loss (training)
            batch_loss += loss.item() * images.size(0)
            train_epoch_loss += loss.item() * images.size(0)
            
            
            if batch_i % print_every_n == 0:    # print every n batches
                print('Epoch: {}, Batch: {}, Avg. Batch Loss: {}'.format(epoch+1,
                                                                   batch_i,
                                                                   batch_loss/print_every_n))
                
                print(40*"-")
                batch_loss = 0.0
                
                model.eval()
                valid_mse = 0
                # valid model on validation loader
                with torch.no_grad():
                    for data in valid_loader:
                        
                        images = data["image"]
                        key_pts = data["keypoints"]
                        
                        # flatten pts 
                        key_pts = key_pts.view(key_pts.size(0), -1)

                        # convert variables to floats for regression loss
                        key_pts = key_pts.type(torch.FloatTensor).to(device)
                        images = images.type(torch.FloatTensor).to(device)
                        
                        # forward pass to get outputs
                        output_pts = model(images)

                        # calculate the loss between predicted and target keypoints
                        loss = criterion(output_pts, key_pts)
                        
                        # adding loss to total loss
                        valid_mse += loss.item() * images.size(0)
                        
                    total_valid_mse = valid_mse / len(valid_loader)
                    
                    if total_valid_mse < best_model_score:
                        print("\n" + "*"*50)
                        print("Total Error in validation set decreased")
                        print(f"New: {total_valid_mse} - Old: {best_model_score}")
                        print(f"Model state saved at - Epoch: {epoch+1} - Batch: {batch_i}")
                        print("*"*50 + "\n")
                        best_model_score = total_valid_mse
                        best_model_dict = model.state_dict()
                        
                model.train()             
                          
        # print epoch loss
        train_epoch_loss = train_epoch_loss / len(train_loader)        
        print(f"Total Training Loss Epoch {epoch+1}: {train_epoch_loss}")
        print("_"*50 + "\n\n")
    print('Finished Training')
    
    return best_model_dict, best_model_score
            

def testAccuracy(model, loader):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)
