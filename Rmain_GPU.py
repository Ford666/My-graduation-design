#from advanced_model import U_net
from unet1 import unet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
#from PIL import Image
import scipy.misc

def save_img(img, im_name, save_folder_name):
    img_np = img.data.cpu().numpy() #torch.Variable stored in GPU => ndarray
    img_np = np.array(img_np, np.float32)
 
    #if normal:  #normalize output image
    #    img_max = np.max(img_np)
    #    img_min = np.min(img_np)
    #    img_cont = (img_np-img_min)/(img_max-img_min)
    export_name = save_folder_name + str(im_name) + '.png'    
    
    #img_pl = Image.fromarray(img_np)
    #if img_pl.mode != 'L':
    #    img_pl = img_pl.convert('L')
    #img_pl = img_pl*255
    #img_pl.save(export_name)
    scipy.misc.imsave(export_name, img_np)

class NPCCloss(nn.Module):

    def __init__(self):
        super(NPCCloss, self).__init__()

    def forward(self, x, y):
        x = x.view(256**2,-1)
        y = y.view( 256**2,-1)
        x_mean = torch.mean(x, dim=0, keepdim=True)
        y_mean = torch.mean(y, dim=0, keepdim=True)
        vx = (x-x_mean)
        vy = (y-y_mean)
        c = torch.mean(vx*vy, dim=0)/(torch.sqrt(torch.mean(vx**2, dim=0)+1e-08) * torch.sqrt(torch.mean(vy ** 2,dim=0)+1e-08))
        output = torch.mean(-c) # torch.mean(1-c**2)
        return output

if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = open('./Rloss_history.txt','w+')
    #Load the preprocessed MNIST dataset

    X_train = np.load('./MNIST/train/train_speckle.npy')
    X_train = Variable(torch.from_numpy(X_train).float(),requires_grad=False)
    y_train = np.load('./MNIST/train/train_image.npy')
    y_train = Variable(torch.from_numpy(y_train).float(), requires_grad=False)

    X_test = (np.load('./MNIST/test/test_speckle.npy'))
    X_test = Variable(torch.from_numpy(X_test).float(),requires_grad=False)
    y_test = np.load('./MNIST/test/test_image.npy')
    y_test = Variable(torch.from_numpy(y_test).float(), requires_grad=False)

    # Subsample the data
    train_size = 10000
    validation_size = 200
    # test_size = 20
    test_size = X_test.shape[0]

    mask = list(range(train_size, train_size + validation_size))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(train_size))
    X_train = X_train[mask]
    y_train = y_train[mask]  
    mask = list(range(test_size))
    X_test = X_test[mask]
    y_test = y_test[mask]

    #print the data shape
    MNIST_data = {
          'X_train: ': X_train, 'y_train: ': y_train,
          'X_val: ': X_val, 'y_val: ': y_val,
          'X_test: ': X_test, 'y_test: ': y_test,
        }
    for k, v in list(MNIST_data.items()):
        print(('%s: ' % k, v.shape))
        f.write(str(k) + str(v.shape) +'\n')
        f.flush()

    # Use TensorDaraset directly to wrap the data into Dataset class
    MNIST_train = TensorDataset(X_train, y_train)
    MNIST_val = TensorDataset(X_val, y_val)
    MNIST_test = TensorDataset(X_test, y_test)

    # Dataloader begins
    batch_size = 1
    train_loader = DataLoader(dataset=MNIST_train, batch_size=batch_size, 
                                shuffle=True, num_workers=batch_size)
    val_loader = DataLoader(dataset=MNIST_val, batch_size=batch_size, 
                                shuffle=True, num_workers=batch_size)
    test_batch_size = 1
    test_loader = DataLoader(dataset=MNIST_test, batch_size=test_batch_size, 
                                shuffle=True, num_workers=test_batch_size)

    # Model
    #model = U_net(in_channels=1, out_channels=1).cuda()
    model = unet().cuda()

    #model.apply(weights_init) 

    # Optimization
    # criterion = nn.MSELoss().cuda()
    criterion = NPCCloss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    lr_schduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #Train
    n_epochs = 20
    iter_per_epoch = int(train_size/batch_size)
    print("Initializing Training!")
    f.write("Initializing Training!\n")
    f.flush()

    best_epoch = 0
    min_loss = 0
    trainout_dir = './MNIST/train/epoch'
    for epoch in range(n_epochs):
        lr_schduler.step() # Adjust L_r every 10 epoch

        model.train()    
        train_loss = 0 

        # mini-batch training using torch
        save_dir = trainout_dir + str(epoch+1) +'/train_out'
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs) #(batch_size,1,256,256)
            outputs = torch.squeeze(outputs) #(256,256)
            labels = labels.unsqueeze(1) #(batch_size,32,32) => (batch_size,1,32,32)
            labels = nn.UpsamplingNearest2d(scale_factor=8)(labels) #(batch_size,1,256,256)
            labels = torch.squeeze(labels) #(256,256)

            loss = criterion(outputs,labels) 
            train_loss += loss.cpu().item()

            #Update weights
            loss.backward()           
            optimizer.step()

            print('Epochs: ', str(epoch+1), '| iterations/iter_per_epoch:', str(i+1),'/',
                    str(iter_per_epoch),' ==>', 'Train loss:', train_loss/(i+ 1))
            
            if (i+1) % 1000 == 0:  
                save_img(outputs, i+1, save_dir)

            if (i+1) % 100 == 0:
                f.write('Epochs: ' + str(epoch+1) + '\t' + 'iterations/iter_per_epoch:' + str(i+1) + '/'
                   + str(iter_per_epoch) + '\t' + 'Train loss:' + str(train_loss/(i+ 1)) + '\n')
                f.flush()
                
        #Validation every 1 epochs
        if (epoch+1) % 1 == 0:
            print('A validation every 1 epochs!')
            f.write('A validation every 1 epochs!\n')
            f.flush()

            # mini-batch validating using torch
            val_loss = 0
            for i, (inputs,labels) in enumerate(val_loader):
                with torch.no_grad():
                    inputs,labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs) #(batch_size,1,256,256)
                    outputs = torch.squeeze(outputs) #(256,256)
                    labels = labels.unsqueeze(1) #(batch_size,32,32) => (batch_size,1,32,32)
                    labels = nn.UpsamplingNearest2d(scale_factor=8)(labels) #(batch_size,1,256,256)
                    labels = torch.squeeze(labels) #(256,256)
                    val_loss += criterion(outputs, labels).cpu().item()    
                    
            val_loss = val_loss/(i+1)

            print('Val loss:', val_loss)
            f.write('Val loss: '+ str(val_loss) + '\n')
            f.flush()

            # save the best model with minimal validation loss
            if val_loss < min_loss:
                best_epoch = epoch
                min_loss = val_loss
                torch.save(model.state_dict(), './Rmodel_epoch_{0}.pkl'.format(best_epoch+1))   
                print('model{} saved!'.format(best_epoch+1))
                f.write('model' + str(best_epoch+1) + 'saved!' + '\n')
                f.flush()
             
    #Test   
    print('Use Rmodel_epoch_', str(best_epoch+1), '.pkl', ' to generate test prediction!')
    f.write('Use Rmodel_epoch_' + str(best_epoch+1) + '.pkl' + ' to generate test prediction!' + '\n')
    f.flush()
    model_path = './Rmodel_epoch_'+str(best_epoch+1)+'.pkl'
    model.load_state_dict(torch.load(model_path))

    model.eval()
    # mini-batch test using torch
    test_loss = 0
    stacked_pre = torch.Tensor([]).cuda()
    stacked_tar = torch.Tensor([]).cuda()
    for i, (inputs,labels) in enumerate(test_loader):
        with torch.no_grad():
            inputs,labels = inputs.cuda(),labels.cuda()
            outputs = model(inputs)
            outputs = torch.squeeze(outputs) #(256,256)
            labels = labels.unsqueeze(1) #(batch_size,32,32) => (batch_size,1,32,32)
            labels = nn.UpsamplingNearest2d(scale_factor=8)(labels) #(batch_size,1,256,256)
            labels = torch.squeeze(labels) #(256,256)

            test_loss += criterion(outputs, labels)  
            
        save_img(outputs, i+1 , './MNIST/test/predict') 
        save_img(labels, i+1, './MNIST/test/target') 
        outputs = outputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        stacked_pre = torch.cat((stacked_pre, outputs),0) 
        stacked_tar = torch.cat((stacked_tar, labels),0)
             
    test_loss = test_loss.cpu().item()/(i+1)

    stacked_pre = stacked_pre.data.cpu().numpy() #prediction (test_size,256,256)
    stacked_tar = stacked_tar.data.cpu().numpy() #target (test_size,256,256)
    print('stacked_pre: ', stacked_pre.shape, '\t', 'stacked_tar: ', stacked_tar.shape)
    f.write('stacked_pre: ' + str(stacked_pre.shape) + '\t' + 'stacked_tar: ' + str(stacked_tar.shape) + '\n')
    f.flush()
   
    np.save('./MNIST/test/stacked_predict.npy',stacked_pre)    
    np.save('./MNIST/test/stacked_target.npy',stacked_tar)

    print('Test_loss: {:.4f}'.format(test_loss))
    f.write('Test_loss: ' + str(test_loss) + '\n')
    f.flush()
               
    print("Finish Prediction!")
    f.write("Finish Prediction!\n")
    f.flush()
    f.close()
   










