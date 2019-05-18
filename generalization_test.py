from unet1 import unet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import scipy.misc

def save_img(img, im_name, save_folder_name):
    img_np = img.data.cpu().numpy() #torch.Variable stored in GPU => ndarray
    img_np = np.array(img_np, np.float32)

    export_name = save_folder_name + str(im_name) + '.png'    
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

    X_test = (np.load('./test/test_speckle.npy'))
    X_test = Variable(torch.from_numpy(X_test).float(),requires_grad=False)
    y_test = np.load('./test/test_image.npy')
    y_test = Variable(torch.from_numpy(y_test).float(), requires_grad=False)

    test_size = X_test.shape[0]

    # Use TensorDaraset directly to wrap the data into Dataset class

    MNIST_test = TensorDataset(X_test, y_test)

    # Dataloader begins
    test_batch_size = 1
    test_loader = DataLoader(dataset=MNIST_test, batch_size=test_batch_size, 
                                shuffle=True, num_workers=test_batch_size)

    # Model
    model = unet().cuda()
    criterion = NPCCloss().cuda()
             
    #Test   
    best_epoch = 20
    model_path = './test/Rmodel_epoch_'+str(best_epoch)+'.pkl'
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
            
        save_img(outputs, i+1 , './test/predict') 
        save_img(labels, i+1, './test/target') 
        outputs = outputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        stacked_pre = torch.cat((stacked_pre, outputs),0) 
        stacked_tar = torch.cat((stacked_tar, labels),0)
             
    test_loss = test_loss.cpu().item()/(i+1)

    stacked_pre = stacked_pre.data.cpu().numpy() #prediction (test_size,256,256)
    stacked_tar = stacked_tar.data.cpu().numpy() #target (test_size,256,256)
    print('stacked_pre: ', stacked_pre.shape, '\t', 'stacked_tar: ', stacked_tar.shape)
   
    np.save('./test/stacked_predict.npy',stacked_pre)    
    np.save('./test/stacked_target.npy',stacked_tar)

    print('Test_loss: {:.4f}'.format(test_loss))
               
    print("Finish Prediction!")

   










