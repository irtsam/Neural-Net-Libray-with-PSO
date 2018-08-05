
import os
import numpy as np
import random
import cv2



path="D:/Study/Portable/NeuralNet_Optimization_PSO/DevanagariHandwrittenCharacterDataset"

class Devan_data_loader:
    def __init__(self,path, data):
        self.data=data
        self.path=path

        if(os.path.isdir(os.path.join(path, 'Dataset'))):
            counter=0
            print(counter)
        else:
            counter=1
            Devan_data_loader.Create_Train_dataset(self.path)
            Devan_data_loader.Create_Test_dataset(self.path)
            
        print(counter)
        
    def getdata(self,dta):
        if(dta=='Train'):
            directt=os.path.join(path, 'Dataset','Train_DataSet')
            Train_dta= np.loadtxt(os.path.join(directt,'Training_Data'))
            Label_dta= np.loadtxt(os.path.join(directt,'Label_data'))
            assert(Train_dta.shape==(1024,78200))
            assert(Label_dta.shape==(46,78200))
            return(Train_dta,Label_dta)
        elif(dta=='Test'):
            directt=os.path.join(path, 'Dataset','TestData')
            Test_dta= np.loadtxt(os.path.join(directt,'Test_data'))
            Label_dta= np.loadtxt(os.path.join(directt,'Label_data'))
            assert(Test_dta.shape==(1024,13800))
            assert(Label_dta.shape==(46,13800))
            return(Test_dta,Label_dta)
            


    
    @staticmethod
    def Create_Train_dataset(path):
        directt=os.path.join(path, 'Dataset','Train_DataSet')
        os.makedirs(directt)
        Train_array=np.zeros([1024, 78200])
        Label_array=np.zeros([46, 78200])
        assert(Train_array.shape==(1024,78200))
        assert(Label_array.shape==(46,78200))
        paths=os.listdir(path)
        if "Train" in paths:
            Tr_pth= os.path.join(path, "Train")
            Training_clases= os.listdir(Tr_pth)
            dict_Train= {}
            for tr in Training_clases:
                dict_Train[os.path.join(Tr_pth,tr)]=os.listdir(os.path.join(Tr_pth,tr))
                
                keyss= list(dict_Train.keys())
                #print(keys)
                appended=0
                for i,kr in enumerate(keyss):
                    val= list(dict_Train[keyss[i]])
                    for vl in val:
                        pthh= os.path.join(kr,vl)
                        x= cv2.imread(pthh,0)
                        x= x.reshape(32*32,)
                        Train_array[:,appended]=x[:]
                        Label_array[i,appended]=1
                        appended=appended+1
        np.savetxt(os.path.join(directt, 'Training_Data'), Train_array)
        np.savetxt(os.path.join(directt, 'Label_data'), Label_array)
        print(appended,i)
        print(len(dict_Train[keyss[0]]))
    
    
    @staticmethod
    def Create_Test_dataset(path):
        """Add support to add labels for tesL"""
        directt=os.path.join(path, 'Dataset','TestData')
        os.makedirs(directt)
        Test_array=np.zeros([1024, 13800])
        Label_array=np.zeros([46, 13800])
        assert(Label_array.shape==(46,13800))
        assert(Test_array.shape==(1024,13800))
        paths=os.listdir(path)
        if "Test" in paths:
            Tr_pth= os.path.join(path, "Test")
            Training_clases= os.listdir(Tr_pth)
            dict_Train= {}
            for tr in Training_clases:
                dict_Train[os.path.join(Tr_pth,tr)]=os.listdir(os.path.join(Tr_pth,tr))
                
                keyss= list(dict_Train.keys())
                #print(keys)
                appended=0
                for i,kr in enumerate(keyss):
                    val= list(dict_Train[keyss[i]])
                    for vl in val:
                        pthh= os.path.join(kr,vl)
                        x= cv2.imread(pthh,0)
                        x= x.reshape(32*32,)
                        Test_array[:,appended]=x[:]
                        Label_array[i,appended]=1
                        appended=appended+1
        np.savetxt(os.path.join(directt, 'Test_data'), Test_array)
        np.savetxt(os.path.join(directt, 'Label_data'), Label_array)
        print(appended,i)
        print(len(dict_Train[keyss[0]]))
    
    

#assert(os.path.isdir(path))
print(os.path.isfile(os.path.join(path, 'Dataset')))
print(os.path.isdir(os.path.join(path, 'Dataset')))
print((os.path.join(path, 'Dataset')))
cl= Devan_data_loader(path,'Test')
cl=Devan_data_loader(path,'Train')
#dataa,labels= cl.getdata('Test')
#dataa,labels= cl.getdata('Train')

        
        
