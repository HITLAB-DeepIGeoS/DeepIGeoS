import sys, json
from PyQt5.QtWidgets import*
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5 import *
from PyQt5 import uic
import os, time
import numpy as np
import cv2
import nibabel as nib
from function_deepigeos import *

import matplotlib.pyplot as plt
import math

ui = r'../DeepIGeoS/ui_deepigeos.ui'

    
class MainDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self, None)
        uic.loadUi(ui, self)
        self.initUI()

        
    def initUI(self):
            
        self.setMouseTracking(True)

        self.pushButton.clicked.connect(self.Button_click)
        self.pushButton_2.clicked.connect(self.Button2_click)
        self.pushButton_3.clicked.connect(self.Button3_click)
        self.pushButton_4.clicked.connect(self.Button4_click)
        self.pushButton_5.clicked.connect(self.Button5_click)
        self.pushButton_6.clicked.connect(self.Button6_click)
        self.pushButton_7.clicked.connect(self.Button7_click)

        
        self.pn = 1
        self.count = 0
        self.ax = 0
        self.img_file_names = ""
        self.seg_file_names = ""
        
        self.usrId = time.time()
        self.usrId = int(time.time()-1495000000)

        
        if not os.path.isdir("../res/%d" % self.usrId):
            os.makedirs('../res/%d' % self.usrId)
            os.makedirs('../res/%d/seg/X' % self.usrId)
            os.makedirs('../res/%d/seg/Y' % self.usrId)
            os.makedirs('../res/%d/seg/Z' % self.usrId)
            os.makedirs('../res/%d/result' % self.usrId)
    
    def Button_click(self):  # get file url and nibabel to numpy
        file_names = QFileDialog.getOpenFileName(self)
        self.img_file_names = file_names
        self.imgs = nib.load(self.img_file_names[0])
        self.imgs = nib.as_closest_canonical(self.imgs).get_fdata()

    def Button6_click(self):   # run p-net 
        # test file paths
        test_images = self.img_file_names[0]
        save_dir = f'../res/{self.usrId}/result'

        # preprocessing
        test_transform = get_transform("valid")

        # device config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # model weight file paths
        pnet_best_ckpt_dir = "./experiments/best_ckpts/brats3d_pnet_init_train"
        pnet_best_ckpt_path = sorted(glob.glob(f"{pnet_best_ckpt_dir}/*.pt"))[-1]

        # load model weights
        pnet = P_RNet3D(c_in=1, c_blk=16, n_classes=2).to(device)
        pnet.load_state_dict(torch.load(pnet_best_ckpt_path))
        pnet.eval()
        
        save_path_pnet = os.path.join(save_dir, "_pred_pnet.nii.gz")

        pnet_pred_labels = pnet_inference(image_path=test_images,
                                          save_path=save_path_pnet,
                                          pnet=pnet,
                                          transform=test_transform,
                                          norm_transform=tio.ZNormalization(masking_method=lambda x: x > 0),
                                          device=device)     
        
        self.segs = nib.load(save_path_pnet)
        self.segs = nib.as_closest_canonical(self.segs).get_fdata()
        self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn)
        self.label.setText('COMPLETE, Click Axis') #텍스트 변환 
        self.label.setFont(QtGui.QFont("궁서",30)) #폰트,크기 조절 
        self.label.setStyleSheet("Color : black") #글자색 변환
        
    def Button3_click(self):   # display x-axis images
        self.count = 120
        self.ax = 0
        self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn)
        self.update_show()
        
    def Button4_click(self):   # display y-axis images
        self.count = 120
        self.ax = 1
        self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn)
        self.update_show()
        
    def Button5_click(self):   # display z-axis images
        self.count = 75
        self.ax = 2
        self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn)
        self.update_show()

        
    def Button2_click(self): 
        result = nib.Nifti1Image(self.segs, np.eye(4))
        nib.save(result, os.path.join(f'../res/{self.usrId}/', 'final_result.nii.gz')) 
        self.label.setText('Save Complete') #텍스트 변환 
        self.label.setFont(QtGui.QFont("궁서",30)) #폰트,크기 조절 
        self.label.setStyleSheet("Color : black") #글자색 변환
        
    def Button7_click(self):   # run r-net 
        # int_result 생성
        path = f'../res/{self.usrId}/seg/'
        int_pos_result, int_neg_result = save_func(self.imgs, path, self.usrId)
        
        # 이전 수행한 int_result 삭제
        file_path = []
        for (root, directories, files) in os.walk(path):
            for file in files:
                if '.png' in file:
                    os.remove(os.path.join(root, file))

        test_images = self.img_file_names[0]
        save_dir = f'../res/{self.usrId}/result'
        
        # preprocessing
        test_transform = get_transform("valid")

        # device config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        rnet_best_ckpt_dir = "./experiments/best_ckpts/brats3d_rnet_init_train"
        rnet_best_ckpt_path = sorted(glob.glob(f"{rnet_best_ckpt_dir}/*.pt"))[-1]

        rnet = P_RNet3D(c_in=4, c_blk=16, n_classes=2).to(device)
        rnet.load_state_dict(torch.load(rnet_best_ckpt_path))
        rnet.eval()
        
        save_path_pnet = os.path.join(save_dir, "_pred_pnet.nii.gz")
        save_path_rnet = os.path.join(save_dir, "_pred_rnet.nii.gz")

        rnet_pred_labels = rnet_inference(image_path=test_images,
                                      pnet_pred_path=save_path_pnet,
                                      fg_point_path=os.path.join(save_dir, "int_pos_result.npy"),
                                      bg_point_path=os.path.join(save_dir, "int_neg_result.npy"),
                                      save_path=save_path_rnet,
                                      rnet=rnet,
                                      transform=test_transform,
                                      norm_transform=tio.ZNormalization(masking_method=lambda x: x > 0),
                                      device=device)

              
        self.segs = nib.load(save_path_rnet)
        self.segs = nib.as_closest_canonical(self.segs).get_fdata()
        self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn)
        self.label.setText('COMPLETE, Click Axis') #텍스트 변환 
        self.label.setFont(QtGui.QFont("궁서",30)) #폰트,크기 조절 
        self.label.setStyleSheet("Color : black") #글자색 변환

    def update_show(self):   # input numpy image to Qlabel
        new_img = cv2.addWeighted(self.img, 0.7, self.seg, 0.3, 0)
        qimg = QImage(new_img.data, new_img.shape[1], new_img.shape[0], new_img.strides[0], QImage.Format_Grayscale8)
        self.label.setPixmap(QPixmap.fromImage(qimg))
        
        if self.ax==0 : axis='Sagittal'
        elif self.ax==1 : axis='Coronal'
        elif self.ax==2 : axis='Axial'
        self.label_2.setText(f'{axis} : {self.count}') #텍스트 변환 
        self.label_2.setFont(QtGui.QFont("궁서",20)) #폰트,크기 조절 
        self.label_2.setStyleSheet("Color : blue") #글자색 변환
        
        
    def wheelEvent(self, event):   # mouse scroll event

        delta = event.angleDelta()

        if delta.y() > 0:
            self.count += 1
            self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn)
            self.update_show()
           
        elif delta.y() < 0:
            self.count -= 1
            self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn)
            self.update_show()    


    def mousePressEvent(self, event):  #event : QMouseEvent
        #clk = (event.x(), event.y())
        if event.buttons() & Qt.LeftButton:
            self.callback_left(event.pos())
        if event.buttons() & Qt.RightButton:
            self.callback_right(event.pos())
    
    
    def callback_left(self, clk):   # mouse left click event
        self.pn = 1
        self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn, clk=clk)

        self.update_show()

    def callback_right(self, clk):   # mouse right click event
        self.pn = 2
        self.img, self.seg = nextImage(self.usrId, self.imgs, self.segs, self.ax, self.count, pn=self.pn, clk=clk)

        self.update_show()

        
if __name__ == '__main__':        
        
    app = QApplication(sys.argv)
    Dialog = MainDialog()
    Dialog.show()
    app.exec_()