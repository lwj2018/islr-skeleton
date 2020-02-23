import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, 
    QHBoxLayout, QVBoxLayout, QApplication, QLabel,
    QProgressBar, QTabWidget, QWidget)
from PyQt5.QtCore import QBasicTimer,Qt
from PyQt5.QtGui import QFont
from socket import *
import numpy
import threading

def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]

 
class Interface(QWidget):
    # TODO
    # 重载keypressevent，实现按q键时退出
    
    def __init__(self):
        super().__init__()
        self.captionSize = 20
        self.captionFontFamily = "STHeitiTC-Light"

        # 数据
        self.nsize = 12
        self.arr = numpy.zeros(shape=self.nsize,dtype=float)

        # 初始化client，开启线程
        self.client = socket(AF_INET, SOCK_STREAM)
        self.client.connect(('localhost', 25000))
        self.t = threading.Thread(target=self.func)
        self.t.start()

        self.initUI()

    def func(self):
        while(1):
            recv_into(self.arr,self.client)

        
    def initUI(self):
        self.timer = QBasicTimer()

        guiBox = QHBoxLayout()
        generalMessageBox = QVBoxLayout()
        cardsBox = QVBoxLayout()
        tabs = QTabWidget()
        cardsBox.addWidget(tabs)

        guiBox.addLayout(generalMessageBox)
        guiBox.addLayout(cardsBox)

        #训练进度
        self.trainCaptionLabel = QLabel()
        self.trainCaptionLabel.setText("训练进度")

        self.epochLabel = QLabel()
        self.epochLabel.setText("Epoch /")

        self.captionBox = QHBoxLayout()
        self.captionBox.addWidget(self.trainCaptionLabel)
        self.captionBox.addStretch(1)
        self.captionBox.addWidget(self.epochLabel)
        
        self.trainProgressBar = QProgressBar()
        self.trainProgressBar.setValue(0)

        self.trainProgressLabel = QLabel()
        self.trainProgressLabel.setText("_ / _")

        self.trainProgressBox = QHBoxLayout()
        self.trainProgressBox.addWidget(self.trainProgressBar)
        self.trainProgressBox.addWidget(self.trainProgressLabel)

        #训练集表现
        self.trainPerformanceBox = QVBoxLayout()
        self.trainPerformanceCaptionLabel = QLabel()
        self.trainPerformanceCaptionLabel.setText("训练集表现")
        self.trainPerformanceCaptionLabel.setFont(QFont(self.captionFontFamily,self.captionSize,QFont.Bold))

        self.trainLossLabel = QLabel()
        self.trainLossLabel.setText("Loss    /")

        self.trainTop1AccLabel = QLabel()
        self.trainTop1AccLabel.setText("Acc@1 /")

        self.trainTop5AccLabel = QLabel()
        self.trainTop5AccLabel.setText("Acc@5 /")

        self.trainPerformanceValueBox = QHBoxLayout()
        self.trainPerformanceValueBox.addWidget(self.trainTop1AccLabel)
        self.trainPerformanceValueBox.addStretch(1)
        self.trainPerformanceValueBox.addWidget(self.trainTop5AccLabel)

        self.trainPerformanceBox.addWidget(self.trainPerformanceCaptionLabel)
        self.trainPerformanceBox.addWidget(self.trainLossLabel)
        self.trainPerformanceBox.addLayout(self.trainPerformanceValueBox)

        #测试集表现
        self.evalPerformanceCaptionLabel = QLabel()
        self.evalPerformanceCaptionLabel.setText("测试集表现")
        self.evalPerformanceCaptionLabel.setFont(QFont(self.captionFontFamily,self.captionSize,QFont.Bold))

        self.evalPerformaceTabs = QTabWidget()
        self.evalBestPerformanceTab = QWidget()
        self.evalPerformanceBox = QVBoxLayout(self.evalBestPerformanceTab)
        self.evalLastPerformanceTab = QWidget()
        self.evalLastPerformanceBox = QVBoxLayout(self.evalLastPerformanceTab)
        
        self.evalPerformaceTabs.addTab(self.evalBestPerformanceTab,"Best result")
        self.evalPerformaceTabs.addTab(self.evalLastPerformanceTab,"Last result")

        #测试集最佳表现
        self.evalLossLabel = QLabel()
        self.evalLossLabel.setText("Loss") 
        self.evalLossValueLabel = QLabel()
        self.evalLossValueLabel.setText("/")
        self.evalLossBox = QHBoxLayout()
        self.evalLossBox.addWidget(self.evalLossLabel)
        self.evalLossBox.addStretch(1)
        self.evalLossBox.addWidget(self.evalLossValueLabel)

        self.evalTop1AccLabel = QLabel()
        self.evalTop1AccLabel.setText("Best Top-1 Acc") 
        self.evalTop1AccValueLabel = QLabel()
        self.evalTop1AccValueLabel.setText("/")
        self.evalTop1AccBox = QHBoxLayout()
        self.evalTop1AccBox.addWidget(self.evalTop1AccLabel)
        self.evalTop1AccBox.addStretch(1)
        self.evalTop1AccBox.addWidget(self.evalTop1AccValueLabel)

        self.evalTop5AccLabel = QLabel()
        self.evalTop5AccLabel.setText("Best Top-5 Acc") 
        self.evalTop5AccValueLabel = QLabel()
        self.evalTop5AccValueLabel.setText("/")
        self.evalTop5AccBox = QHBoxLayout()
        self.evalTop5AccBox.addWidget(self.evalTop5AccLabel)
        self.evalTop5AccBox.addStretch(1)
        self.evalTop5AccBox.addWidget(self.evalTop5AccValueLabel)

        self.evalPerformanceBox.addLayout(self.evalLossBox)
        self.evalPerformanceBox.addLayout(self.evalTop1AccBox)
        self.evalPerformanceBox.addLayout(self.evalTop5AccBox)

        #测试集最近一次结果
        self.evalLastLossLabel = QLabel()
        self.evalLastLossLabel.setText("Loss") 
        self.evalLastLossValueLabel = QLabel()
        self.evalLastLossValueLabel.setText("/")
        self.evalLastLossBox = QHBoxLayout()
        self.evalLastLossBox.addWidget(self.evalLastLossLabel)
        self.evalLastLossBox.addStretch(1)
        self.evalLastLossBox.addWidget(self.evalLastLossValueLabel)

        self.evalLastTop1AccLabel = QLabel()
        self.evalLastTop1AccLabel.setText("Top-1 Acc") 
        self.evalLastTop1AccValueLabel = QLabel()
        self.evalLastTop1AccValueLabel.setText("/")
        self.evalLastTop1AccBox = QHBoxLayout()
        self.evalLastTop1AccBox.addWidget(self.evalLastTop1AccLabel)
        self.evalLastTop1AccBox.addStretch(1)
        self.evalLastTop1AccBox.addWidget(self.evalLastTop1AccValueLabel)

        self.evalLastTop5AccLabel = QLabel()
        self.evalLastTop5AccLabel.setText("Top-5 Acc") 
        self.evalLastTop5AccValueLabel = QLabel()
        self.evalLastTop5AccValueLabel.setText("/")
        self.evalLastTop5AccBox = QHBoxLayout()
        self.evalLastTop5AccBox.addWidget(self.evalLastTop5AccLabel)
        self.evalLastTop5AccBox.addStretch(1)
        self.evalLastTop5AccBox.addWidget(self.evalLastTop5AccValueLabel)

        self.evalLastPerformanceBox.addLayout(self.evalLastLossBox)
        self.evalLastPerformanceBox.addLayout(self.evalLastTop1AccBox)
        self.evalLastPerformanceBox.addLayout(self.evalLastTop5AccBox)


        generalMessageBox.addLayout(self.captionBox)
        generalMessageBox.addLayout(self.trainProgressBox)
        generalMessageBox.addLayout(self.trainPerformanceBox)
        generalMessageBox.addWidget(self.evalPerformanceCaptionLabel)
        generalMessageBox.addWidget(self.evalPerformaceTabs)

        #操作界面
        self.operationTab = QWidget()
        self.operationBox = QVBoxLayout(self.operationTab)
        self.controlBox = QHBoxLayout()

        self.runButton = QPushButton()
        self.runButton.setText("运行")

        self.stopButton = QPushButton()
        self.stopButton.setText("停止")

        self.controlBox.addWidget(self.runButton)
        self.controlBox.addStretch(1)
        self.controlBox.addWidget(self.stopButton)

        self.choosePathButton = QPushButton()
        self.choosePathButton.setText("选择存储路径")

        self.saveAsButton = QPushButton()
        self.saveAsButton.setText("另存为...")

        self.operationBox.addLayout(self.controlBox)
        self.operationBox.addWidget(self.choosePathButton)
        self.operationBox.addWidget(self.saveAsButton)

        #设置界面
        self.settingsTab = QWidget()
        self.settingsBox = QVBoxLayout(self.settingsTab)

        self.lrLabel = QLabel()
        self.lrLabel.setText("学习率")

        self.settingsBox.addWidget(self.lrLabel)

        #加入标签页
        tabs.addTab(self.operationTab, "操作")
        tabs.addTab(self.settingsTab, "设置")


        self.setLayout(guiBox)    
        
        self.setWindowTitle('来啊训练啊') 
        self.timer.start(100,self)   
        self.show()

    def timerEvent(self, e):
        self.epochLabel.setText("Epoch "+str(int(self.arr[0])))

        self.trainProgressLabel.setText(str(int(self.arr[1]))
          +" / "+str(int(self.arr[2])))

        if self.arr[2]>0:
            percent = int(self.arr[1]*100/self.arr[2])
        else:
            percent = 0
        self.trainProgressBar.setValue(percent)

        self.trainLossLabel.setText("Loss    %.2f"%self.arr[3])
        self.trainTop1AccLabel.setText("Acc@1 %.3f"%
          self.arr[4]+"%")
        self.trainTop5AccLabel.setText("Acc@5 %.3f"%
          self.arr[5]+"%")

        self.evalLossValueLabel.setText("%.2f"%self.arr[6])
        self.evalTop1AccValueLabel.setText("%.3f"%self.arr[7])
        self.evalTop5AccValueLabel.setText("%.3f"%self.arr[8])

        self.evalLastLossValueLabel.setText("%.2f"%self.arr[9])
        self.evalLastTop1AccValueLabel.setText("%.3f"%self.arr[10])
        self.evalLastTop5AccValueLabel.setText("%.3f"%self.arr[11])

        self.timer.start(100,self)

        
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    gui = Interface()
    sys.exit(app.exec_())