# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\samet\OneDrive\Masaüstü\Yeni klasör (2)\gameDesigner4.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1192, 640)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(27, 40, 56);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.leName = QtWidgets.QLineEdit(self.centralwidget)
        self.leName.setGeometry(QtCore.QRect(30, 30, 601, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.leName.setFont(font)
        self.leName.setStyleSheet("border: 1px solid;\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(200, 23, 109);\n"
"")
        self.leName.setText("")
        self.leName.setObjectName("leName")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(10, 80, 1171, 541))
        self.groupBox.setStyleSheet("background-color: rgb(42, 71, 94);")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.lblGameInfos = QtWidgets.QLabel(self.groupBox)
        self.lblGameInfos.setGeometry(QtCore.QRect(650, 220, 491, 171))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblGameInfos.setFont(font)
        self.lblGameInfos.setStyleSheet("color: rgb(255, 255, 255);")
        self.lblGameInfos.setObjectName("lblGameInfos")
        self.lblPicture = QtWidgets.QLabel(self.groupBox)
        self.lblPicture.setGeometry(QtCore.QRect(20, 80, 600, 320))
        self.lblPicture.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lblPicture.setText("")
        self.lblPicture.setScaledContents(True)
        self.lblPicture.setObjectName("lblPicture")
        self.lblTitle = QtWidgets.QLabel(self.groupBox)
        self.lblTitle.setGeometry(QtCore.QRect(20, 20, 601, 50))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.lblTitle.setFont(font)
        self.lblTitle.setAutoFillBackground(False)
        self.lblTitle.setStyleSheet("color: rgb(255, 255, 255);")
        self.lblTitle.setObjectName("lblTitle")
        self.btnPreviousGame = QtWidgets.QPushButton(self.groupBox)
        self.btnPreviousGame.setGeometry(QtCore.QRect(20, 480, 150, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnPreviousGame.setFont(font)
        self.btnPreviousGame.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnPreviousGame.setStyleSheet("background-color: rgb(27, 40, 56);\n"
"color: rgb(102, 192, 244);")
        self.btnPreviousGame.setObjectName("btnPreviousGame")
        self.btnNextGame = QtWidgets.QPushButton(self.groupBox)
        self.btnNextGame.setGeometry(QtCore.QRect(470, 480, 150, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnNextGame.setFont(font)
        self.btnNextGame.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnNextGame.setStyleSheet("background-color: rgb(27, 40, 56);\n"
"color: rgb(102, 192, 244);")
        self.btnNextGame.setObjectName("btnNextGame")
        self.leSkip = QtWidgets.QLineEdit(self.groupBox)
        self.leSkip.setGeometry(QtCore.QRect(580, 420, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.leSkip.setFont(font)
        self.leSkip.setStyleSheet("color: rgb(255, 255, 255);")
        self.leSkip.setPlaceholderText("")
        self.leSkip.setObjectName("leSkip")
        self.l = QtWidgets.QLabel(self.groupBox)
        self.l.setGeometry(QtCore.QRect(540, 420, 31, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.l.setFont(font)
        self.l.setStyleSheet("color: rgb(255, 255, 255);")
        self.l.setObjectName("l")
        self.btnPreviousPic = QtWidgets.QPushButton(self.groupBox)
        self.btnPreviousPic.setGeometry(QtCore.QRect(270, 410, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.btnPreviousPic.setFont(font)
        self.btnPreviousPic.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnPreviousPic.setStyleSheet("")
        self.btnPreviousPic.setFlat(True)
        self.btnPreviousPic.setObjectName("btnPreviousPic")
        self.btnNextPic = QtWidgets.QPushButton(self.groupBox)
        self.btnNextPic.setGeometry(QtCore.QRect(330, 410, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.btnNextPic.setFont(font)
        self.btnNextPic.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnNextPic.setStyleSheet("")
        self.btnNextPic.setFlat(True)
        self.btnNextPic.setObjectName("btnNextPic")
        self.teDescription = QtWidgets.QTextEdit(self.groupBox)
        self.teDescription.setEnabled(False)
        self.teDescription.setGeometry(QtCore.QRect(650, 80, 501, 111))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.teDescription.setFont(font)
        self.teDescription.setStyleSheet("color: rgb(255, 255, 255);")
        self.teDescription.setObjectName("teDescription")
        self.btnDisplay = QtWidgets.QPushButton(self.centralwidget)
        self.btnDisplay.setGeometry(QtCore.QRect(660, 30, 91, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.btnDisplay.setFont(font)
        self.btnDisplay.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnDisplay.setStyleSheet("color: rgb(102, 192, 244);\n"
"background-color: rgb(42, 71, 94);\n"
"")
        self.btnDisplay.setFlat(False)
        self.btnDisplay.setObjectName("btnDisplay")
        self.btnClear = QtWidgets.QPushButton(self.centralwidget)
        self.btnClear.setGeometry(QtCore.QRect(760, 30, 61, 35))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.btnClear.setFont(font)
        self.btnClear.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnClear.setStyleSheet("color: rgb(102, 192, 244);\n"
"background-color: rgb(42, 71, 94);")
        self.btnClear.setObjectName("btnClear")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1192, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Game Recommendation"))
        self.leName.setPlaceholderText(_translate("MainWindow", "Enter the game name..."))
        self.lblGameInfos.setText(_translate("MainWindow", "<html><head/><body><p>Similarity</p><p>Review Score</p><p>Release Date</p><p>Developer</p><p>Publisher</p></body></html>"))
        self.lblTitle.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Title</p></body></html>"))
        self.btnPreviousGame.setText(_translate("MainWindow", "Previous"))
        self.btnNextGame.setText(_translate("MainWindow", "Next"))
        self.leSkip.setText(_translate("MainWindow", "1"))
        self.l.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" text-decoration: underline;\">Skip:</span></p></body></html>"))
        self.btnPreviousPic.setText(_translate("MainWindow", "◄"))
        self.btnNextPic.setText(_translate("MainWindow", "►"))
        self.teDescription.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;\"><span style=\" font-family:\'monospace\'; color:#000000; background-color:#ffffff;\">Description</span></p></body></html>"))
        self.btnDisplay.setText(_translate("MainWindow", "Display"))
        self.btnClear.setText(_translate("MainWindow", "Clear"))