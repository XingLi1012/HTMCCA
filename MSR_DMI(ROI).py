import scipy.io as sio
from numpy import *
from PIL import Image#图片处理库
import scipy.io as sio
import os
import sys
from numpy import *
import time
import datetime
import copy#深拷贝

PATH_R1='E:\\01_李兴1盘\\01_博士生学术区\\01_人体行为识别数据库与特征\\01_数据库\\02_处理库\\01_MSR\\01_深度数据\\03_6视角投影图(mat)(int16)\\'
# PATH_W1='C:\\Users\\99677\\Desktop\\MSR_分段DMI\\'
PATH_W1='E:\\01_李兴1盘\\01_博士生学术区\\01_人体行为识别数据库与特征\\01_数据库\\02_处理库\\01_MSR\\01_深度数据\\MSR_分段DMI(续)\\'

if(1==1):
    def DeepCopy(Objects):  # 深拷贝

        objects = copy.deepcopy(Objects)
        return objects

    def ROI_array_return_boundary_value(Array):
        array = DeepCopy(Array)
        h, w = shape(array)
        # print(shape(array))

        if (1 == 1):  # 宽度轴范围
            w_mim = 10000
            w_max = 0
            axis_w = array.sum(axis=0)  #对每一列相加
            for ww in range(w):  # 找最小点
                if (ww > w_mim):
                    break
                else:
                    if (axis_w[ww] != 0):  # 最小点
                        if (ww < w_mim):
                            w_mim = ww
                        break

            for ww in range(w):  # 找最大点
                if (w - 1 - ww < w_max):
                    break
                else:
                    if (axis_w[w - 1 - ww] != 0):  # 最大点
                        if (w - 1 - ww > w_max):
                            w_max = w - 1 - ww
                        break
        if (1 == 1):  # 高度轴范围
            h_mim = 10000
            h_max = 0
            axis_h = array.sum(axis=1)  #
            for hh in range(h):  # 找最小点
                if (hh > h_mim):
                    break
                else:
                    if (axis_h[hh] != 0):  # 最小点
                        if (hh < h_mim):
                            h_mim = hh
                        break

            for hh in range(h):  # 找最大点
                if (h - 1 - hh < h_max):
                    break
                else:
                    if (axis_h[h - 1 - hh] != 0):  # 最大点
                        if (h - 1 - hh > h_max):
                            h_max = h - 1 - hh
                        break

        return (h_mim, h_max, w_mim, w_max)

    def Normalized_brightness(Array):
        array = DeepCopy(Array)
        h, w = shape(array)
        array_max = amax(array)

        array_min = 100000000
        # DTM_w非零最小值
        for hh in range(h):  #
            for ww in range(w):
                # if(ww<24):continue
                # if(ww>255):break
                if (array[hh][ww] != 0):
                    if (array[hh][ww] < array_min):
                        array_min = array[hh][ww]
                        # DTM_w归一化
        for hh in range(h):  #
            for ww in range(w):
                # if(ww<24):continue
                # if(ww>255):break
                if (array[hh][ww] != 0):
                    array[hh][ww] = (array[hh][ww] - array_min) * 255 / (array_max - array_min)
        return array



i=0
files= os.listdir(PATH_R1)
# print(shape(files))
for file in files: #遍历文件夹
    i=i+1
    if(i>571):
        if(file[-3:]=='mat'):
            if (1 == 1):  # 读取mat文件
                if "beh." in file:  # 读前视图
                    data = sio.loadmat(PATH_R1 + "\\" + file)
                    depth = data['depth']

                    f, h, w = shape(depth)
                    print(f)
                    DMI_beh_all = zeros((240, 320))
                    #print(amax(depth))
                    for hh in range(h):
                        for ww in range(w):
                            DMI_beh_all[hh][ww] = amax(depth[:, hh:hh + 1, ww:ww + 1])
                    (h_mim, h_max, w_mim, w_max) = ROI_array_return_boundary_value(array(DMI_beh_all))
                    #print(h_mim, h_max, w_mim, w_max)

                    DMI_beh=zeros((h_max-h_mim, w_max-w_mim))
                    k=f//6 #6帧为一段
                    #print(k)
                    for kk in range(k):
                        DMI_beh=zeros((h_max-h_mim, w_max-w_mim))
                        for hh in range(h_mim, h_max + 1):
                            for ww in range(w_mim, w_max + 1):
                                DMI_beh=array(DMI_beh)
                                DMI_beh[hh-h_mim-1][ww-w_mim-1]=amax(depth[(kk*6):(kk*6+7), hh:hh + 1, ww:ww + 1])
                        DMI_beh = Normalized_brightness(DMI_beh)
                        # if(kk==5):
                        #     DMI_beh = Image.fromarray(array(DMI_beh))  #
                        #     DMI_beh = DMI_beh.convert('L')
                        #     print(DMI_beh.show())
                        count = 1

                        if (1 == count):  # 存储开关

                            DMI_beh = Image.fromarray(array(DMI_beh))  #
                            DMI_beh = DMI_beh.convert('L')
                            DMI_beh.save(
                                PATH_W1 + "\\" + file[:-7] + 'f_' + str(kk+1) + '.png')


                    if((f-6*k)<6 and (f-6*k)>0):
                        DMI_beh = zeros((h_max - h_mim, w_max - w_mim))
                        for hh in range(h_mim, h_max + 1):
                            for ww in range(w_mim, w_max+1):
                                DMI_beh = array(DMI_beh)
                                DMI_beh[hh-h_mim-1][ww-w_mim-1] = amax(depth[f-5:f+1, hh:hh+ 1, ww:ww + 1])
                        DMI_beh = Normalized_brightness(DMI_beh)
                        count=1
                        if (1 == count):  # 存储开关

                            DMI_beh = Image.fromarray(array(DMI_beh))  #
                            DMI_beh = DMI_beh.convert('L')
                            DMI_beh.save(
                                PATH_W1 + "\\" + file[:-7] + 'f_' + str(k+1) + '.png')

                    print(file + ':' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))



                if "rig." in file:  #
                    data = sio.loadmat(PATH_R1 + "\\" + file)
                    depth = data['depth']

                    f, h, w = shape(depth)
                    DMI_rig_all = zeros((240, 700))
                    for ff in range(f):
                        for hh in range(h):
                            for ww in range(w):
                                DMI_rig_all[hh][ww] = amax(depth[:, hh:hh + 1, ww:ww + 1])

                    (h_mim, h_max, w_mim, w_max) = ROI_array_return_boundary_value(array(DMI_rig_all))
                    # print(h_mim, h_max, w_mim, w_max)

                    DMI_rig = zeros((h_max - h_mim, w_max - w_mim))

                    k = f // 6  # 6帧为一段
                    # print(k)
                    for kk in range(k):
                        DMI_rig = zeros((h_max - h_mim, w_max - w_mim))
                        for hh in range(h_mim, h_max + 1):
                            for ww in range(w_mim, w_max + 1):
                                DMI_rig = array(DMI_rig)
                                DMI_rig[hh - h_mim - 1][ww - w_mim - 1] = amax(
                                    depth[(kk * 6):(kk * 6 + 7), hh:hh + 1, ww:ww + 1])
                        DMI_rig = Normalized_brightness(DMI_rig)
                        count = 1
                        if (1 == count):  # 存储开关

                            DMI_rig = Image.fromarray(array(DMI_rig))  #
                            DMI_rig = DMI_rig.convert('L')
                            DMI_rig.save(
                                PATH_W1 + "\\" + file[:-7] + 's_' + str(kk + 1) + '.png')

                    if ((f-6*k)<6 and (f-6*k)>0):
                        DMI_rig = zeros((h_max - h_mim, w_max - w_mim))
                        for hh in range(h_mim, h_max + 1):
                            for ww in range(w_mim, w_max + 1):
                                DMI_rig = array(DMI_rig)
                                DMI_rig[hh - h_mim - 1][ww - w_mim - 1] = amax(depth[f - 5:f + 1, hh:hh + 1, ww:ww + 1])
                        DMI_rig = Normalized_brightness(DMI_rig)
                        count = 1
                        if (1 == count):  # 存储开关

                            DMI_rig = Image.fromarray(array(DMI_rig))  #
                            DMI_rig = DMI_rig.convert('L')
                            DMI_rig.save(
                                PATH_W1 + "\\" + file[:-7] + 's_' + str(k+1) + '.png')

                    print(file + ':' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))



                if "top." in file:
                    data = sio.loadmat(PATH_R1 + "\\" + file)
                    depth = data['depth']

                    f, h, w = shape(depth)
                    DMI_top_all = zeros((700, 320))
                    for ff in range(f):
                        for hh in range(h):
                            for ww in range(w):
                                DMI_top_all[hh][ww] = amax(depth[:, hh:hh + 1, ww:ww + 1])

                    (h_mim, h_max, w_mim, w_max) = ROI_array_return_boundary_value(array(DMI_top_all))
                    DMI_top = zeros((h_max - h_mim, w_max - w_mim))

                    k = f  // 6  # 6帧为一段
                    # print(k)
                    for kk in range(k):
                        DMI_top = zeros((h_max - h_mim, w_max - w_mim))
                        for hh in range(h_mim, h_max + 1):
                            for ww in range(w_mim, w_max + 1):
                                DMI_top = array(DMI_top)
                                DMI_top[hh - h_mim - 1][ww - w_mim - 1] = amax(
                                    depth[(kk * 6):(kk * 6 + 7), hh:hh + 1, ww:ww + 1])
                        DMI_top = Normalized_brightness(DMI_top)
                        count = 1
                        if (1 == count):  # 存储开关
                            #DMI_top = Normalized_brightness(DMI_top)
                            DMI_top = Image.fromarray(array(DMI_top))  #
                            DMI_top = DMI_top.convert('L')
                            DMI_top.save(
                                PATH_W1 + "\\" + file[:-7] + 't_' + str(kk + 1) + '.png')

                    if ((f-6*k)<6 and (f-6*k)>0):
                        DMI_top = zeros((h_max - h_mim, w_max - w_mim))
                        for hh in range(h_mim, h_max + 1):
                            for ww in range(w_mim, w_max + 1):
                                DMI_top = array(DMI_top)
                                DMI_top[hh - h_mim - 1][ww - w_mim - 1] = amax(depth[f - 5:f + 1, hh:hh + 1, ww:ww + 1])
                        DMI_top = Normalized_brightness(DMI_top)
                        count = 1
                        if (1 == count):  # 存储开关
                            # DMI_top = Normalized_brightness(DMI_top)
                            DMI_top = Image.fromarray(array(DMI_top))  #
                            DMI_top = DMI_top.convert('L')
                            DMI_top.save(
                                PATH_W1 + "\\" + file[:-7] + 't_' + str(k+1) + '.png')

                    print(file+':'+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))