from __future__ import print_function
import logging
import numpy as np
import pandas as pd
import sys

def MoveToWorldFrame(head, himax):
    phi = head[3] + himax[3] + np.pi

    rotation = np.array([[np.cos(himax[3]), -np.sin(himax[3])], [np.sin(himax[3]), np.cos(himax[3])] ])
    xy = np.array([[head[0], head[1]]]).transpose()
    xy = rotation @ xy
    xy = xy + np.array([[himax[0], himax[1]]]).transpose()
    x, = xy[0]
    y, = xy[1]

    return x, y, phi

def SaveResultsToCSV(labels, predictions, timestamps, csvName):

    x_gt = []
    y_gt = []
    z_gt = []
    phi_gt = []
    x_pr = []
    y_pr = []
    z_pr = []
    phi_pr = []

    for i in range(len(timestamps)):
        label = labels[i]
        x_gt.append(label[0])
        y_gt.append(label[1])
        z_gt.append(label[2])
        phi_gt.append(label[3])
        pred = predictions[i]
        x_pr.append(pred[0])
        y_pr.append(pred[1])
        z_pr.append(pred[2])
        phi_pr.append(pred[3])

    df = pd.DataFrame(data={'timestamps': timestamps, 'x_gt': x_gt, 'y_gt': y_gt, 'z_gt': z_gt, 'ph_gt': phi_gt, 'x_pr': x_pr, 'y_pr': y_pr, 'z_pr': z_pr, 'phi_pr': phi_pr})

    df.to_csv(csvName, index=False, header=True)

def SaveResultsToCSVinWorldFrame(labels, predictions, timestamps, camPoses, csvName):
    x_gt = []
    y_gt = []
    #z_gt = []
    phi_gt = []
    x_pr = []
    y_pr = []
    #z_pr = []
    phi_pr = []
    cam_x = []
    cam_y = []
    cam_z = []
    cam_phi = []

    for i in range(len(timestamps)):
        cam = camPoses[i]
        cam_x.append(cam[0])
        cam_y.append(cam[1])
        cam_z.append(cam[2])
        cam_phi.append(cam[3])
        label = labels[i]
        label = MoveToWorldFrame(label, cam)
        x_gt.append(label[0])
        y_gt.append(label[1])
        phi_gt.append(label[2])
        pred = predictions[i]
        pred = MoveToWorldFrame(pred, cam)
        x_pr.append(pred[0])
        y_pr.append(pred[1])
        phi_pr.append(pred[2])

    df = pd.DataFrame(
        data={'timestamps': timestamps, 'cam_x': cam_x, 'cam_y': cam_y, 'cam_z': cam_z, 'cam_phi': cam_phi,
              'x_gt': x_gt, 'y_gt': y_gt, 'ph_gt': phi_gt, 'x_pr': x_pr,
              'y_pr': y_pr, 'phi_pr': phi_pr})

    df.to_csv(csvName, index=False, header=True)