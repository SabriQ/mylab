from mylab.Cvideo import Video
import os,sys,glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mylab.Cdecs import *
import csv
from mylab.Cfile import TrackFile

class NCRvideo(Video):
    exp_name = "Novel Context Recognition"
    def __init__(self,video_path):  
        super().__init__(video_path)
    
    def draw_ncr_rois(self):
        return self.draw_rois(aim="NCR")

    def correct_outlies(self,savepath,show=True,save=False):        

        track = TrackFile(self.video_track_path)

        corrected_coords = track.behave_track[["Body_x","Body_y"]]

        self.masks,self.coords = self.draw_ncr_rois()

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,100)
        ret,frame = cap.read()
        cap.release()
        # correct outlier 
        count = 0
        if show:
            plt.imshow(frame)
            # plt.imshow(self.masks[0])
            for row in corrected_coords.iterrows():    
                if sum(self.masks[0][int(row[1][1]),int(row[1][0]),:])>255:
                    count +=1
                    x_1,y_1 = row[1][0],row[1][1]
                    # plt.scatter(row[1][0],row[1][1],color="red")
                    try:
                        corrected_coords.loc[row[0]] = corrected_coords.loc[row[0]-1]
                        # row changed before and after re-assignment(赋值前后，row 的值变化了)
                        x_2,y_2 = row[1][0],row[1][1]
                    except:
                        corrected_coords.loc[row[0]] = corrected_coords.loc[row[0]+1]
                        # row changed before and after re-assignment(赋值前后，row 的值变化了)
                        x_2,y_2 = row[1][0],row[1][1]

                    # plt.plot((x_1,x_2),(y_1,y_2),ls="--",color="grey")
                    # plt.scatter(row[1][0],row[1][1],color="green")
            plt.plot(corrected_coords["Body_x"],corrected_coords["Body_y"],color="green",lw=1)
            plt.title(self.video_name)
            if save:
                plt.savefig(savepath)
            plt.show()
        else:
            for row in corrected_coords.iterrows():    
                if sum(self.masks[0][int(row[1][1]),int(row[1][0]),:])>255:
                    count +=1
                    corrected_coords.loc[row[0]] = corrected_coords.loc[row[0]-1]
        print("corrected %s points"%count)
        return corrected_coords

    def plot_trajectory_in_mask(self):
        corrected_coords = self.correct_outlies(show=False)
        plt.imshow(self.masks[0])
        plt.plot(corrected_coords["Body_x"],corrected_coords["Body_y"],color="red")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def _construct_heatmap(self):
        corrected_coords = self.correct_outlies(show=False)
        len_x,len_y = corrected_coords.max()-corrected_coords.min()
        min_x,min_y = corrected_coords.min()
        max_x,max_y = corrected_coords.max()
        return corrected_coords,len_x,len_y,min_x,min_y,max_x,max_y




    def plot_heatmap(self,shrink_n=5,norm=None,show=True,save=False,**kwarg):

        corrected_coords,len_x,len_y,min_x,min_y,max_x,max_y = self._construct_heatmap()

        heatmap_matrix = np.full((int(np.ceil(480/shrink_n)),int(np.ceil(640/shrink_n))),0)

        for x,y in zip(corrected_coords["Body_x"],corrected_coords["Body_y"]):
            heatmap_matrix[int(y//shrink_n)-1,int(x//shrink_n)-1] +=1
        print("maxtimes in placebins:%s"%np.max(heatmap_matrix))

        if not norm is None:
            print("Normalized to 1 to %s"%norm)
        else:
            norm = np.max(heatmap_matrix)

        nor = matplotlib.colors.Normalize(0,norm)
        plt.imshow(heatmap_matrix,norm=nor,interpolation="gaussian")
        plt.title(self.video_name)
        plt.colorbar()
        # plt.xticks([])
        # plt.yticks([])
        if save:
            plt.savefig(**kwarg)
        if show:
            plt.show()
        plt.close("all")

    def plot_heatmap_crop(self,shrink_n=5,norm=None,show=True,save=False,**kwarg):

        corrected_coords,len_x,len_y,min_x,min_y,max_x,max_y = self._construct_heatmap()
        #add 10 to pad the heat_matrix
        heatmap_matrix = np.full((int(np.ceil(len_y/shrink_n))+10,int(np.ceil(len_x/shrink_n))+10),0)

        for x,y in zip(corrected_coords["Body_x"],corrected_coords["Body_y"]):
            x = x-min_x
            y = y-min_y
            # center of the padded matrix
            heatmap_matrix[int(y//shrink_n)-1+5,int(x//shrink_n)-1+5] +=1 
        print("maxtimes in placebins:%s"%np.max(heatmap_matrix))

        if not norm is None:
            print("Normalized to 1 to %s"%norm)
        else:
            norm = np.max(heatmap_matrix)

        nor = matplotlib.colors.Normalize(0,norm)
        plt.imshow(heatmap_matrix,norm=nor,interpolation="gaussian")
        plt.title(self.video_name)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        if save:
            plt.savefig(**kwarg)
        if show:
            plt.show()
        plt.close("all")

if __name__=="__main__":

    pass