import os
import re

import pandas as pd
import numpy as np
import math

from scipy.ndimage import median_filter, gaussian_filter1d
from pyloess import loess
import scipy.signal as signal


import matplotlib.pyplot as plt
import cv2


#Set parameters of the video 
fps = 240

#Only consider things past this frame to avoid noise from the beginning
frame_thresh = 250 

#Set parameters for smoothing
MEDIAN_WINDOW   = 10
GAUSSIAN_WINDOW = 30
GAUSSIAN_SIGMA  = GAUSSIAN_WINDOW /6 

parts_list = ['Head','Back','Wing_R','Shoulder_R','Trunk','Tail_R','Beak','Chest_L','Chest_R','Tail_L','Shoulder_L','Wing_L','Tail_Center']

#For plotting distance from midline
feature_list = ['Wing_R','Wing_L','Tail_R','Tail_L']
color_list = ['green','purple','blue','orange']
line_types = ['-','-','--','--']

wings = ['Left','Right','Avg']


#For iterating through tests
#Make sure it matches the folder and the order of plotting
kinematics_list = ['Trajectory','Velocity','Acceleration','Distance','Jerk']

pd.options.display.max_colwidth = None

#---------------------------------------------------------------------------------------------------------------
                                            #File Management 


                     #Create dataframe and folders for each file

def get_files (path_csv_processed, path_save, path_vid_folder):
    
    #Initialize list to save everything to a dataframe
    all_rows = []
    
    #Extract date , test type , rec number, and song number for each file in folder
    for file in os.listdir(path_csv_processed):
        if 'test' not in file:
    
            # --- Date ---
            match_date = re.search(r"\d{6}", file)  # looks for 6-digit date like 250525
            date = match_date.group(0) if match_date else None
    
            # --- Test type ---
            if "control" in file and "FD" in file:
                test = "Control_FD"
                
            elif "control" in file:
                test = "Control_MD"
            elif "stim" in file and "control" not in file:
                test = "Stim"
            else:
                test = "Unknown"
    
            # --- Recording number ---
            match_rec = re.search(r"rec(\d+)", file)
            rec_num = match_rec.group(1) if match_rec else None
    
            # --- Song number (last two digits only) ---
            match_song = re.search(r"top_(\d+)\.", file)
            song_num = match_song.group(1)[-2:] if match_song else None
    
    
            #Get and save video files for each song
            vid_file = file
            vid_subfolder = file
            
            vid_file = vid_file.split('000_')[1]
            vid_file = vid_file.replace('.analysis.csv','.mp4')
            
            vid_subfolder =vid_subfolder.split('_video_top')[0] + '\\'
            vid_path = path_vid_folder + vid_subfolder + vid_file
            
            
            #Create subfolders for each date, recording number, song, and frame files         
            path_date = f'{path_save}{date}\\'
            os.makedirs(path_date,exist_ok = True)
            path_rec = f'{path_save}{date}\\rec{rec_num}_{test}\\'
            os.makedirs(path_rec,exist_ok = True)
            path_song = f'{path_save}{date}\\rec{rec_num}_{test}\\Song_{song_num}\\'
            os.makedirs(path_song,exist_ok = True)
            path_frames = f'{path_save}{date}\\rec{rec_num}_{test}\\Song_{song_num}\\Frames\\'
            os.makedirs(path_frames, exist_ok = True) 
    
            #Make subfolders for each plot type in each song folder extracted
            for word in kinematics_list:
                plot_path = f'{path_song}{word}\\'
                os.makedirs(plot_path,exist_ok = True)
    
            #Add all these to a dataframe
            row = {
                "Date": date,
                "Test": test,
                "Rec_num": rec_num,
                "Song_num": song_num,
                "File": file,
                "Folder_path": path_song,
                "Vid_file": vid_path,
                "Frame_path": path_frames
            }
            all_rows.append(row)
    
    # Build one DataFrame
    df_files = pd.DataFrame(all_rows)
    return df_files



                        #Create frames for each video 

def get_frames(df_files, need_frames = True):
    if need_frames:
        for j, row in df_files.iterrows():
            vid = cv2.VideoCapture(row['Vid_file'])
            frame_folder = row['Frame_path']
            
            success,image = vid.read()
            count = 0
        
            #Save into the song folder 
            while success:
                frame_path = os.path.join(frame_folder, f"frame{count:04d}.jpg")      
                cv2.imwrite(frame_path, image)
            
                count += 1
                success,image = vid.read()
        
            vid.release()
        
            print(f'Done getting frames for video - rec{row['Rec_num']} - song {row['Song_num']}')

            
#---------------------------------------------------------------------------------------------------------------------------
                                            #Extracting Coordinates from CSV

#Extract and smooth coordinates            
def extract_coords (csv_path, parts_list=parts_list,fps = fps,MEDIAN_WINDOW = MEDIAN_WINDOW, GAUSSIAN_WINDOW = GAUSSIAN_WINDOW, GAUSSIAN_SIGMA = GAUSSIAN_SIGMA):

    temp = pd.read_csv(csv_path) 

    frame_num = len(temp)

    #Create dataframes for raw and smoothed coordinates from cleaned and interpolated 
    df_raw = pd.DataFrame()
    df_smooth = pd.DataFrame()
    
    #Save frames and time(s) into dataframes
    df_raw['Frame_num'] = range(frame_num)
    df_raw['Time'] = df_raw['Frame_num'] / fps
    df_smooth['Frame_num'] = range(frame_num)
    df_smooth['Time'] = df_smooth['Frame_num'] / fps
    
    #Iterate through each part
    for part in parts_list:
        part_x = part + '.x'
        part_y = part + '.y'
    
        #Save coordinates into dataframe
        df_raw[part_x] = temp[part_x]
        df_raw[part_y] = temp[part_y]
    
        #Calculate smoothed coordinates from gaussian distribution for each part
        pos_x = df_raw[part_x].to_numpy()
        med_x = median_filter(pos_x, size=MEDIAN_WINDOW, mode='nearest')
        smooth_x = gaussian_filter1d(med_x, sigma=GAUSSIAN_SIGMA)
        
        pos_y = df_raw[part_y].to_numpy()
        med_y = median_filter(pos_y, size=MEDIAN_WINDOW, mode='nearest')
        smooth_y = gaussian_filter1d(med_y, sigma=GAUSSIAN_SIGMA)
    
        #save into smoothed dataframe
        df_smooth[part_x] = smooth_x
        df_smooth[part_y] = smooth_y

    print(f'Coordinates Extracted!')
    
    return df_raw, df_smooth


#---------------------------------------------------------------------------------------------------------------------------
                                #Process coordinates from video - not relative to anything

#Calculate kinematics from coordinates:
def calc_kinematics(df_coords,parts_list = parts_list,fps=fps):
    
    #Create Dataframe
    df_kin = pd.DataFrame()
    df_kin['Frame_num'] = df_coords['Frame_num']
    df_kin['Time'] = df_coords['Time']
    
    
    #Iterate through parts
    for part in parts_list:
        part_x = part + '.x'
        part_y = part + '.y'
    
        #Find differences between coordinates
        dx = np.diff(df_coords[part_x], prepend=df_coords[part_x][0])
        dy = np.diff(df_coords[part_y], prepend=df_coords[part_y][0])
    
        #Calculate velocity
        velocity = np.hypot(dx, dy) * fps
    
        #Calculate acceleration
        acceleration = np.gradient(velocity,1/fps)

        #Calculate Jerk
        jerk = np.gradient(acceleration, 1/fps)

    
        #Add to kinematics dataframe
        df_kin[f'{part}_velocity'] = velocity
        df_kin[f'{part}_acceleration'] = acceleration
        df_kin[f'{part}_jerk'] = jerk

    print(f'Done exctracting kinematics!')

    return df_kin 


#Plot kinematics from coordinates:
def plot_kinematics(df_kinematics, df_coords,path_to_plots, vid_name, fps = fps, parts_list = parts_list,want_time = False, show_plot = False):

    for part in parts_list:
    
        #Plot trajectory:
        x_coord = df_coords[f'{part}.x']
        y_coord = df_coords[f'{part}.y']
        
        plt.plot(x_coord,y_coord,color='green',linewidth = 0.5)
        plt.xlabel(f'{part} x-coordinates (px)')
        plt.ylabel(f'{part} y-coordinates in (px)')
        plt.title(f'{part} Trajectory (2D)')
        plt.savefig(f'{path_to_plots}{kinematics_list[0]}\\{part}_Trajectory_{vid_name}.png')

        if show_plot:
            plt.show()
        plt.close()


        
        #Set x-axes for frames and time for next plots:
        frames = df_kinematics['Frame_num'] 
        time = df_kinematics['Time']

        x_axis = frames
        x_label = 'Frames'
        
        if want_time:
            x_axis = time
            x_label = 'Time'

        
        #Plot velocity:
        v = df_kinematics[f'{part}_velocity']
        
        plt.plot(x_axis,v,linewidth = 0.5)
        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{part} velocity in px/frame')
        plt.title(f'{vid_name} | {part} Velocity vs {x_label}')
        plt.savefig(f'{path_to_plots}{kinematics_list[1]}\\{part}_Velocity_{x_label}_{vid_name}.png')

        if show_plot:
            plt.show()
        plt.close()

      
        #Plot acceleration:
        a = df_kinematics[f'{part}_acceleration']
        
        plt.plot(frames,a,color = 'red',linewidth = 0.5)
        plt.xlabel('{x_label}')
        plt.ylabel(f'{part} acceleration in px/frame')
        plt.title(f'{vid_name} | {part} Acceleration vs Frames')
        
        if show_plot:
            plt.show()
            plt.close()
        plt.savefig(f'{path_to_plots}{kinematics_list[2]}\\{part}_Acceleration_{x_label}_{vid_name}.png')
        plt.close()

        
        #Plot jerk:
        a = df_kinematics[f'{part}_jerk']
        
        plt.plot(frames,a,color = 'red',linewidth = 0.5)
        plt.xlabel('{x_label}')
        plt.ylabel(f'{part} Jerk in px/frame')
        plt.title(f'{vid_name} | {part} Jerk vs Frames')
        
        if show_plot:
            plt.show()
            plt.close()
        plt.savefig(f'{path_to_plots}{kinematics_list[4]}\\{part}_Jerk_{x_label}_{vid_name}.png')
        plt.close()
    print(f'Plots Made!')

#-----------------------------------------------------------------------------------------------------------------------
                                    #Extract Relative Distance from Midline

#Calculate and filter distance from midline:
def distance_from_midline(df_coords, save_path, MEDIAN_WINDOW = MEDIAN_WINDOW, GAUSSIAN_WINDOW = GAUSSIAN_WINDOW, GAUSSIAN_SIGMA = GAUSSIAN_SIGMA):

    time = df_coords['Time']
    df_raw_dist = pd.DataFrame()
    
    for feature in feature_list:
    
        #Initialize lists
        distance_Head_list = []
        distance_Back_list = []
        distance_Trunk_list = []
        distance_Tail_list = []
        distance_avg_list = []
    
        #Calculate Euclidian distances from feature to midline uses Head, Back, Trunk, and Tail Center to be midline 
        for i, row in df_coords.iterrows():
            distance_Back = math.sqrt((row[f'{feature}.x'] - row['Back.x'])**2 + (row[f'{feature}.y'] - row['Back.y'])**2)
            distance_Back_list.append(distance_Back)
            
            distance_Head = math.sqrt((row[f'{feature}.x'] - row['Head.x'])**2 + (row[f'{feature}.y'] - row['Head.y'])**2)
            distance_Head_list.append(distance_Head)
        
            distance_Trunk = math.sqrt((row[f'{feature}.x'] - row['Trunk.x'])**2 + (row[f'{feature}.y'] - row['Trunk.y'])**2)
            distance_Trunk_list.append(distance_Trunk)
        
            distance_Tail = math.sqrt((row[f'{feature}.x'] - row['Tail_Center.x'])**2 + (row[f'{feature}.y'] - row['Tail_Center.y'])**2)
            distance_Tail_list.append(distance_Tail)   
        
            distance_avg =np.mean([distance_Back,distance_Head, distance_Trunk,distance_Tail])   
            distance_avg_list.append(distance_avg)
        
    
        #Save them into dataframe
        df_raw_dist['Time'] = time
        df_raw_dist[f'Head-{feature}'] = distance_Head_list
        df_raw_dist[f'Back-{feature}'] = distance_Back_list
        df_raw_dist[f'Trunk-{feature}'] = distance_Trunk_list
        df_raw_dist[f'Tail-{feature}'] = distance_Tail_list
        df_raw_dist[f'Average-{feature}'] = distance_avg_list


    #Median and Gaussian Filtering
    df_dist_filtered = pd.DataFrame()
    df_dist_filtered['Time'] = time 
    
    for column in df_raw_dist:
        if 'Average' in column:
            dist = df_raw_dist[column].to_numpy()
            med_dist = median_filter(dist, size=MEDIAN_WINDOW, mode='nearest')
            smooth_dist = gaussian_filter1d(med_dist, sigma=GAUSSIAN_SIGMA)
            loess_dist = loess(time.to_numpy(),dist, span=0.03, degree=2)
            df_dist_filtered[column] = loess_dist[:,1]

    return df_raw_dist, df_dist_filtered



#Plot WS distances
def plot_distances(df_dist, save_path, vid_name, feature_list = feature_list, color_list = color_list, line_types = line_types, want_time=False, want_combo = False, show_plot = False):
    
    c = 0
    for feature in feature_list:

        #Set x-axis 
        frames = range(len(df_dist))
        time = df_dist['Time']
        x_axis = frames
        x_label = 'Frames'
        if want_time:
            x_axis = time
            x_label = 'Time (s)'

        #Plot for each feature
        plt.plot(x_axis,df_dist[f'Average-{feature}'],label = f'{feature}', color = color_list[c], linestyle = line_types[c])
        plt.xlabel(x_label)
        #plt.ylabel(f'Euclidian distance between {feature} and midline')
        plt.legend()
    
        c += 1

        #Plot and save for each feature in their own folder
        if want_combo is False:
            plt.ylabel(f'Euclidian distance between {feature} and midline')
            plt.title(f'{vid_name} | {feature} - Midline Distance vs {x_label}')    
            plt.savefig(f'{save_path}{kinematics_list[3]}\\{feature}_distance_from_midline_{x_label}_{vid_name}.png')
            if show_plot:
                plt.show()
            plt.close()
            
            

    #Plot combined distances and save 
    if want_combo:
        plt.ylabel(f'Euclidian distance between features and midline')
        plt.title(f'{vid_name} | Features - Midline Distance vs {x_label}')
        plt.savefig(f'{save_path}{kinematics_list[3]}\\Combined_Distance_from_Midline_{x_label}_{vid_name}.png')
        if show_plot:
            plt.show()
        plt.close()
    print(f'Distance Plots Made!')

#---------------------------------------------------------------------------------------------------------------
                        #Processs Kinematics Relavent to Wingspread (WS)

                #Calculate kinematics relative to wingspread
def calc_WS_kinematics(df_dist_filtered, feature_list = feature_list, fps = fps): 
    #Create Dataframe
    df_WS_kin = pd.DataFrame()
    df_WS_kin['Frame_num'] = range(len(df_dist_filtered))
    df_WS_kin['Time'] = df_WS_kin['Frame_num'] /fps
    
    #Iterate through parts
    for feature in feature_list:

        #Calculate velocity
        velocity = np.gradient(df_dist_filtered[f'Average-{feature}'],1/fps)
    
        #Calculate acceleration
        acceleration = np.gradient(velocity, 1/fps)

        #Calculate jerk
        jerk = np.gradient(acceleration, 1/fps)
    
        #Add to kinematics dataframe
        df_WS_kin[f'{feature}_velocity'] = velocity
        df_WS_kin[f'{feature}_acceleration'] = acceleration
        df_WS_kin[f'{feature}_jerk'] = jerk

    
    return df_WS_kin


    
#Plot wingspread kinematics
def plot_WS_kinematics(df_kinematics,path_to_plots, vid_name, fps = fps, parts_list = parts_list,want_time = False, show_plot = False):
    
    #Set x-axes for frames and time for next plots
    frames = df_kinematics['Frame_num'] 
    time = df_kinematics['Time']
    x_axis = frames
    x_label = 'Frames'
    if want_time:
        x_axis = time
        x_label = 'Time'

        
    for feature in feature_list:

        #Plot velocity:
        v = df_kinematics[f'{feature}_velocity']
        plt.plot(x_axis,v,linewidth = 0.5)
        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{feature.replace('Average-','')} velocity in px/frame')
        plt.title(f'{vid_name} | {feature.replace('Average-','')} Velocity vs {x_label}')
        plt.savefig(f'{path_to_plots}{kinematics_list[1]}\\{feature.replace('Average-','')}_Velocity_{x_label}_{vid_name}.png')
        
        if show_plot:
            plt.show()
        plt.close()
        
        
        #Plot acceleration:
        a = df_kinematics[f'{feature}_acceleration']
        plt.plot(frames,a,color = 'red',linewidth = 0.5)
        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{feature.replace('Average-','')} acceleration in px/frame')
        plt.title(f'{vid_name} | {feature.replace('Average-','')} Acceleration vs Frames')
        
        if show_plot:
            plt.show()
            plt.close()
        plt.savefig(f'{path_to_plots}{kinematics_list[2]}\\{feature.replace('Average-','')}_Acceleration_{x_label}_{vid_name}.png')
        plt.close()


        #Plot jerk:
        j = df_kinematics[f'{feature}_jerk']
        plt.plot(frames,j,color = 'darkorange',linewidth = 0.5)
        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{feature.replace('Average-','')} Jerk in px/frame')
        plt.title(f'{vid_name} | {feature.replace('Average-','')} Jerk vs Frames')
        
        if show_plot:
            plt.show()
            plt.close()
        plt.savefig(f'{path_to_plots}{kinematics_list[4]}\\{feature.replace('Average-','')}_Jerk_{x_label}_{vid_name}.png')
        plt.close()
    print(f'Kinematic Plots Made!')


def combo_plots(dict_WS, df_files, show_plot = False):
    for key in dict_WS.keys():
    
        fig, ax= plt.subplots(4,2,sharex=True, sharey='row', figsize = (6,12))
        plt.subplots_adjust(wspace=0.1, hspace=0.2) # Increase horizontal and vertical space
        fig.suptitle(f'{key} - Features', fontsize=14, fontweight ='roman', y= 0.95)
    
    
        #For saving figures
        rec = key.split('-')[0]
        rec = rec[0:3] + '_' + rec[3:]
        rec_num = rec[-2:]
        
        song = key.split('-')[1]
        song = song[0:4] + '_' + song[4:]
        song_num = song[-2:]
    
        fig_title = key.replace('-', '_')
    
        save_path = df_files[(df_files[f'Rec_num'] == f'{rec_num}') & (df_files[f'Song_num'] == f'{song_num}')]['Folder_path'].to_string(index = False)
    
    
    
        c = 0
        
        for wing in wings:
            if 'Avg' not in wing:
                
                df = dict_WS[key][wing]['df_WS']
    
                #Get features you want to plot
                time = df[f'Time'][400:900]
                d = df[f'WS_{wing}'][400:900]
                a = df[f'Acceleration_{wing}'][400:900]
                j = df[f'Jerk_{wing}'][400:900]
                v = df[f'Velocity_{wing}'][400:900]
                frame_onset = dict_WS[key][wing]['WS_Stats']['Timings']['Onset_frame']
                time_onset = dict_WS[key][wing]['WS_Stats']['Timings']['Onset_time']
                
    
                #Plot Dem
                ax[0][c].plot(time, d, color = 'royalblue', label = 'WS')
                ax[0][c].axvline(x = time_onset, color = 'black', linestyle = '--', alpha = 0.5)
                
                ax[0][0].set_ylabel('Wingspread (px)')
                ax[0][c].set_title(f'{wing} Wing')
    
    
                ax[1][c].plot(time,v, color = 'darkcyan', label= 'Velocity')
                ax[1][c].axvline(x = time_onset, color = 'black', linestyle = '--', alpha = 0.5)
                
                ax[1][0].set_ylabel('Velocity (px/s)')
                #ax[1][c].set_title('Velocity')
    
    
                ax[2][c].plot(time,a, color = 'darkred', label = 'Acceleration')
                ax[2][c].axvline(x = time_onset, color = 'black', linestyle = '--', alpha = 0.5)
                
                ax[2][0].set_ylabel(f'Acceleration (px/s\u00b2)')
                #ax[2][c].set_title('Acceleration')
                
    
                ax[3][c].plot(time,j, color = 'darkorange', label = 'Jerk')
                ax[3][c].axvline(x = time_onset, color = 'black', linestyle = '--', label = 'WS Onset',alpha = 0.5)
                
                ax[3][0].set_ylabel(f'Jerk (px/s\u00b3)')
                ax[3][c].set_xlabel('Time (s)')
                #ax[3][c].set_title('Jerk')
                plt.xlabel('Time (s)')
    
    
                c +=1
    
        #fig.savefig(f'{save_path}Features_{fig_title}.jpg')

        if show_plot == False:
            plt.close()

        
    print(f'Summary Plots Created!')
#---------------------------------------------------------------------------------------------------------------

    
#Get averages for both wings
def WS_avg(df_dist, df_WS_kin, wing = None):
    df_WS = pd.DataFrame()


    #Calculate average values and kinematics for both wings
    wing_L = df_dist['Average-Wing_L']
    wing_R = df_dist['Average-Wing_R']
    time = df_dist['Time']

    if wing == 'Avg':
        d = np.mean((wing_L,wing_R),axis = 0)
        v = np.mean((df_WS_kin['Wing_R_velocity'], df_WS_kin['Wing_L_velocity']),axis=0)
        a = np.mean((df_WS_kin['Wing_R_acceleration'], df_WS_kin['Wing_L_acceleration']),axis=0)
        j = np.mean((df_WS_kin['Wing_R_jerk'], df_WS_kin['Wing_L_jerk']),axis=0)

        
    if wing == 'Left':
        d = wing_L
        v =  df_WS_kin['Wing_L_velocity']
        a =  df_WS_kin['Wing_L_acceleration']
        j = df_WS_kin['Wing_L_jerk']
        #print('Left Wing - WS')

    if wing == 'Right':
        d = wing_R
        v =  df_WS_kin['Wing_R_velocity']
        a =  df_WS_kin['Wing_R_acceleration']
        j = df_WS_kin['Wing_R_jerk']
        #print('Right Wing - WS')

        
    #Save to new dataframe just for fun
    df_WS['Time'] = time
    df_WS[f'WS_{wing}'] = d
    df_WS[f'Velocity_{wing}'] = v
    df_WS[f'Acceleration_{wing}']= a
    df_WS[f'Jerk_{wing}']= j


    return df_WS


#---------------------------------------------------------------------------------------------------------------
                                            #Extract wingspread parameters / data


def WS_stats(df_WS, wing = None):

    #Initialize dictionary
    dict_WS_stats = {}
    
    #Save to new dataframe just for fun
    time = df_WS['Time'] 
    wing_avg =df_WS[f'WS_{wing}'] 
    avg_v = df_WS[f'Velocity_{wing}'] 
    avg_a = df_WS[f'Acceleration_{wing}']
    
    
    #Timings - Establish Winspread onset and offset
    #Set threshold to 15% the peak wingspread 
    distance_thresh = (np.max(wing_avg) - np.min(wing_avg)) *.15 + np.min(wing_avg)
    
    #Onset: find the first frame to cross threshold
    bool_on = wing_avg >= distance_thresh 
    for idx in range(len(bool_on)):
        if bool_on[idx] and idx > frame_thresh:
            frame_onset = idx
            break
    
    time_onset = time[frame_onset]
    frame_onset
    
    #Offset: find the first frame to cross threshold
    bool_off = wing_avg <= distance_thresh
    for idx in range(len(bool_off)):
        if idx > frame_onset:
            if np.all(bool_off[idx:idx+10] == True):
                frame_offset = idx
                break
    
    time_offset = time[frame_offset]
    frame_offset

    #Timings - Wingspread Duration 
    WS_duration = time_offset - time_onset

    #Save to dictionary
    dict_WS_stats['Timings'] = {'Duration_WS': WS_duration, 'Onset_frame': frame_onset, 'Onset_time': time_onset, 'Offset_frame': frame_offset, 'Offset_time': time_offset}


    
    #Peak WS
    peak_WS = np.max(wing_avg)
    peak_WS_frame = np.argmax(wing_avg) + 1
    peak_WS_time = time[peak_WS_frame] 
    
    #Latency to peak wingspread
    peak_WS_latency = peak_WS_time - time_onset

    #Save to dictionary
    dict_WS_stats['Peak_WS'] = {'Peak_value': peak_WS,'Peak_latency': peak_WS_latency, 'Peak_frame': peak_WS_frame, 'Peak_time':peak_WS_time}

        
    #Kinematics
    
    #Velocity
    max_v = np.max(avg_v)
    min_v = np.min(avg_v)
    v_max_latency = (time[np.argmax(avg_v)] - frame_onset) 
    v_min_latency = (time[np.argmin(avg_v)] - frame_onset) 
    max_v_frame = np.argmax(avg_v)
    min_v_frame = np.argmin(avg_v)

    
    #Acceleration
    max_a = np.max(avg_a)
    min_a = np.min(avg_a)
    a_max_latency = (time[np.argmax(avg_a)] - frame_onset) 
    a_min_latency = (time[np.argmin(avg_a)] - frame_onset) 
    max_a_frame = np.argmax(avg_a)
    min_a_frame = np.argmin(avg_a)

    #Save to dictionary 
    dict_WS_stats['Kinematics'] = {'Velocity':{'Max_velocity': max_v,'Max_velocity_latency': v_max_latency, 'Max_velocity_frame': max_v_frame, 'Min_velocity': min_v, 'Min_velocity_latency': v_min_latency,'Min_velocity_frame': min_v_frame},
                            'Acceleration':{'Max_acceleration': max_a,'Max_acceleration_latency': a_max_latency, 'Max_acceleration_frame': max_a_frame, 'Min_acceleration': min_a, 'Min_acceleration_latency': a_min_latency,'Min_acceleration_frame': min_a_frame}}

    return dict_WS_stats

#---------------------------------------------------------------------------------------------------------------
                                            #Extract Wing Pump Info
    
def find_pumps(df_WS, wing = None, want_peaks = False):

    #Load data
    wing_avg = df_WS[f'WS_{wing}']
    avg_v = df_WS[f'Velocity_{wing}']
    avg_a = df_WS[f'Acceleration_{wing}']


    #Make threshold 
    peak_WS = np.max(wing_avg)
    min_WS = np.min(wing_avg)
    floor = ((peak_WS - min_WS)*0.35) + min_WS

    
    #Find peaks and troughs to find pumps 
    peaks, props= signal.find_peaks(wing_avg, height = [floor, peak_WS])
    inv_wing_avg = -wing_avg
    troughs,temp = signal.find_peaks(inv_wing_avg, height = [-peak_WS,-floor])
    
    #DS = Downstroke
    #US = Upstroke

    #Create Dictionary 
    dict_pumps = {}

    
    #Iterate through peaks
    for j in range(len(troughs)):
        
        #The last peak doesn't count as a pump
        if j < len(peaks)-1 and len(peaks) == len(troughs) +1:
            pump_WS = wing_avg[peaks[j]:peaks[j+1]]
    
            #Find frames for each pump
            #Do I need to save these in the dictionary?
            frames_pump = range(peaks[j],peaks[j + 1]-1)
            frames_DS = range(peaks[j],troughs[j])
            frames_US = range(troughs[j], peaks[j+1] - 1)
            
            
            #Establish locations for each pump and stroke
            pump_start = frames_pump[0]
            pump_end = frames_pump[-1]
            trough_loc = troughs[j]
    
            mean_v_DS = np.mean(avg_v[frames_DS])
            mean_v_US = np.mean(avg_v[frames_US])
    
            mean_a_DS = np.mean(avg_a[frames_DS])
            mean_a_US = np.mean(avg_a[frames_US])
            
            
            #Time between pumps / between downstroke and upstroke 
            #Duration is the same as latency in this case as 
            pump_latency = (frames_pump[-1] - frames_pump[0]) /240 
            duration_DS = (frames_DS[-1] - frames_DS[0]) /240 
            duration_US = (frames_US[-1] - frames_US[0]) /240 
    
    
            #Save into dictionary 
            dict_pumps[f'Pump {str(j+1)}'] = {'Frames':{'Pump_Frames': frames_pump, 'DS_Frames': frames_DS, 'US_Frames': frames_US}, 'Durations':{'Pump_Duration':pump_latency,'DS_Duration':duration_DS, 'US_Duration':duration_US},'Kinematics':{'Avg_V_DS':mean_v_DS,'Avg_V_US':mean_v_US,'Avg_A_DS':mean_a_DS,'Avg_A_US':mean_a_US}}
        if want_peaks:
            return dict_pumps, peaks, troughs

    
    return dict_pumps



#---------------------------------------------------------------------------------------------------------------

                                        #Between Song Comparisons
def song_averages(dict_WS):

    avg_WS = pd.DataFrame()
    
    for wing in wings:
    
        average_d = []
        average_v = []
        average_a = []
        average_j = []
        
        for key in dict_WS.keys():
            time = dict_WS[key][wing]['df_WS'][f'Time']
            d = dict_WS[key][wing]['df_WS'][f'WS_{wing}']
            v = dict_WS[key][wing]['df_WS'][f'Velocity_{wing}']
            a = dict_WS[key][wing]['df_WS'][f'Acceleration_{wing}']
            j = dict_WS[key][wing]['df_WS'][f'Jerk_{wing}']
    
            average_d.append(d)
            average_v.append(v)
            average_a.append(a)
            average_j.append(j)
    
            
        average_d = pad_list(average_d)
        average_v = pad_list(average_v)
        average_a = pad_list(average_a)
        average_j = pad_list(average_j)
    
    
        average_d = np.mean(average_d, axis = 0)
        average_v = np.mean(average_v, axis = 0)
        average_a = np.mean(average_a, axis = 0)
        average_j = np.mean(average_j, axis = 0)
    
        avg_WS[f'WS_{wing}'] = average_d
        avg_WS[f'Velocity_{wing}'] = average_v
        avg_WS[f'Acceleration_{wing}'] = average_a
        avg_WS[f'Jerk_{wing}'] = average_j

    return avg_WS

                            #Find timings using average Jerk 

def find_timings(dict_WS):

    #Velocity Histogram
    v_list = []
    j_list = []
    
    frame_thresh_on = 400 #Only look for crossing v_thresh after this frame 
    frame_thresh_off =1000
    #Set the trough of the bimodal distribution as the velocity threshold
    v_thresh = np.exp(5)  #Make sure to convert back to actual velocity 
    neg_v_thresh = -np.exp(5.67) 
    
    for wing in wings:
    
        j_avg = []

        for key in dict_WS.keys():
            time = dict_WS[key][wing]['df_WS'][f'Time']
            d = dict_WS[key][wing]['df_WS'][f'WS_{wing}']
            j = dict_WS[key][wing]['df_WS'][f'Jerk_{wing}']
            j_cut = dict_WS[key][wing]['df_WS'][f'Jerk_{wing}'][900:]
            v = (dict_WS[key][wing]['df_WS'][f'Velocity_{wing}'])

            j_list.extend(j_cut)

            
            #Define Putative Movement
            #If the velocity is above or below the velcotiy thresholds label point red 
            new_d = d[(v >= v_thresh) | (v <= neg_v_thresh)].to_numpy()
            idx_d = d[(v >= v_thresh) | (v <= neg_v_thresh)].index.to_numpy()
            
            #Make it within the range of thre frame threshold 
            new_d = new_d[(idx_d >= frame_thresh_on) & (idx_d <= frame_thresh_off)]
            idx_d = idx_d[(idx_d >= frame_thresh_on) & (idx_d <= frame_thresh_off)]

            #Find the first point where the velocity is positive
            putative_onset = np.min(idx_d[v[idx_d] >= 0])
            putative_offset = np.max(idx_d[v[idx_d] <= 0])

            
            
                                                            #Finidng Onset through Jerk
            
            #Set the Jerk threshold to 3x the standard deviation 
            j_thresh = np.nanstd(j_list)*2
            neg_j_thresh = -np.nanstd(j_list)*2
            
            #Find local jerk maxima
            j_maxima,props = signal.find_peaks(j, height = [j_thresh,np.max(j)])
            j_maxima = j_maxima[(j[j_maxima] >= j_thresh) & (j_maxima >= frame_thresh_on) & (j_maxima <=frame_thresh_off)] #Filter maxima to be within frame thresholds and above jerk threshold (optional)
            
            #Find distances between jerk maxima frame and putative movement onset 
            distances = np.abs(j_maxima - idx_d[0])
            min_index = np.where(distances == np.min(distances)) #Find the smallest distance 
            frame_onset = j_maxima[min_index][0] #The smallest distance is onset of movement




                                                            #Finidng Offset through Jerk

            neg_j_maxima,props = signal.find_peaks(-j, height = [j_thresh,np.max(-j)])
            neg_j_maxima = neg_j_maxima[(j[neg_j_maxima] <= neg_j_thresh) & (neg_j_maxima >= frame_thresh_on) & (neg_j_maxima <=frame_thresh_off)] #Filter maxima to be within frame thresholds and above jerk threshold (optional)
            
            distances = np.abs(neg_j_maxima - putative_offset)
            min_index = np.where(distances == np.min(distances)) #Find the smallest distance 
            frame_offset = neg_j_maxima[min_index][0] #The smallest distance is onset of movement

        

                                                            #aving


            time_onset = time[frame_onset]
            time_offset = time[frame_offset]
            duration = time_offset - time_onset
            peak_latency = time[dict_WS[key][wing]['WS_Stats']['Peak_WS']['Peak_frame']] - time_onset

            #Update dictionary 
            timings = dict_WS[key][wing]['WS_Stats']['Timings']
            timings['Onset_frame'] = frame_onset
            timings['Onset_time'] = time_onset 
            timings['Offset_frame'] = frame_offset
            timings['Offset_time'] = time_offset
            timings['Duration_WS'] = duration
    
            dict_WS[key][wing]['WS_Stats']['Peak_WS']['Peak_latency'] = peak_latency


    return dict_WS


#---------------------------------------------------------------------------------------------------------------

                                #Troubleshooting 


                #Show specific frame for any video
def show_frame(df_files,desired_frame,date,recording,song):

    # Select the matching row from df_files
    selection_df = df_files.loc[
        (df_files["Rec_num"].str.contains(str(recording))) &
        (df_files["Song_num"].str.contains(str(song))) &
        (df_files["Date"].str.contains(str(date)))
    ]
    
    # Ensure selection is not empty
    if selection_df.empty:
        print("No matching entry found in df_files")
    else:
        # Get frame folder path (first match only)
        frame_folder = selection_df["Frame_path"].iloc[0]
    
        if not os.path.exists(frame_folder):
            print(f"Frame folder not found: {frame_folder}")
        else:
            # Build full frame path
            frame_file = os.path.join(frame_folder, f"frame{desired_frame:04d}.jpg")
    
            if os.path.exists(frame_file):
                print(f"✅ Found frame: {frame_file}")
                
                # Load and display
                image = cv2.imread(frame_file)
                cv2.imshow(f"Frame {desired_frame}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Frame {desired_frame} not found in {frame_folder}")

            return 



                            #Show video of specified frames 
def check_vid(start_frame, end_frame, path_frame_folder, output_path):

    #Set Output Path
    out_path = f"{output_path}clip_video_frame{start_frame}-{end_frame}.mp4"
    
    #Set video parameters
    fps = 30
    frame_size = (1920, 1080)
    frames = range(start_frame,end_frame,1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

    #Stitch together frames to make video 
    for f in frames:
        for file in os.listdir(path_frame_folder):
            if f'frame{f:04}' in file:
                img = cv2.imread(path_frame_folder + file)
        out.write(img)
    out.release()
    print(f'Video for frames {start_frame} - {end_frame} saved as {out_path}')

#---------------------------------------------------------------------------------------------------------------
                                                    #General functions 

    #Rescale stuff
def rescale (array, array_2):
    new_max = np.nanmax(array_2)
    new_min = np.nanmin(array_2)
    rescaled = new_min + (array - np.nanmin(array)) * (new_max - new_min) / (array.max() - np.nanmin(array))
    return rescaled


#Pad to same size 
def pad_list(lists):

    lengths = []

    for i in range(len(lists)):
        length = len(lists[i])
        lengths.append(length)
    
    max_frames = np.max(lengths)
    new_list = []
    for i in lists:
        b = list(i)
        if len(b) == max_frames:
            new_list.append(b)
        if len(b) < max_frames:
            for _ in range(max_frames - len(b)):
                b.append(np.nan)
    
            new_list.append(b)
    return new_list


def basic_linreg(x,y):

    n = len(x)

    #Find means and standard deviations
    sx = np.std(x)
    sy = np.std(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    #correlation coefficient
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum((x*y))
    sum_x2 = np.sum(np.square(x))
    sum_y2 = np.sum(np.square(y))

    r = ((n*sum_xy) - (sum_x * sum_y))/(np.sqrt(((n*sum_x2 - ((sum_x)**2)) * (n*sum_y2 - ((sum_y)**2)))))


    #Find beta weights
    B1 = r * (sy/sx)
    B0 = mean_y - (B1*mean_x)

    #Create model
    model = B0 + B1*x

    print(f'The beta weights are B0 = {B0} | B1 = {B1}. The correleation coefficient is r = {r}.')

    return model