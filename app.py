# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 08:53:17 2021

@author: Micah Gross
"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\BASPO\.spyder-py3\LLOK"
# streamlit run LLOK_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import xlsxwriter
from io import BytesIO
import base64
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

##%%
@st.cache(allow_output_mutation=True)# avoids error due to changing this function's output in a successive step
def split_trials(data_export_files, Options):#(Directory,Files,opts):#,opts
    from scipy.signal import savgol_filter
    Split_Trials={}
    # if any([Opts[x] for x in Opts.keys() if type(Opts[x])==bool]):
    #     print('Processing options:')
    #     for o in [sorted(Opts.keys())[i] for i,key in enumerate(sorted(Opts.keys())) if Opts[key]==True]:
    #         print(o)
        
    for file in data_export_files:# file = data_export_files[0] # file = data_export_files[data_export_files.index(file)+1]
        # load_nr = int(file.name.split('.')[0])
        # '''
        # file_name = data_export_file_names[0]
        # file_name = data_export_file_names[0].replace('.txt','.3.txt')
        # int(file_name.split('.txt')[0].split('_')[-1][0])
        # '''
        load_nr = int(file.name.split('.txt')[0].split('_')[-1][0])
        if load_nr not in Split_Trials.keys():
            Split_Trials[load_nr] = {}
        
        # if len(file.name)>5:# e.g., like 1.1.txt, not 1.txt
        if len(file.name.split('.txt')[0].split('_')[-1])>1:# e.g., like 1.1.txt, not 1.txt
            # rep_nr_first = int(file.name.split('.')[1])
            rep_nr_first = int(file.name.split('.txt')[0].split('_')[-1].split('.')[1])
        else:# e.g. file='1.txt'
            rep_nr_first = 1

        df = pd.read_csv(file, sep='\t', header=5, names=['timestamp', 'position', 'F_left', 'F_right'])# get raw data into dataframe
        df['timestamp'] = pd.Series([pd.Timestamp(df.loc[i, 'timestamp']) for i in df.index])# convert to true timestamp format
        df['sample_duration'] = pd.Series([0] + [(df.loc[i, 'timestamp'] - df.loc[i-1, 'timestamp']).delta/10**9 for i in df.index[1:]])# delta return nanoseconds; dividing by 10**9 gives seconds
        df['time'] = df['sample_duration'].cumsum()
        df['position'] = df['position']/1000.# convert mm to m
        if Options['smooth_position']:
            df['position'] = savgol_filter(df['position'], 11, 3, mode='nearest')
            
        if Options['smooth_F']:
            df['F_left'] = savgol_filter(df['F_left'], 11, 3, mode='nearest')
            df['F_right'] = savgol_filter(df['F_right'], 11, 3, mode='nearest')
            
        df['F_tot'] = df['F_left'] + df['F_right']
        for side in ['left', 'right', 'tot']:# side='left'
            df['RFD_'+side] = pd.concat([pd.Series(0), df['F_'+side].diff().iloc[1:]])
            
        df['v'] = pd.concat([pd.Series(0), -1*df['position'].diff().iloc[1:]/df['sample_duration'].iloc[1:]])# velocity as dx/dt, reversing the direction in the process
        if Options['smooth_v']:
            df['v'] = savgol_filter(df['v'], 11, 3, mode='nearest')
            
        df['P'] = df['F_tot']*df['v']
        [pos_min, pos_max] = [df['position'].min(), df['position'].max()]# range of position measurments
        pos_mid = (pos_min + pos_max)/2.# middle position, crossed during each action
        idx_mid = [i for i in df.index[1:] if ((df.loc[i-1, 'position']>=pos_mid) & (df.loc[i, 'position']<pos_mid))]# time points where middle position is crossed (direction specific), indicating an action
        crop_on = []
        crop_off = []
        for n,i_mid in enumerate(idx_mid):# n,i_mid = 0,idx_mid[0] # n,i_mid = n+1,idx_mid[n+1]
            crop_on.append(i_mid-1000 if i_mid-1000>=0 else 0)# 1000 samples prior to crossing middle position
            crop_off.append(i_mid+1000 if i_mid+1000<len(df) else len(df))# 1000 samples after crossing middle position
        
        nr_trials = len(idx_mid)
        for r in list(range(nr_trials)):# r=0
            rep_nr_in_load = r + rep_nr_first
            Split_Trials[load_nr][rep_nr_in_load] = df.loc[crop_on[r]:crop_off[r], ['time', 'sample_duration', 'position', 'F_left', 'F_right', 'F_tot', 'v', 'P', 'RFD_left', 'RFD_right', 'RFD_tot']]
            
    return Split_Trials

##%%
@st.cache()
def crop_center_trials(Split_Trials):
    Trials = {}
    for load_nr in sorted(Split_Trials.keys()):# load_nr = sorted(Split_Trials.keys())[0]
        Trials[load_nr] = {}
        for trial_nr in sorted(Split_Trials[load_nr].keys()):# trial_nr = sorted(Split_Trials[load_nr].keys())[0]
            # extract trial from dictionary
            trial = Split_Trials[load_nr][trial_nr]
            # index and time of peak force
            i_Fpeak = trial['F_tot'].idxmax()
            t_Fpeak = trial.loc[i_Fpeak,'time']
            # crop roughly and center
            on_rough = trial[trial['time']>t_Fpeak-0.5].index.min()# index 0.5 prior to Fpeak
            off_rough = trial[trial['time']>t_Fpeak+0.5].index.min()-1# index 0.5 prior to Fpeak
            [on_rough, off_rough] = [on_rough if on_rough>=trial.index.min() else trial.index.min(),# in case data recording did not start early enough
                                    off_rough if off_rough<=trial.index.max() else trial.index.max()]# in case data recorded stopped too early
            trial = trial.loc[on_rough:off_rough].reset_index(drop=True)
            # reset time variable to start at 0
            trial['time'] = trial['sample_duration'].cumsum()
            # put it back in the dictionary
            Trials[load_nr][trial_nr] = trial
    
    return Trials

##%%
@st.cache()
def crop_precisely(Trials, f_threshold=20, s_threshold=0.06, RFD_threshold=1):# ,v_threshold=0.1,F_threhold_off=40
    '''
    sample input:
        f_threshold=20
        s_threshold=0.06
        RFD_threshold=1
    sample execution:
        Cropped_Trials = crop_precisely(Trials)
        Cropped_Trials = crop_precisely(Trials,f_threshold=20,s_threshold=0.06,RFD_threshold=1)
    '''
    Cropped_Trials = {}
    for load_nr in sorted(Trials.keys()):# load_nr=sorted(Trials.keys())[0]
        if load_nr not in Cropped_Trials.keys():
            Cropped_Trials[load_nr] = {}
            
        for trial_nr in sorted(Trials[load_nr].keys()):# trial_nr=sorted(Trials[load_nr].keys())[0]
            # extract trial from dictionary
            trial = Trials[load_nr][trial_nr]
            
            i_50pctFpeak = trial[trial['F_tot']>0.5*trial['F_tot'].max()].index.min()# the last time total force is below half of max
            try:
                on_RFD_0 = min([trial[((trial.index<i_50pctFpeak) & (trial['RFD_left']<0))].index.max(),
                                    trial[((trial.index<i_50pctFpeak) & (trial['RFD_right']<0))].index.max()])# last time up until 50% total Fpeak that either left or right RFD is below 0
            except:
                on_RFD_0 = trial[((trial.index<i_50pctFpeak) & (trial['RFD_tot']<0))].index.max()# in case F_left and F_right don't exist because calculate_F_from_a==True
        
            try:
                on_RFD = min([trial[((on_RFD_0<=trial.index) & (trial.index<i_50pctFpeak) & (trial['RFD_left']<f_threshold))].index.max(),
                                    trial[((on_RFD_0<=trial.index) & (trial.index<i_50pctFpeak) & (trial['RFD_right']<f_threshold))].index.max()])# last time between on_RFD_0 and 50% total Fpeak that either left or right RFD is below threshold
            except:
                on_RFD = 0
            
            on_s = trial[trial['position']<trial.loc[:10,'position'].mean()-s_threshold].index.min()# first time position departs from starting posistion (mean of first 11 samples) by at least 's_threhold'
            try:
                on_F = min([trial[((trial.index<=on_s) & (trial.index<trial['F_tot'].idxmax()) & (trial['F_left']<f_threshold))].index.max(),
                                trial[((trial.index<=on_s) & (trial.index<trial['F_tot'].idxmax()) & (trial['F_right']<f_threshold))].index.max()])# last time up until total Fpeak that either left or right F is below threshold
            except:
                on_F = 0
                
            on = on_RFD if ((on_RFD>0) & (on_RFD<trial['v'].idxmax())) else on_F
            off = trial['v'].idxmax()
            # put cropped trial into new dictionary
            Cropped_Trials[load_nr][trial_nr] = trial.loc[on:off].copy()
            # reset time to begin at 0
            Cropped_Trials[load_nr][trial_nr]['time'] = Cropped_Trials[load_nr][trial_nr]['time']-Cropped_Trials[load_nr][trial_nr]['time'].iloc[0]
            
    return Cropped_Trials
        
##%%
def get_trial_parameters(Cropped_Trials, loads_kg):
    '''
    sample execution:
        trial_parameters = get_trial_parameters(Cropped_Trials)
    '''
    trial_parameters = pd.DataFrame()
    for load_nr in sorted(Cropped_Trials.keys()):# load_nr = sorted(Cropped_Trials.keys())[0]
        for trial_nr in sorted(Cropped_Trials[load_nr].keys()):# trial_nr = sorted(Cropped_Trials[load_nr].keys())[0]
            # extract trial from dictionary
            trial = Cropped_Trials[load_nr][trial_nr]
            # get parameters for cropped trial
            tp = pd.DataFrame({'Load_nr': load_nr,
                             'Trial_nr': trial_nr,
                             'Load_kg': loads_kg[load_nr-1],
                             'Fpeak': trial['F_tot'].max(),
                             'Fmean': trial['F_tot'].mean(),
                             'vpeak': trial['v'].max(),
                             'vmean': trial['v'].mean(),
                             'Ppeak': trial['P'].max(),
                             'Pmean': trial['P'].mean(),
                             'RFDpeak': trial['RFD_tot'].max(),
                             'RFDmean': trial['RFD_tot'].mean(),
                             'Fpeak_left': trial['F_left'].max(),
                             'Fpeak_right': trial['F_right'].max(),
                             'Fmean_left': trial['F_left'].mean(),
                             'Fmean_right': trial['F_right'].mean(),
                             'RFDpeak_left': trial['RFD_left'].max(),
                             'RFDpeak_right': trial['RFD_right'].max(),
                             'Lenght': trial['position'].max() - trial['position'].min(),
                             'Duration': trial['time'].max(),
                             't_Fpeak': trial.loc[trial['F_tot'].idxmax(), 'time']},
                index=[0])
            trial_parameters = trial_parameters.append(tp).reset_index(drop=True)
            
    trial_parameters = trial_parameters[['Load_nr','Trial_nr','Load_kg','Fpeak','Fmean','vpeak','vmean','Ppeak','Pmean','RFDpeak','RFDmean','Fpeak_left','Fpeak_right','Fmean_left','Fmean_right','RFDpeak_left','RFDpeak_right','Lenght','Duration','t_Fpeak']]
    
    return trial_parameters
    
##%%
def get_load_parameters(subj_name, test_date, trial_parameters, method='mean3', criterion='Ppeak', oneLine=False):#,method='best',criterion='P'
    '''
    sample input:
        method = 'mean3'
        criterion = 'Ppeak'
        oneLine = False
    sample execution:
        load_parameters = get_load_parameters(subj_name, test_date, trial_parameters, method='mean3', criterion='Ppeak', oneLine=False)
        oneLine_load_parameters = get_load_parameters(subj_name, test_date, trial_parameters, method='mean3', criterion='Ppeak', oneLine=True)
    '''
    if oneLine==False:
        load_parameters=pd.DataFrame(columns=['subj_name','test_date']+list(trial_parameters.columns))
    elif oneLine==True:
        load_parameters=pd.DataFrame(columns=['subj_name','test_date'])
        
    for load_nr in sorted(trial_parameters['Load_nr'].unique()):# load_nr = sorted(trial_parameters['Load_nr'].unique())[0]
        load_trials=trial_parameters[trial_parameters['Load_nr']==load_nr]
        if method=='mean3':
            idxs=load_trials.sort_values(by=[criterion]).iloc[1:4].index
        elif method=='best':
            idxs=load_trials[criterion].idxmax()
            
        if oneLine==False:
            selected_trials=load_trials.loc[idxs]
            load_parameters.loc[load_nr,'Load_nr']=load_nr
            for par in trial_parameters.columns[2:]:# par = trial_parameters.columns[2:][0]
                load_parameters.loc[load_nr,par]=selected_trials[par].mean()
                
        elif oneLine==True:
            try:
                selected_trials=selected_trials.append(load_trials.loc[idxs])
            except:
                selected_trials=load_trials.loc[idxs]
            
            for par in trial_parameters.columns[2:]:# par = trial_parameters.columns[2:][0]
                load_parameters.loc[0,par+'_'+str(load_nr)]=selected_trials[par].mean()
            
    load_parameters['subj_name']=subj_name
    load_parameters['test_date']=test_date
        
    return load_parameters

##%%
def quick_plot(Trials, Cropped_Trials, trial_nr):# trial_nr = 1
    trials = Trials[list(Trials.keys())[0]]
    cropped_trials = Cropped_Trials[list(Cropped_Trials.keys())[0]]
    trial = trials[trial_nr]
    cropped_trial = cropped_trials[trial_nr]
    time = trial['time'] - trial.loc[cropped_trial.index[0],'time']
    cropped_time = cropped_trial['time']# - trial.loc[cropped_trial.index[0],'time']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,
                             y=trial['F_tot'],
                             name='F l+r',
                             line={'color': 'blue',
                                   'width': 1,
                                   },
                             ),
                  )
    fig.add_trace(go.Scatter(x=cropped_time,#['time'],
                             y=cropped_trial['F_tot'],
                             name='cropped',
                             line={'color': 'red',
                                   'width': 2,
                                   },
                             ),
                  )
    fig.update_yaxes(
        title_text='force (N)',
        )
    fig.update_xaxes(
        title_text='time (s)',
        )
    fig.update_layout(
        title='trial '+str(trial_nr),
        showlegend=True,# fig['layout']['showlegend'] = True
        )
    
    return fig

##%%
def all_plots(Trials, Cropped_Trials):
    fig = make_subplots(
        rows=len(Trials.keys()),
        row_titles=['load '+str(ld) for ld in Trials.keys()],
        # cols=max([len(Trials[ld]) for ld in Trials.keys()]),
        cols=max([max([tr for tr in Trials[ld].keys()]) for ld in Trials.keys()]),
        subplot_titles=sum([['trial '+str(t) if ld==1 else '' for t in range(1, 1+max([len(Trials[ld]) for ld in Trials.keys()]))] for ld in Trials.keys()],[]),
        # subplot_titles=sum([['trial '+str(t) for t in range(1, 1+max([len(Trials[ld]) for ld in Trials.keys()]))] for ld in Trials.keys()],[]),
        shared_xaxes=True,
        shared_yaxes=True,
        )
    for l,ld in enumerate(sorted(Trials.keys()), start=1):# l,ld = 1,list(Trials.keys())[0] # l,ld = l+1,list(Trials.keys())[l]
        for t,tr in enumerate(sorted(Trials[ld].keys()), start=1):# t,tr = 1,list(Trials[ld].keys())[0] # t,tr = t+1,list(Trials[ld].keys())[t]
            trial = Trials[ld][tr]
            cropped_trial = Cropped_Trials[ld][tr]
            time = trial['time'] - trial.loc[cropped_trial.index[0],'time']
            cropped_time = cropped_trial['time']# - trial.loc[cropped_trial.index[0],'time']
            fig.add_trace(
                go.Scatter(
                    y=trial['F_tot'],
                    x=time,
                    name='F l+r',
                    line={'color': 'blue',
                          'width': 1,
                          },
                    ),
            row=ld,
            col=tr
            )
            fig.add_trace(
                go.Scatter(
                    y=cropped_trial['F_tot'],
                    x=cropped_time,
                    name='cropped',
                    line={'color': 'red',
                          'width': 2,
                          },
                    ),
            row=ld,
            col=tr
            )
            if ld == max(Trials.keys()):
                fig['layout']['xaxis'+str(
                    (ld-1)*max([len(Trials[ld]) for ld in Trials.keys()])+tr
                    )]['title'] = {'text': 'time (s)'}
            if tr == 1:
                fig['layout']['yaxis'+str(
                    (ld-1)*max([len(Trials[ld]) for ld in Trials.keys()])+tr
                    )]['title'] = {'text': 'force (N)'}
            # if tr == max([len(Trials[ld]) for ld in Trials.keys()]):
            #     fig['layout']['showlegend'] = True
            # else:
            #     fig['layout']['showlegend'] = False
        # fig.update_yaxes(
        #     title_text='force (N)',
        #     )
        # fig.update_xaxes(
        # title_text='time (s)',
        # )
    fig['layout']['showlegend'] = False
    # fig['layout']['xaxis11']['title'] = {'text': 'time (s)'}
    fig.show()        
    return fig
    
##%%
def significant_digits(x, n):
    '''
    Parameters
    ----------
    x : float
        the value or array to round.
    n : int
        the number of significant digits to which to round.

    Returns
    -------
    x_rounded : float
        the rounded value or array.
    
    https://stackoverflow.com/questions/57590046/round-a-series-to-n-number-of-significant-figures
    '''
    # xs = [123.949, 23.87, 1.9865, 0.0129500]
    # x=int(xs[0])
    # x = xs[0]
    # x = xs[1]
    # x = xs[2]
    # x = xs[3]
    # x = xs
    # x = np.array(xs)
    # x = pd.Series(xs)
    # x = pd.Series(xs, index=range(5,9))
    # x = pd.DataFrame(np.array([xs, list(reversed(xs))]).T, columns=['a','b'])
    def round_value(xi, n):
        try:# if it's a numerical value
            float(xi)
            if type(xi) == int:
                xi_rounded = xi
            if 'float' in str(type(xi)):
                offset = int(n - np.floor(np.log10(abs(xi)))) - 1
                xi_rounded = np.round(xi, offset)
                if xi_rounded==int(xi_rounded):
                    xi_rounded = int(np.round(xi, offset))
        except:# if it's not numerical
            xi_rounded = xi
        return xi_rounded
    if type(x) == int or 'float' in str(type(x)):
        x_rounded = round_value(x, n)
    
    if type(x) == list:
        x_rounded = [round_value(xi, n) for xi in x]
    if type(x) == np.ndarray:
        x_rounded = np.array(
            [round_value(xi, n) for xi in x]
            )
    if type(x) == pd.Series:
        x_rounded = pd.Series(
            [round_value(xi, n) for xi in x],
            index=x.index
            )
    if type(x) == pd.DataFrame:
        x_rounded = pd.DataFrame(
                np.array([[round_value(xi, n) for xi in x[col]] for col in x.columns]).T,
                columns=x.columns
                )
    return x_rounded

##%%
def generate_excel(trial_parameters, load_parameters, oneLine_load_parameters, subj_name, test_date, **kwargs):
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    if 'sign_digits' in kwargs:
        sign_digits = kwargs.get('sign_digits')
        trial_parameters = significant_digits(trial_parameters, sign_digits)
        load_parameters = significant_digits(load_parameters, sign_digits)
        oneLine_load_parameters = significant_digits(oneLine_load_parameters, sign_digits)
    writing_excel_container = st.empty()
    writing_excel_container.text('writing to excel')
    output = BytesIO()
    
    with pd.ExcelWriter(output) as writer:
        trial_parameters.to_excel(writer, sheet_name='trial parameters', index=False)
        load_parameters.to_excel(writer, sheet_name='load parameters', index=False)
        oneLine_load_parameters.to_excel(writer, sheet_name='load parameters one line', index=False)
        oneLine_load_parameters.T.to_excel(writer, sheet_name='load parameters one column', index=True, header=False)
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
    download_filename = '_'.join([subj_name,
                                  str(test_date),
                                  'Results'
                                  ]
                                 ) + '.xlsx'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Results as Excel File</a>' # decode b'abc' => abc
#%%
st.write("""

# Langlauf Zugtest Datenauswertung

""")
st.sidebar.header('Options')
Options = {}
c1,c2,c3 = st.sidebar.beta_columns([0.5,1,0.5])
c1.write('quick check')
c3.write('entire test')
Options['evaluation_mode'] = c2.select_slider('', options=[' ', '  '],
                                              # value=' '
                                               value='  '
                                              )
if Options['evaluation_mode']==' ':
    Options['evaluation_mode'] = 'quick check'
elif Options['evaluation_mode']=='  ':
    Options['evaluation_mode'] = 'entire test'

Options['save_variables'] = False# st.sidebar.checkbox('save variables', value=True)# 
Options['smooth_position'] = st.sidebar.checkbox('smooth position', value=True,
                                                          # key='smooth_position',
                                                          )
Options['smooth_F'] = st.sidebar.checkbox('smooth force', value=False,
                                                          # key='smooth_F',
                                                          )
Options['smooth_v'] = st.sidebar.checkbox('smooth velocity', value=False,
                                                          # key='smooth_v',
                                                          )
Options['parameters_to_excel'] = st.sidebar.checkbox('write results to excel', value=True)
if Options['evaluation_mode'] == 'quick check':
    upper_container = st.empty()
    quick_check_file = st.file_uploader("upload one txt export file", accept_multiple_files=False)
    if quick_check_file is not None:
        Split_Trials = split_trials([quick_check_file], Options)
        Trials = crop_center_trials(Split_Trials)
        Cropped_Trials = crop_precisely(Trials)
        with upper_container:
            col1, col2 = st.beta_columns([0.2, 1])
            trial_nr = col1.radio('select trial in file to display',
                                  options=sorted(Trials[list(Trials.keys())[0]].keys()),
                                  )
            quick_fig = quick_plot(Trials, Cropped_Trials, trial_nr)
            col2.plotly_chart(quick_fig)
        if Options['save_variables']:

            for (path, _, files) in os.walk(os.path.join(os.getcwd(), 'saved_variables')):
                for f in files:
                    os.remove(os.path.join(path, f))

            for f in [quick_check_file]:
                with open(os.path.join(os.getcwd(), 'saved_variables','.'.join(f.name.split('.')[:-1])+'_bytesIO.txt'), 'wb') as fp:
                    fp.write(f.getbuffer())

            jsonified_Split_Trials = {}
            for load_nr in Split_Trials.keys():
                jsonified_Split_Trials[load_nr] = {}
                for rep_nr_in_load in Split_Trials[load_nr].keys():
                    jsonified_Split_Trials[load_nr][rep_nr_in_load] = Split_Trials[load_nr][rep_nr_in_load].to_json(orient='index', date_format='iso')
            with open(os.path.join(os.getcwd(), 'saved_variables','Split_Trials.json'), 'w') as fp:
                json.dump(jsonified_Split_Trials, fp)
    
            jsonified_Trials = {}
            for load_nr in Trials.keys():
                jsonified_Trials[load_nr] = {}
                for rep_nr_in_load in Trials[load_nr].keys():
                    jsonified_Trials[load_nr][rep_nr_in_load] = Trials[load_nr][rep_nr_in_load].to_json(orient='index', date_format='iso')
            with open(os.path.join(os.getcwd(),'saved_variables','Trials.json'), 'w') as fp:
                json.dump(jsonified_Trials, fp)

            jsonified_Cropped_Trials = {}
            for load_nr in Cropped_Trials.keys():
                jsonified_Cropped_Trials[load_nr] = {}
                for rep_nr_in_load in Cropped_Trials[load_nr].keys():
                    jsonified_Cropped_Trials[load_nr][rep_nr_in_load] = Cropped_Trials[load_nr][rep_nr_in_load].to_json(orient='index', date_format='iso')
            with open(os.path.join(os.getcwd(),'saved_variables','Cropped_Trials.json'), 'w') as fp:
                json.dump(jsonified_Cropped_Trials, fp)

            with open(os.path.join(os.getcwd(), 'saved_variables','Options.json'), 'w') as fp:
                json.dump(Options, fp)
                
                
    
    
if Options['evaluation_mode'] == 'entire test':
    with st.beta_expander('file upload', expanded=True):
        data_export_files = st.file_uploader('upload all txt export files', accept_multiple_files=True)
    if data_export_files is not None and data_export_files != []:
        with st.beta_expander('test settings', expanded=True):
            # col1, col2, col3 = st.beta_columns([1,0.4,1])
            # sex = col2.select_slider(
            #     'select athlete sex',
            #     options=['male', 'female'],
            #     )
            subj_name = ' '.join(reversed(data_export_files[0].name.replace('.txt','').split('-')[0].split('_')[:2]))

            # st.write(
            #     ' '.join(reversed(data_export_files[0].name.replace('.txt','').split('-')[0].split('_')[:2]))
            #     )
            # st.write(subj_name)
            # st.write(type(subj_name))
            test_date = datetime.date(
                    *[
                        int(x) for x in [
                            '20'+x for x in data_export_files[0].name.replace('.txt','').split('-')[0].split('_')[4:5]
                            ] + [
                                x for x in data_export_files[0].name.replace('.txt','').split('-')[0].split('_')[3:5]
                                ]
                        ]
                    )
            # st.write(
            #     datetime.date(
            #         *[
            #             int(x) for x in [
            #                 '20'+x for x in data_export_files[0].name.replace('.txt','').split('-')[0].split('_')[4:5]
            #                 ] + [
            #                     x for x in data_export_files[0].name.replace('.txt','').split('-')[0].split('_')[3:5]
            #                     ]
            #             ]
            #         )
            #     )
            # st.write(test_date)
            # st.write(type(test_date))
            test_date = st.date_input('select test date',
                                      value=test_date,
                                      )
            subj_name = st.text_input('enter athlete name',
                                      value=subj_name,
                                      )
            body_mass = st.text_input('body mass (kg)',
                                      # value=80
                                      )
            if body_mass != '':
                try:
                    body_mass = float(body_mass)
                except:
                    body_mass = None
                    st.write('Error: body mass must be numerical')
            
            st.write('enter loads (kg)')
            col1, col2, col3 = st.beta_columns(3)
            load_1 = col1.text_input('load 1',
                                      # value=6
                                      )
            load_2 = col2.text_input('load 2',
                                      # value=12
                                      )
            load_3 = col3.text_input('load 3',
                                      # value=18
                                      )
            if all([
                    load_1 != '',
                    load_2 != '',
                    load_3 != ''
                    ]):
                try:
                    loads_kg = [float(load_1), float(load_2), float(load_3)]# load_1, load_2, load_3 = 6, 18, 30 or load_1, load_2, load_3 = 6.5, 18.5, 30.5 or load_1, load_2, load_3 = '6', '18.5', '30.5'
                    if not all([loads_kg[i]<loads_kg[i+1] for i in range(len(loads_kg)-1)]):
                        loads_kg = None
                        st.write('Error: loads must be in ascending order')
                except:# load_1, load_2, load_3 = 'a', 'b', []
                    st.write('Error: loads must be numerical')
            else:
                loads_kg = None
        
    # #     # expander._setattr__('expanded', False)

        if all([
                subj_name != '',
                body_mass != '',
                loads_kg is not None,
                ]):
            Split_Trials = split_trials(data_export_files, Options)
            trial_selections = {}
            with st.beta_expander('trial selector', expanded=True):
                for ld in sorted(Split_Trials.keys()):
                    trial_selections[ld] = st.multiselect(
                        label='select valid trials, load '+str(ld),
                        options=[tr for tr in sorted(Split_Trials[ld].keys())],
                        default=[tr for tr in sorted(Split_Trials[ld].keys())],
                        )
            # if trial_selections != {}:
            #     st.write(trial_selections)
            Selected_Trials = {}
            for ld in trial_selections.keys():
                Selected_Trials[ld] = {}
                for tr in trial_selections[ld]:
                    Selected_Trials[ld][tr] = Split_Trials[ld][tr]
            
            Trials = crop_center_trials(Selected_Trials)
            Cropped_Trials = crop_precisely(Trials)
            
            fig = all_plots(Trials, Cropped_Trials)
            st.plotly_chart(fig)
            
            trial_parameters = get_trial_parameters(Cropped_Trials, loads_kg)
            load_parameters = get_load_parameters(subj_name, test_date, trial_parameters, method='mean3', criterion='Ppeak', oneLine=False)
            oneLine_load_parameters = get_load_parameters(subj_name, test_date, trial_parameters, method='mean3', criterion='Ppeak', oneLine=True)
            if Options['parameters_to_excel']:
                st.markdown(generate_excel(trial_parameters, load_parameters, oneLine_load_parameters, subj_name, test_date), unsafe_allow_html=True)#, sign_digits=3
            
            if Options['save_variables']:
                for (path, _, files) in os.walk(os.path.join(os.getcwd(), 'saved_variables')):
                    for f in files:
                        os.remove(os.path.join(path, f))
                
                with open(os.path.join(os.getcwd(), 'saved_variables','subj_name.json'), 'w') as fp:
                    json.dump(subj_name, fp)
    
                with open(os.path.join(os.getcwd(), 'saved_variables','body_mass.json'), 'w') as fp:
                    json.dump(body_mass, fp)
        
                with open(os.path.join(os.getcwd(), 'saved_variables','loads_kg.json'), 'w') as fp:
                    json.dump(loads_kg, fp)
        
                for f in data_export_files:
                    with open(os.path.join(os.getcwd(), 'saved_variables','.'.join(f.name.split('.')[:-1])+'_bytesIO.txt'), 'wb') as fp:
                        fp.write(f.getbuffer())
        
                with open(os.path.join(os.getcwd(), 'saved_variables','test_date.json'), 'w') as fp:
                    json.dump(str(test_date.strftime('%Y-%m-%d')), fp)# json.dump(test_date.strftime('%Y-%m-%dT%H:%M:%S.%f'), fp)
        
                jsonified_Split_Trials = {}
                for load_nr in Split_Trials.keys():
                    jsonified_Split_Trials[load_nr] = {}
                    for rep_nr_in_load in Split_Trials[load_nr].keys():
                        jsonified_Split_Trials[load_nr][rep_nr_in_load] = Split_Trials[load_nr][rep_nr_in_load].to_json(orient='index', date_format='iso')
                with open(os.path.join(os.getcwd(), 'saved_variables','Split_Trials.json'), 'w') as fp:
                    json.dump(jsonified_Split_Trials, fp)
        
                with open(os.path.join(os.getcwd(), 'saved_variables','trial_selections.json'), 'w') as fp:
                    json.dump(trial_selections, fp)
        
                jsonified_Selected_Trials = {}
                for load_nr in Selected_Trials.keys():
                    jsonified_Selected_Trials[load_nr] = {}
                    for rep_nr_in_load in Selected_Trials[load_nr].keys():
                        jsonified_Selected_Trials[load_nr][rep_nr_in_load] = Selected_Trials[load_nr][rep_nr_in_load].to_json(orient='index', date_format='iso')
                with open(os.path.join(os.getcwd(), 'saved_variables','Selected_Trials.json'), 'w') as fp:
                    json.dump(jsonified_Selected_Trials, fp)
        
                jsonified_Cropped_Trials = {}
                for load_nr in Cropped_Trials.keys():
                    jsonified_Cropped_Trials[load_nr] = {}
                    for rep_nr_in_load in Cropped_Trials[load_nr].keys():
                        jsonified_Cropped_Trials[load_nr][rep_nr_in_load] = Cropped_Trials[load_nr][rep_nr_in_load].to_json(orient='index', date_format='iso')
                with open(os.path.join(os.getcwd(), 'saved_variables','Cropped_Trials.json'), 'w') as fp:
                    json.dump(jsonified_Cropped_Trials, fp)

                jsonified_Trials = {}
                for load_nr in Trials.keys():
                    jsonified_Trials[load_nr] = {}
                    for rep_nr_in_load in Trials[load_nr].keys():
                        jsonified_Trials[load_nr][rep_nr_in_load] = Trials[load_nr][rep_nr_in_load].to_json(orient='index', date_format='iso')
                with open(os.path.join(os.getcwd(), 'saved_variables','Trials.json'), 'w') as fp:
                    json.dump(jsonified_Trials, fp)
        
                with open(os.path.join(os.getcwd(), 'saved_variables','Options.json'), 'w') as fp:
                    json.dump(Options, fp)
        




                    
            
            
                
        
#%%
def recover_saved_variables():
    #%%
    import os
    import json
    from io import BytesIO
    from datetime import datetime
    import pandas as pd

    with open(os.path.join(os.getcwd(),'saved_variables','Options.json'), 'r') as fp:
        Options = json.load(fp)
    if Options['evaluation_mode'] == 'quick check':
        for (_, _, files) in os.walk(os.path.join(os.getcwd(), 'saved_variables')):
            files = [os.path.join(os.getcwd(), 'saved_variables', f) for f in files if '_bytesIO.txt' in f]
        for f in files:
            with open(f, 'rb') as fh:
                quick_check_file = BytesIO(fh.read())

        with open(os.path.join(os.getcwd(),'saved_variables','Split_Trials.json'), 'r') as fp:
            int_key_dict = json.load(fp)# Trials = json.load(fp)
            Split_Trials = {}
            for load_nr in int_key_dict.keys():
                Split_Trials[int(load_nr)] = {}
                for rep_nr_in_load in int_key_dict[load_nr].keys():
                    Split_Trials[int(load_nr)][int(rep_nr_in_load)] = pd.read_json(int_key_dict[load_nr][rep_nr_in_load], orient='index', convert_dates=True)
    
        with open(os.path.join(os.getcwd(),'saved_variables','Trials.json'), 'r') as fp:
            int_key_dict = json.load(fp)# Trials = json.load(fp)
            Trials = {}
            for load_nr in int_key_dict.keys():
                Trials[int(load_nr)] = {}
                for rep_nr_in_load in int_key_dict[load_nr].keys():
                    Trials[int(load_nr)][int(rep_nr_in_load)] = pd.read_json(int_key_dict[load_nr][rep_nr_in_load], orient='index', convert_dates=True)
        
        with open(os.path.join(os.getcwd(),'saved_variables','Cropped_Trials.json'), 'r') as fp:
            int_key_dict = json.load(fp)# Trials = json.load(fp)
            Cropped_Trials = {}
            for load_nr in int_key_dict.keys():
                Cropped_Trials[int(load_nr)] = {}
                for rep_nr_in_load in int_key_dict[load_nr].keys():
                    Cropped_Trials[int(load_nr)][int(rep_nr_in_load)] = pd.read_json(int_key_dict[load_nr][rep_nr_in_load], orient='index', convert_dates=True)

        del f, fh, fp, files, load_nr, rep_nr_in_load, int_key_dict
    elif Options['evaluation_mode'] == 'entire test':
        data_export_files = []
        data_export_file_names = []
        for (_, _, files) in os.walk(os.path.join(os.getcwd(), 'saved_variables')):
            files = [os.path.join(os.getcwd(), 'saved_variables', f) for f in files if '_bytesIO.txt' in f]
        for f in files:
            with open(f, 'rb') as fh:
                file = BytesIO(fh.read())
                data_export_files.append(file)
                data_export_file_names.append(f.split('\\')[-1].replace('_bytesIO',''))
        with open(os.path.join(os.getcwd(),'saved_variables','subj_name.json'), 'r') as fp:
            subj_name = json.load(fp)
        with open(os.path.join(os.getcwd(),'saved_variables','body_mass.json'), 'r') as fp:
            body_mass = json.load(fp)
        with open(os.path.join(os.getcwd(),'saved_variables','loads_kg.json'), 'r') as fp:
            loads_kg = json.load(fp)
        with open(os.path.join(os.getcwd(),'saved_variables','test_date.json'), 'r') as fp:
            test_date = datetime.date(datetime.strptime(json.load(fp) , '%Y-%m-%d'))# 

        with open(os.path.join(os.getcwd(),'saved_variables','Split_Trials.json'), 'r') as fp:
            int_key_dict = json.load(fp)# Trials = json.load(fp)
            Split_Trials = {}
            for load_nr in int_key_dict.keys():
                Split_Trials[int(load_nr)] = {}
                for rep_nr_in_load in int_key_dict[load_nr].keys():
                    Split_Trials[int(load_nr)][int(rep_nr_in_load)] = pd.read_json(int_key_dict[load_nr][rep_nr_in_load], orient='index', convert_dates=True)
    
        with open(os.path.join(os.getcwd(),'saved_variables','trial_selections.json'), 'r') as fp:
            trial_selections = json.load(fp)
    
        with open(os.path.join(os.getcwd(),'saved_variables','Selected_Trials.json'), 'r') as fp:
            int_key_dict = json.load(fp)# Trials = json.load(fp)
            Selected_Trials = {}
            for load_nr in int_key_dict.keys():
                Selected_Trials[int(load_nr)] = {}
                for rep_nr_in_load in int_key_dict[load_nr].keys():
                    Selected_Trials[int(load_nr)][int(rep_nr_in_load)] = pd.read_json(int_key_dict[load_nr][rep_nr_in_load], orient='index', convert_dates=True)
    
        with open(os.path.join(os.getcwd(),'saved_variables','Trials.json'), 'r') as fp:
            int_key_dict = json.load(fp)# Trials = json.load(fp)
            Trials = {}
            for load_nr in int_key_dict.keys():
                Trials[int(load_nr)] = {}
                for rep_nr_in_load in int_key_dict[load_nr].keys():
                    Trials[int(load_nr)][int(rep_nr_in_load)] = pd.read_json(int_key_dict[load_nr][rep_nr_in_load], orient='index', convert_dates=True)
        
        with open(os.path.join(os.getcwd(),'saved_variables','Cropped_Trials.json'), 'r') as fp:
            int_key_dict = json.load(fp)# Trials = json.load(fp)
            Cropped_Trials = {}
            for load_nr in int_key_dict.keys():
                Cropped_Trials[int(load_nr)] = {}
                for rep_nr_in_load in int_key_dict[load_nr].keys():
                    Cropped_Trials[int(load_nr)][int(rep_nr_in_load)] = pd.read_json(int_key_dict[load_nr][rep_nr_in_load], orient='index', convert_dates=True)
    
        del f, fh, fp, files, file, load_nr, rep_nr_in_load, int_key_dict
    
