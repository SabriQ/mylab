import json
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import pandas as pd
import os,sys
import pickle


def save_pkl(result,result_path):
    with open(result_path,'wb') as f:
        pickle.dump(result,f)
    print("result is saved at %s"% result_path)

def load_pkl(result_path):

    with open(result_path,'rb') as f:
        try:
            result = pd.read_pickle(result_path)    
        except:
            result = pickle.load(f)
            
    # print("result is loaded")
    return result

def load_txt(path):
    with open(path,'r',encoding="utf-8") as f:
        js = f.read()
        reult =  json.loads(js)
    return result

def save_txt(result, path):
    if isinstance(result,str):
        with open(path,'w',encoding="utf-8") as f:
            f.write(json.dumps(result,indent=4))
        print("save: %s" %path)
    else:
        print("result to save is not str")

def _dataframe2nparray(df):
    if isinstance(df,dict):
        print("df is a dict")
        for key in list(df.keys()):
            if isinstance(df[key],pd.core.frame.DataFrame):
                df[str(key)+"_column"]=np.array(df[key].columns)
                df[key]=df[key].values                    
                # print("%s has transferred to numpy array"%key)
            if isinstance(df[key],pd.core.series.Series):
                df[key]=df[key].values
            if isinstance(df[key],dict):
                return _dataframe2nparray(df[key])
        return df
    elif isinstance(df,pd.core.frame.DataFrame):
        print("df is a DataFrame")
        return {"df":df.values,"df_columns":np.array(df.columns)}
    else:
        print(" can not be transferred to nparray")

def spread(a):
    """flattern list with list in it
    """
    return sum((spread(x) if type(x) is list else [x] for x in a), [])

def cdf_detail_xy(cdf_list:list):
    """
    for plot a detailed cdf
    """
    y = 1 * np.arange(len(cdf_list)) / (len(cdf_list) - 1)
    x = cdf_list.copy()
    x.sort()
    return x,y

def savepkl2mat(savematname,result):
    spio.savemat(savematname,_dataframe2nparray(result))
    print("saved %s"%savematname)

def Normalize_list(datalist):
    """
    ??????????????????????????????
    (x-min)/(max-min)
    """
    minimum = np.min(datalist)
    maxum = np.max(data)
    return [(i-minimum)/(maxum-minimum) for i in datalist]
    
def Normalize_df(temp,axis=0):
    """
    ??????????????????????????????
    (x-min)/(max-min)
    temp?????????pd.DataFrame
    axis=0,??????????????????
    axis=1 ??????????????????
    """
    return (temp-temp.min(axis=axis))/(temp.max(axis=axis)-temp.min(axis=axis))

def Standarization(df):
    temp_mean = np.mean(np.reshape(df.values,(1,-1))[0])
    temp_std = np.std(np.reshape(df.values,(1,-1))[0],ddof=1)
    Standarized_df = (df-temp_mean)/temp_std
    print("mean and std", temp_mean,temp_std)
    return Standarized_df,temp_mean,temp_std

def Normalization(df):
    """
    ?????????df??????Normalize?????????????????????????????????????????????????????????
    """
    residual = np.max(np.reshape(df.values,(1,-1))[0])-np.min(np.reshape(df.values,(1,-1))[0])
#     residual = df.max().max()-df.min().min()
    minimum = np.min(np.reshape(df.values,(1,-1))[0])
    normalized_df = (df-minimum)/residual
    print("residual and minimum", residual,minimum)
    return normalized_df,residual,minimum

def Standarization_list(datalist):
    """
    ?????????
    ???x-mean)/std
    """
    mean = np.mean(datalist)
    std = np.std(datalist)
    return [(x-mean)/std for i in datalist]

def crossfoot(point,line_points):
    """
    ??????????????????x0,y0???
    ???????????????????????????[(x1,y1),(x2,y2)]
    ?????? ?????? ??????x,y ???????????????????????????distance_crossfoot,???????????????????????????
    (x,y,distance_crossfoot,min(distance_point1,distance_point2))
    """
    x0,y0 = point
    (x1,y1),(x2,y2) = line_points
    if x1==x2: #???????????????????????????
        x = x1
        y = y0
    else:
        k = (y2-y1)/(x2-x1)
        if k == 0: # ???????????????????????????
            x = x0
            y = y1
        else:
            y = ((k**2)*y0+y1+k*(x0-x1))/((k**2)+1)
            x = x1+(y-y1)/k
    distance_crossfoot = np.sqrt((x0-x)**2+(y0-y)**2)
    distance_point1 = np.sqrt((x0-x1)**2+(y0-y1)**2)
    distance_point2 = np.sqrt((x0-x2)**2+(y0-y2)**2)
    return (x,y,distance_crossfoot,min(distance_point1,distance_point2))

def normalized_distribution_test(datalist):
    """
    ????????????????????????????????????

    ??????????????????????????????
        ????????????????????????????????????????????????????????????,
        ???????????????????????? ????????????????????????wilcoxon_ranksumstest
    ???????????????????????????P??????
        ????????????????????????spearman, kendall
        ?????????????????????pearson (default)
    ????????????????????????????????????
    P?????????0.05???????????????????????????????????????????????????
    ????????????????????????p-value
    """
    return stats.shapiro(datalist)
    
def find_close_fast(arr, e):    
    # start_time = datetime.datetime.now()            
    low = 0    
    high = len(arr) - 1    
    idx = -1     
    while low <= high:        
        mid = int((low + high) / 2)        
        if e == arr[mid] or mid == low:            
            idx = mid            
            break        
        elif e > arr[mid]:            
            low = mid        
        elif e < arr[mid]:            
            high = mid     
    if idx + 1 < len(arr) and abs(e - arr[idx]) > abs(e - arr[idx + 1]):        
        idx += 1            
    # use_time = datetime.datetime.now() - start_time    
    return idx #0????????????

def find_close_fast2(arr,e):
    np.add(arr,e*(-1))
    min_value = min(np.abs(np.add(arr,e*-1)))
    locations = np.where(np.abs(np.add(arr,e*-1))==min_value)
    return locations[0][0]
#    return arr[idx],idx, use_time.seconds * 1000 + use_time.microseconds / 1000

def epoch_detection(trace,tracetime,baseline,thresh,minduration,min_peak_distance=150,show=False):
    """
    which is not ready for process now
    trace: any timeseries data. 
    tracetime: time info, which is the same length of trace. 
    baseline: the start or stop of an transient-like epoch
    thresh: the minimum absolute deviation from baseline, which could be negtive.
    minduration: the minimum duration of one epoch, in the same scale with tracetime
    min_peak_distance: in seconds
    show: to show the detected epoch in red while the trace in black
    """
    trace = np.array(trace)
    tracetime = np.array(tracetime)
    if baseline > thresh:
        trace = trace*-1
        baseline=-1*baseline
        thresh = -1*thresh
        print("detecting valley")
    else:
        print("detecting crest")

   

    points = np.reshape(np.argwhere(trace>baseline),-1)
    starts=[];starts.append(points[0])
    ends=[];ends.append(points[-1])
    for i in range(len(points)-1):
        if not i == 0:
            if points[i-1]-points[i]<-1:
                starts.append(points[i])
        if points[i+1]-points[i]>1:
            ends.append(points[i])

    print(len(starts),len(ends))

    epochs_indexes=[]
    area_under_curves=[]
    for start,end in zip(starts,ends):
        if tracetime[end+1]-tracetime[start]>minduration:
            epochs_indexes.append((start,end+1))
            area_under_curves.append(np.trapz(trace[start,end+1]))
        # calculate area under curve

    # find peaks
    timeconsuming_of_eachframe= (np.max(tracetime)-np.min(tracetime))/len(tracetime)
    peak_indexes=signal(trace,height=thresh,distance=int(min_peak_distance/timeconsuming_of_eachframe))
    

    return epochs_indexes,area_under_curves,peak_indexes


def corr(rsv1,rsv2):
    normalizations = []
    normalization_1 = normalized_distribution_test(rvs1)[1]
    normalization_2 = normalized_distribution_test(rvs2)[1]
    if normalization_1 > 0.05:
        normalizations.append(1)
    else:
        normalizations.append(0)

    if normalization_1 > 0.05:
        normalizations.append(1)
    else:
        normalizations.append(0)

    if all(normalizations):
        print("???????????????????????????")
        return stats.pearsonr(rsv1,rsv2)
    else:
        print("??????????????????????????????????????????%s"%normalizations)
        return stats.spearmanr(rsv1,rsv2)
        # return stats.kendalltau(rvs1,rsv2)

def rlc(x):
    name=[]
    length=[]
    
    for i,c in enumerate(x,0):
        if i ==0:
            name.append(x[0])
            count=1
        elif i>0 and x[i] == name[-1]:
            count += 1
        elif i>0 and x[i] != name[-1]:
            name.append(x[i])
            length.append(count)
            count = 1
    length.append(count)
    return name,length   

def rlc2(X):
    name=[]
    length=[]
    idx_min=[]
    idx_max=[]
    for i,x in enumerate(X,0):
        if i == 0:
            name.append(x)
            idx_min.append(i)
            count =1
        elif i>0 and x==name[-1]:
            count = count +1
        elif i>0 and x!=name[-1]:
            idx_max.append(i)
            idx_min.append(i)
            name.append(x)
            length.append(count)
            count=1
    length.append(count)
    idx_max.append(i)
    df = {"name":name,"length":length,"idx_min":idx_min,"idx_max":idx_max}
    return pd.DataFrame(df)
    

if __name__ == "__main__":
    result = Wilcoxon_ranksumstest([1,2,4,6,8],[3,5,7,1])