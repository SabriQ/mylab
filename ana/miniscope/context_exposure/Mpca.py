from sklearn.decomposition import PCA
from dPCA import dPCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mylab.ana.miniscope.context_exposure.Canamini import AnaMini
import copy
#%% for population analysis
#累计可解释方差贡献率曲线

#降维，得到新的降维后的特征空间矩阵

# n_components 有三种选择模式
## 数字
## mle
## 在(0,1)的数字， svd_solver = "auto",full","arpack","randomized"

# 返还原空间用于数据降噪inverse_transform

def construct_PCA_matrix(s:AnaMini):
    """

    """
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num(place_bin_nums=[4,4,30,4,4,4])
    s.add_Context()
    df,index = s.trim_df("S_dff",placebin = np.arange(0,50))
    
    Trial_Num = s.result["Trial_Num"]
    process = s.result["process"]
    place_bin_No = copy.deepcopy(s.result["place_bin_No"])
    # 将backward的place_bin_No 反向增加
    max_placebin = 49
    for i in place_bin_No[(process>3) | (process==0)].index:
        place_bin_No[i] = 2*max_placebin-place_bin_No[i]+1

    Trial_Num=Trial_Num[index]
    process=process[index]
    Context= s.result["Context"][index]


    x = df[index].groupby([Context,place_bin_No]).mean()

    return x


def pca(x,**kargws):
    """
    x: dataframe with "context" and "placebins" as two of the multi-indexes
    """

    x_arr = x.values
    
    context_0_index = x.index.get_level_values(level="Context")==0
    context_1_index = x.index.get_level_values(level="Context")==1
    context_2_index = x.index.get_level_values(level="Context")==2

    pca = PCA(**kargws)
    pca = pca.fit(x_arr)
    x_dr = pca.transform(x_arr)

    fig,axes = plt.subplots(3,3,figsize=(20,15))
    axes[0,0].plot(np.arange(0,10),np.cumsum(pca.explained_variance_ratio_)[0:10])
    axes[0,0].scatter(np.arange(0,10),np.cumsum(pca.explained_variance_ratio_)[0:10],s=15)
    axes[0,0].scatter(np.arange(0,10),pca.explained_variance_ratio_[0:10],s=15)
    axes[0,0].set_xticks(np.arange(0,10))
    axes[0,0].set_title("PCA-analysis")
    axes[0,0].set_xlabel("number of components after dimension reduction")
    axes[0,0].set_ylabel("cumulative explained variance ratio")

    spatial_points={
    "nosepoke": 0,
    "turnover_1": 3,
    "context_enter": 7,
    "context_exit": 37,
    "turnover_2": 41,
    "choice1": 45,
    "choice2": 49,
    "turnover_3": 57,
    "context_reverse_enter": 61,
    "context_reverse_exit":91,
    "turnover_4": 95,
    "trial_end": 99}

    for ax,c in zip(axes.flat[1:],[0,1,2,3,4,5,6,7]):
        for ctx in [context_0_index,context_1_index]:
            ax.plot(np.arange(0,len(x_dr[ctx,:])),x_dr[ctx,:][:,c])
            ax.legend(["ctx0","ctx1"])
        for point in spatial_points.keys(): # for different point
            if "choice" in point:
                color = "red"
            elif "turn" in point:
                color = "green"
            elif "context" in point:
                color = "orange"
            else:
                color = "black"
            ax.axvline(spatial_points[point],linestyle="--",color=color)
        ax.set_title("PC%s"%(c+1))
        ax.set_xlabel("Placebins")
    plt.tight_layout()
    plt.show()
    
    result = {
    "x_dr":x_dr,
    "pca":pca,
    "context_0_index":context_0_index,
    "context_1_index":context_1_index,
    "context_2_index":context_2_index
    }

    return result





def construct_dPCA_matrix(s:AnaMini,):
    """
    construct matrix for demixed OCA according to 
        [max_Trial_Num, neurons, placebin context decision]

    """
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num(according="Body",place_bin_nums=[4,4,30,4,4,4])
    s.add_Context()

    # ========================
    df,index = s.trim_df("S_dff",placebin = np.arange(0,50))
    Trial_Num = s.result["Trial_Num"]
    process = s.result["process"]
    place_bin_No = copy.deepcopy(s.result["place_bin_No"])
    # 将backward的place_bin_No 反向增加
    max_placebin = 49
    for i in place_bin_No[(process>3) | (process==0)].index:
        place_bin_No[i] = 2*max_placebin-place_bin_No[i]+1

    Trial_Num=Trial_Num[index]
    process=process[index]
    Context= s.result["Context"][index]
    dpca_matrix = pd.DataFrame(df).groupby([Trial_Num,place_bin_No]).mean()

    s.add_behave_choice_side()
    s.add_behave_forward_context(according="Enter_ctx")

    decisions = set(s.result["behave_choice_side"])
    contexts  =  set(s.result["behave_forward_context"])
    cells = list(s.df.columns)

    trial_df = pd.DataFrame()
    trial_df["context"] = s.result["behave_forward_context"]
    trial_df["decision"] = s.result["behave_choice_side"]
    trial_df["reward"] = s.result["behavelog_info"]["Choice_class"]
    trial_df["Trial"] = s.result["behavelog_info"]["Trial_Num"]
    trial_dict = {}
    for context in contexts:
        for decision in decisions:
            trial_dict["context_%s_decison_%s"%(context,decision)]= list(trial_df["Trial"][(trial_df["context"]==context) & (trial_df["decision"]==decision)])

    max_Trial_Num = np.max([len(i) for i in trial_dict.values()])
    print("max_Trial_Num:%s,neurons:%s,context:%s,decision:%s,placebin:%s."%(max_Trial_Num,len(cells),len(contexts),len(decisions),100))
    #构建 空的np.arrays [max_Trial_Num,cell_Num,context_num,decision_Num,placebin_num]
    trialR = np.full((max_Trial_Num,len(cells),len(contexts),len(decisions),100),np.nan)
    for c,context in enumerate(contexts):
        for d,decision in enumerate(decisions):
            for i,t in enumerate(trial_dict["context_%s_decison_%s"%(context,decision)],0):
                temp = dpca_matrix.loc[t]
                for p in temp.index:
                    trialR[i,:,c,d,p]=temp.loc[p]

    if len(decisions)==1:
        trialR = np.squeeze(trialR)
        labels = 'cp'
        join = {'c': ['c', 'cp'], 'p':['p']}
    else:
        labels = 'cdp'
        join = {'c': ['c', 'cp'],'d': ['d', 'dp'], 'p':['p'], 'cd': ['cd','cdp']}

    #先求没有np.nan的均值，然后再将np.nan替换为0
    R = np.nanmean(trialR,axis=0)
    trialR[np.isnan(trialR)]=0
    R[np.isnan(R)]=0

    return R,trialR,labels,join

def _construct_dPCA_trialR(s:AnaMini,contexts=[0,1],decisions=["left"],rewards=[0,1],according="rewards"):
    """
    which is for construct_dPCA_matrixs
    Arguments:
        contexts: only context location in contexts are under consideration
        decisions: only decision in decisions are under consideration
    """
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num(according="Body",place_bin_nums=[4,4,30,4,4,4])
    s.add_Context()
    # ========================
    df,index = s.trim_df("S_dff",placebin = np.arange(0,50))
    Trial_Num = s.result["Trial_Num"]
    process = s.result["process"]
    place_bin_No = copy.deepcopy(s.result["place_bin_No"])

    # 将backward的place_bin_No 反向增加
    max_placebin = 49
    for i in place_bin_No[(process>3) | (process==0)].index:
        place_bin_No[i] = 2*max_placebin-place_bin_No[i]+1
    Trial_Num=Trial_Num[index]
    process=process[index]
    Context= s.result["Context"][index]
    # construct dpca_matrix with indexes [Trial_Num, place_bin_No]
    dpca_matrix = pd.DataFrame(df).groupby([Trial_Num,place_bin_No]).mean()

    s.add_behave_choice_side()
    s.add_behave_forward_context(according="Enter_ctx")

    decisions = set(s.result["behave_choice_side"]) if decisions is None else decisions# set like {"left"}
    decisions = sorted(decisions)
    contexts  =  set(s.result["behave_forward_context"]) if contexts is None else contexts # only context locations in Argument are under consideration
    contexts = sorted(contexts)
    rewards = set(s.result["behavelog_info"]["Choice_class"]) if rewards is None else rewards
    rewards = sorted(rewards)
    cells = list(s.df.columns)


    # construct behave log info as trial_df with columns ["context","decision","reward","Trial"]

    trial_df = pd.DataFrame()
    trial_df["context"] = s.result["behave_forward_context"]
    trial_df["decision"] = s.result["behave_choice_side"]
    trial_df["reward"] = s.result["behavelog_info"]["Choice_class"]
    trial_df["Trial"] = s.result["behavelog_info"]["Trial_Num"]

    # classify trials as a dict trial_dict with keys like "context_0_decison_left", "context_1_decison_left"
    trial_dict = {}
    for context in contexts:
        if according == "decisions":
            pass
        if according == "rewards":
            decisions == rewards
        for decision in decisions:
            trial_dict["context_%s_decison_%s"%(context,decision)]= list(trial_df["Trial"][(trial_df["context"]==context) & (trial_df["decision"]==decision)])

    # calculate the max_Trial_Num which is needed to construct the blank numpy arrays
    max_Trial_Num = np.max([len(i) for i in trial_dict.values()])
    print("max_Trial_Num:%s,neurons:%s,context:%s,decision:%s,placebin:%s."%(max_Trial_Num,len(cells),len(contexts),len(decisions),100))

    #构建 空的np.arrays [max_Trial_Num,cell_Num,context_num,decision_Num,placebin_num]
    trialR = np.full((max_Trial_Num,len(cells),len(contexts),len(decisions),100),np.nan)

    for c,context in enumerate(contexts):
        for d,decision in enumerate(decisions):
            for i,t in enumerate(trial_dict["context_%s_decison_%s"%(context,decision)],0):
                temp = dpca_matrix.loc[t] # index the fr in dpca_matrix according to trial_Num
                temp = (temp-temp.mean(axis=0))/temp.std(axis=0) # apply z-score for each neuron along different place bins for this trial
                for p in temp.index:
                    trialR[i,:,c,d,p]=temp.loc[p]

    return trialR

def construct_dPCA_matrixs(ss:list,contexts_list,decisions_list,rewards_list,according="decisions"):

    trialRs = []
    for s,contexts,decisions,rewards in zip(ss,contexts_list,decisions_list,rewards_list):
        trialRs.append(_construct_dPCA_trialR(s,contexts,decisions,rewards,according))

    # max_Trial_Num, cells,contexts,decisions,placebins
    dims = []    
    for trialR in trialRs:
        dims.append(trialR.shape)
    # the max_Trial_Num of all sessions
    max_Trial_Num = np.max(np.array(dims)[:,0])

    # we need the context_num decisions_num and place_bin_Num is the same for different matrix
    if not (np.array(dims)[:,2] == np.array(dims)[:,2][0]).all(): # context_num
        print("contexts number is different in different sessions")
        return
    if not (np.array(dims)[:,3] == np.array(dims)[:,3][0]).all(): # decision_num
        print("decisions number is different in different sessions")
        return
    if not (np.array(dims)[:,4] == np.array(dims)[:,4][0]).all(): # placebin_num
        print("placebin number is different in different sesssions")
        return

    newtrialRs=[]
    for trialR in trialRs:
        if trialR.shape[0] < max_Trial_Num:
            # if the Trial_num of trialR is less than max_Trial_Num, then construct a new trialR with the max_Trial_Num and fill it with the original trialR
            newtrialR = np.full((max_Trial_Num,*trialR.shape[1:]),np.nan)
            for i in range(trialR.shape[0]): #traverse the first dimension which is the Trial_Num
                newtrialR[i] = trialR[i]
            newtrialRs.append(newtrialR)
        else:
            newtrialRs.append(trialR)

    concatenated_trialR = np.concatenate(newtrialRs,axis=1) # concatenate along cells
    # if length of the decision dimension is 1 the squeeze it
    if concatenated_trialR.shape[3]==1:
        concatenated_trialR = np.squeeze(concatenated_trialR)
        labels = 'cp'
        join = {'c': ['c', 'cp'], 'p':['p']}
    else:
        labels = 'cdp'
        join = {'c': ['c', 'cp'],'d': ['d', 'dp'], 'p':['p'], 'cd': ['cd','cdp']}

    #先求没有np.nan的均值，然后再将np.nan替换为0
    R = np.nanmean(concatenated_trialR,axis=0)

    concatenated_trialR[np.isnan(concatenated_trialR)]=0
    R[np.isnan(R)]=0
    # #center data
    # if len(R.shape)==3:
    #     R = R- np.mean(R.reshape((R.shape[0],-1)),1)[:,None,None]
    # if len(R.shape)==4:
    #     R = R- np.mean(R.reshape((R.shape[0],-1)),1)[:,None,None,None]
    return R,concatenated_trialR,labels,join

def demixed_pca(R,trialR,labels,join,**kwargs):
    """
    R
    trialR
    labels
    join
    """

    dpca = dPCA.dPCA(labels=labels,join=join,regularizer='auto',**kwargs) #,n_components=50
    dpca.protect = ['p']

    dpca = dpca.fit(R,trialR)
    Z = dpca.transform(R,marginalization=None)
    significance_masks = dpca.significance_analysis(X = R,trialX=trialR ,n_shuffles=20, n_splits=10, n_consecutive=10)

    result = {
    "dpca":dpca,
    "Z":Z,
    "significance_masks":significance_masks
    }
    return result

def plot_dPCA(result):
    dpca=result["dpca"]
    Z = result["Z"]
    significance_masks=result["significance_masks"]

    placebins = np.arange(100)
    spatial_points={
    "nosepoke": np.where(placebins==0),
    "turnover_1": np.where(placebins==3),
    "context_enter": np.where(placebins==7),
    "context_exit": np.where(placebins==37),
    "turnover_2": np.where(placebins==41),
    "choice1": np.where(placebins==45),
    "choice2": np.where(placebins==49),
    "turnover_3": np.where(placebins==57),
    "context_reverse_enter": np.where(placebins==61),
    "context_reverse_exit": np.where(placebins==91),
    "turnover_4": np.where(placebins==95),
    "trial_end": np.where(placebins==99)}

    components = Z.keys()
    components_len = len(components)
    plt.figure(figsize=(30,4*components_len),dpi=100)
    # plt.xkcd()
    # plt.rc('font',family='Arial')
    for n,comp in enumerate(components,0):
        for i in range(3):
            plt.subplot(components_len,3,3*n+i+1)
            for c in range(len(Z[comp][i])):
                plt.plot(placebins,Z[comp][i,c])
            plt.legend(("context0","context1"))
            if n==0:
                plt.ylim(-1,1)
            for point in spatial_points.keys():
                if "choice" in point:
                    color = "red"
                elif "turn" in point:
                    color = "green"
                elif "context" in point:
                    color = "orange"
                else:
                    color = "black"
                plt.axvline(spatial_points[point][0][0],color=color,linestyle="--")
            plt.tight_layout()
            if i == 0:
                plt.ylabel(comp,size=50)
            if n==components_len-1:
                plt.xlabel("placebins",size=40)
    plt.show()

def dPCA_to_delete(s:AnaMini,**kwargs):
    """
    """
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num(according="Body",place_bin_nums=[4,4,30,4,4,4])


    # ========================
    df,index = s.trim_df("S_dff",placebin = np.arange(0,50))
    Trial_Num = s.result["Trial_Num"]
    process = s.result["process"]
    place_bin_No = copy.deepcopy(s.result["place_bin_No"])
    # 将backward的place_bin_No 反向增加
    max_placebin = 49
    for i in place_bin_No[(process>3) | (process==0)].index:
        place_bin_No[i] = 2*max_placebin-place_bin_No[i]+1
    Trial_Num=Trial_Num[index]
    process=process[index]
    Context= s.result["Context"][index]


    dpca_matrix = pd.DataFrame(df).groupby([Trial_Num,place_bin_No]).mean()

    s.add_behave_choice_side()
    s.add_behave_forward_context(according="Enter_ctx")

    ctx0_choice0_trial = s.result["behavelog_info"]["Trial_Num"][(context==0)&(choice_side==0)]
    ctx0_choice1_trial = s.result["behavelog_info"]["Trial_Num"][(context==0)&(choice_side==1)]

    ctx1_choice0_trial = s.result["behavelog_info"]["Trial_Num"][(context==1)&(choice_side==0)]
    ctx1_choice1_trial = s.result["behavelog_info"]["Trial_Num"][(context==1)&(choice_side==1)]


    # 判断以下变量的size
    # max trial num in each condition(contexts)
    max_Trial_Num = max(ctx0_choice0_trial.shape[0],ctx0_choice1_trial.shape[0],ctx1_choice0_trial.shape[0],ctx1_choice1_trial.shape[0])
    neurons = dpca_matrix.shape[1]
    placebin = len(set(s.result["place_bin_No"]))
    context=2 if len(set(context)) > 1 else None
    decision=2 if len(set(context)) > 1 else None

    print("max_Trial_Num:%s,neurons:%s,context:%s,decision:%s,placebin:%s."%(max_Trial_Num,neurons,context,decision,placebin))
    #构建 空的np.array
    trialR = np.full((max_Trial_Num,neurons,context,decision,placebin),np.nan)


    # 填充矩阵，其中没有值的地方都是np.nan
    for i,trial in enumerate(ctx0_choice0_trial,0):
        temp_trial_matrix = dpca_matrix.loc[trial]
        for p in np.arange(0,placebin):
            try:
                trialR[i,:,0,0,p]=temp_trial_matrix.loc[p]
            except:
                trialR[i,:,0,0,p]=0 # 将np.nan替换为0
        
    for i,trial in enumerate(ctx0_choice1_trial,0):
        temp_trial_matrix = dpca_matrix.loc[trial]
        for p in np.arange(0,placebin):
            try:
                trialR[i,:,0,1,p]=temp_trial_matrix.loc[p]
            except:
                trialR[i,:,0,1,p]=0 # 将np.nan替换为0
                
    for i,trial in enumerate(ctx1_choice0_trial,0):
        temp_trial_matrix = dpca_matrix.loc[trial]
        for p in np.arange(0,placebin):
            try:
                trialR[i,:,1,0,p]=temp_trial_matrix.loc[p]
            except:
                trialR[i,:,1,0,p]=0 # 将np.nan替换为0
                
    for i,trial in enumerate(ctx1_choice1_trial,0):
        temp_trial_matrix = dpca_matrix.loc[trial]
        for p in np.arange(0,placebin):
            try:
                trialR[i,:,1,1,p]=temp_trial_matrix.loc[p]
            except:
                trialR[i,:,1,1,p]=0 # 将np.nan替换为0

    #先求没有np.nan的均值，然后再将np.nan替换为0
    R = np.nanmean(trialR,axis=0)
    # trialR[np.isnan(trialR)]=0

    result = demixed_pca(R,trialR,**kwargs)

    return result
