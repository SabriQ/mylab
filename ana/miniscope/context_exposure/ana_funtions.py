
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import os,sys,glob,re,copy
import scipy.io as spio
import pickle
import itertools
import scipy.stats as stats
from mylab.ana.miniscope.Mplacecells import *
from mylab.ana.miniscope.context_exposure.Mpca import *
from mylab.ana.miniscope.context_exposure.Canamini import AnaMini
from sklearn.linear_model import LogisticRegressionCV as LRCV


#%% for single cell analysis

def cellid_Context(s:AnaMini,*args,**kwargs):
    """
    s.add_Context()
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num(placebins=[4,4,30,4,4,4])
    """
    print("FUNC::cellids_Context")
    df,index = s.trim_df(*args,**kwargs)
    df = df[index]


    Trial_Num = s.result["Trial_Num"][index]
    Context= s.result["Context"][index]

    meanfr_df = df.groupby([Trial_Num,Context]).mean()
    ctx_meanfr = meanfr_df.groupby(["Context"]).mean()

    result = {}
    result["meanfr_df"] = meanfr_df
    result["ctx_meanfr"] = ctx_meanfr

    if len(set(Context)) > 1:
        for ctxes in itertools.combinations(set(Context),2):
            a,b = ctxes
            idx_a = meanfr_df.index.get_level_values(level="Context")==a
            idx_b = meanfr_df.index.get_level_values(level="Context")==b
            ctx_pvalue = meanfr_df.apply(func=lambda x: stats.ranksums(x[idx_a],x[idx_b])[1],axis=0)
            CSI = (ctx_meanfr.loc[a,:]-ctx_meanfr.loc[b,:])/(ctx_meanfr.loc[a,:]+ctx_meanfr.loc[b,:])

            ContextA_cells=[]
            ContextB_cells=[]
            non_context_cells=[]

            for cellid,csi, p in zip(ctx_meanfr.columns,CSI,ctx_pvalue):
                if p>=0.05:
                    non_context_cells.append(cellid)
                else:
                    if csi>0:
                        ContextA_cells.append(cellid)
                    elif csi<0:
                        ContextB_cells.append(cellid)
                    else:
                        print("meanfr of cell %s is equal in Context %s and Context %s"%(cellid,a,b))
                        non_context_cells.append(cellid)

            result["ctx%s_%s"%(a,b)]={
            "ctx_pvalue":ctx_pvalue,
            "CSI":CSI,
            "context%s_cells"%a:ContextA_cells,
            "context%s_cells"%b:ContextB_cells,
            "non_context_cells":non_context_cells
            }


    return result

def cellid_RD_incontext(s:AnaMini,*args,**kwargs):
    """
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num()
    s.add_Context
    s.add_Body_speed(scale=0.33)
    s.add_running_direction(self,according="Body")
    """
    print("FUNC::cellid_RD_incontext")
    df,index = s.trim_df(*args,**kwargs)
    df = df[index]
    Trial_Num = s.result["Trial_Num"][index]
    Context= s.result["Context"][index]
    in_context_running_direction=s.result["running_direction"][index]

    meanfr_df=df.groupby([Trial_Num,in_context_running_direction]).mean()

    rd_meanfr=meanfr_df.groupby(["running_direction"]).mean()
    result = {}

    result["meanfr_df"] = meanfr_df

    idx_0 = meanfr_df.index.get_level_values(level="running_direction")==0
    idx_1 = meanfr_df.index.get_level_values(level="running_direction")==1
    pvalue = meanfr_df.apply(func=lambda x: stats.ranksums(x[idx_0],x[idx_1])[1],axis=0)
    RDSI = (rd_meanfr.loc[0,:]-rd_meanfr.loc[1,:])/(rd_meanfr.loc[0,:]+rd_meanfr.loc[1,:])
    left_cells=[]
    right_cells=[]
    non_rd_cells=[]
    for cellid,p,si in zip(meanfr_df.columns,pvalue,RDSI):
        if p>=0.05:
            non_rd_cells.append(cellid)
        else:
            if si > 0:
                left_cells.append(cellid)
            elif si<0:
                right_cells.append(cellid)
            else:
                non_rd_cells.append(cellid)
                print("meanfr of cell %s is equal in two runnin directions"%cellid)
    result={
    "pvalue":pvalue,
    "RDSI":RDSI,
    "left_cells":left_cells,
    "right_cells":right_cells,
    "non_rd_cells":non_rd_cells
    }

    return result

def cellid_RD2_incontext(s:AnaMini,*args,**kwargs):
    """
    cellid_RD2_incontext: return left or right cell in different contexts
    s.align_behave_ms()
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num()
    s.add_Context
    s.add_Body_speed(scale=0.33)
    s.add_running_direction(self,according="Body")
    """
    print("FUNC::cellid_RD_incontext")
    df,index = s.trim_df(*args,**kwargs)
    df = df[index]
    Trial_Num = s.result["Trial_Num"][index]
    Context= s.result["Context"][index]
    in_context_running_direction=s.result["running_direction"][index]

    meanfr_df=df.groupby([Trial_Num,Context,in_context_running_direction]).mean()

    ctx_rd_meanfr=meanfr_df.groupby(["Context","running_direction"]).mean()
    result = {}

    result["meanfr_df"] = meanfr_df
    result["ctx_rd_meanfr"] = ctx_rd_meanfr

    for ctx in set(Context):
        ctx_idx = meanfr_df.index.get_level_values(level="Context")==ctx
        idx_0 = meanfr_df[ctx_idx].index.get_level_values(level="running_direction")==0
        idx_1 = meanfr_df[ctx_idx].index.get_level_values(level="running_direction")==1
        ctx_rd_pvalue = meanfr_df[ctx_idx].apply(func=lambda x: stats.ranksums(x[idx_0],x[idx_1])[1],axis=0)
        ctx_rd_RDSI = (ctx_rd_meanfr.loc[(ctx,0),:]-ctx_rd_meanfr.loc[(ctx,1),:])/(ctx_rd_meanfr.loc[(ctx,0),:]+ctx_rd_meanfr.loc[(ctx,1),:])
        left_cells=[]
        right_cells=[]
        non_rd_cells=[]
        for cellid, p, si in zip(meanfr_df.columns,ctx_rd_pvalue,ctx_rd_RDSI):
            if p>=0.05:
                non_rd_cells.append(cellid)
            else:
                if si > 0:
                    left_cells.append(cellid)
                elif si<0:
                    right_cells.append(cellid)
                else:
                    non_rd_cells.append(cellid)
                    print("meanfr of cell %s is equal in two runnin directions"%cellid)

        result["context_%s"%ctx] = {
        "pvalue":ctx_rd_pvalue,
        "RDSI":ctx_rd_RDSI,
        "left_cells":left_cells,
        "right_cells":right_cells,
        "non_rd_cells":non_rd_cells
        }

    return result


def cellid_PC_incontext(s:AnaMini,*args,shuffle_times=1000,**kwargs):
    """
    return Place cells that existed in all contexts

    s.align_behave_ms()
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num(according="Head",place_bin_nums=[4,4,30,4,4,4],behavevideo)
    s.add_Body_speed(scale=0.33)
    """
    print("FUNC::cellid_PC_incontext")
    df,index = s.trim_df(*args,**kwargs)
    df=df[index]
    Trial_Num = s.result["Trial_Num"][index]

    in_context_placebin_num = s.result["place_bin_No"][index]

    
    result={
    "observed_SIs":Cal_SIs(df,in_context_placebin_num),
    "shuffle_func":bootstrap_Cal_SIs(df,in_context_placebin_num),
    "shuffle_SIs":[]
    }

    for i in range(shuffle_times):
        sys.stdout.write("%s/%s"%(i+1,shuffle_times))
        sys.stdout.write("\r")
        result["shuffle_SIs"].append(result["shuffle_func"]().values)

 
    del result["shuffle_func"]
    result["shuffle_SIs"] = pd.DataFrame(result["shuffle_SIs"],columns=df.columns)
    result["zscore"] = (result["observed_SIs"]-result["shuffle_SIs"].mean())/result["shuffle_SIs"].std()
    result["place_cells"] = result["zscore"][result["zscore"]>1.96].index.tolist()

    return result

def cellid_PC2_incontext(s:AnaMini,*args,shuffle_times=1000,**kwargs):
    """
    return place cells that existed in different contexts

    s.align_behave_ms()
    s.add_Trial_Num_Process()
    s.add_alltrack_placebin_num(according="Head",place_bin_nums=[4,4,30,4,4,4],behavevideo)
    s.add_Body_speed(scale=0.33)
    """
    print("FUNC::cellid_PC_incontext")
    df,index = s.trim_df(*args,**kwargs)
    df=df[index]
    Trial_Num = s.result["Trial_Num"][index]
    Context= s.result["Context"][index]

    in_context_placebin_num = s.result["place_bin_No"][index]


    result = {}
    for ctx in set(Context):
        result["context_%s"%ctx]={
        "observed_SIs":Cal_SIs(df[Context==ctx],in_context_placebin_num[Context==ctx]),
        "shuffle_func":bootstrap_Cal_SIs(df[Context==ctx],in_context_placebin_num[Context==ctx]),
        "shuffle_SIs":[]
        }


    for i in range(shuffle_times):
        sys.stdout.write("%s/%s"%(i+1,shuffle_times))
        sys.stdout.write("\r")
        for ctx in set(Context):
            result["context_%s"%ctx]["shuffle_SIs"].append(result["context_%s"%ctx]["shuffle_func"]().values)
    for ctx in set(Context):
        del result["context_%s"%ctx]["shuffle_func"]
        result["context_%s"%ctx]["shuffle_SIs"] = pd.DataFrame(result["context_%s"%ctx]["shuffle_SIs"],columns=df.columns)
        result["context_%s"%ctx]["zscore"] = (result["context_%s"%ctx]["observed_SIs"]-result["context_%s"%ctx]["shuffle_SIs"].mean())/result["context_%s"%ctx]["shuffle_SIs"].std()
        result["context_%s"%ctx]["place_cells"] = result["context_%s"%ctx]["zscore"][result["context_%s"%ctx]["zscore"]>1.96].index.tolist()

    return result



# heatmap and line map
def SingleCell_MeanFr_in_SingleTrial_along_Placebin(s:AnaMini,*args,**kwargs) :    
    """
    generate a dict contains a matrix of each context
    the structure of matrix is [len(cellids),len(place_bins),len(trials)]
    这里 不在函数内使用 add*函数，是因为多个函数一块使用的场景可以把这些步骤放到外面节省时间
    s.add_Trial_Num_Process()
    s.add_Context()
    s.add_alltrack_placebin_num(place_bin_nums=[4,4,30,4,4,4])
    s.add_Body_speed(scale=0.33)

    s.add_behave_forward_context(according="Enter_ctx")
    s.add_behave_choice_side()
    s.add_behave_reward()
    """
    # 添加需要的数据
    
    print("FUNC::SingleCell_MeanFr_in_SingleTrial_along_Placebin")

    df,index = s.trim_df(*args,**kwargs)
    df = df[index]

    Trial_Num = s.result["Trial_Num"]
    process = s.result["process"]
    place_bin_No = copy.deepcopy(s.result["place_bin_No"])
    speed = pd.Series(s.result["Body_speed"],name="Body_speed")
    # 将backward的place_bin_No 反向增加
    max_placebin = 49
    for i in place_bin_No[(process>3) | (process==0)].index:
        place_bin_No[i] = 2*max_placebin-place_bin_No[i]+1

    Trial_Num=Trial_Num[index]
    process=process[index]
    Context= s.result["Context"][index]    
    speed = speed[index]


    meanspeed = speed.groupby([Trial_Num,Context,place_bin_No]).mean()
    

    #  meanfr by untrimmed "Trial_Num","Context","place_bin_No"
    # meanfr = df.groupby([s.result["Trial_Num"],s.result["Context"],s.result["place_bin_No"]]).mean()
    meanfr = df.groupby([Trial_Num,Context,place_bin_No]).mean()
    meanfr.index.names = ['Trial_Num', 'Context', 'place_bin_No']

    cellids = s.result["idx_accepted"]
    Context_Matrix_info = {}
    Context_Matrix_info["cellids"] = cellids
    Context_Matrix_info["mouse_id"] = s.result["mouse_id"][0]
    Context_Matrix_info["part"] = s.result["part"][0]
    Context_Matrix_info["index"] = s.result["index"][0]
    Context_Matrix_info["aim"] = s.result["aim"][0]

    # Context_Matrix_info["place_bins"] = np.unique(meanfr.index.get_level_values("place_bin_No")) 
    Context_Matrix_info["place_bins"] = np.arange(0,100)

    ###
    trials =np.unique(meanfr.index.get_level_values("Trial_Num"))
    place_bins = Context_Matrix_info["place_bins"]
    matrix = np.full([len(cellids),len(place_bins),len(trials)],np.nan)
    matrix_meanspeed = np.full([len(place_bins),len(trials)],np.nan)
    for i,t in enumerate(trials):
        for j,p in enumerate(place_bins):
            try:
                matrix[:,j,i] = meanfr.xs((t,p),level=["Trial_Num","place_bin_No"]).values
            except:
                pass
            try:
                matrix_meanspeed[j,i] = meanspeed.xs((t,p),level=["Trial_Num","place_bin_No"]).values
            except:
                pass

    Context_Matrix_info["Matrix_cellids_placebins_trials"] = matrix
    Context_Matrix_info["meanspeed"]=matrix_meanspeed

    Context_Matrix_info["trials"] = {}
    for c in set(Context):
        trials = np.unique(meanfr.xs(c,level="Context").index.get_level_values("Trial_Num"))
        Context_Matrix_info["trials"]["context%s"%c] = list(trials)

    Context_Matrix_info["trials"]["left_right"] = []
    Context_Matrix_info["trials"]["right_right"] = []
    Context_Matrix_info["trials"]["left_wrong"] = []
    Context_Matrix_info["trials"]["right_wrong"]= []

    for i,choice in enumerate(zip(s.result["behave_choice_side"],s.result["behave_reward"]),1):
        if choice == ('left',1):
            Context_Matrix_info["trials"]["left_right"].append(i)
        elif choice == ('left',0):
            Context_Matrix_info["trials"]["left_wrong"].append(i)
        elif choice == ('right',1):
            Context_Matrix_info["trials"]["right_right"].append(i)
        elif choice == ('right',0):
            Context_Matrix_info["trials"]["right_wrong"].append(i)
        else:
            print("unexpected trial type")

    return Context_Matrix_info

def plot_MeanFr_along_Placebin(Context_Matrix_info:dict,idx,placebins:list=None,save=False,format="png",show=True,savedir=None):

    """
    Arguments:
        Context_Matrix_info: the output of SingleCell_MeanFr_in_SingleTrial_along_Placebin
    Returns:
        plot single cell MeanFr and save as png. No return
    """
    if save:
        if savedir is None:
            savedir = os.getcwd()
        filename = "mouse%sid%spart%sindex%saim%s.png"%(Context_Matrix_info["mouse_id"],idx,Context_Matrix_info["part"],Context_Matrix_info["index"],Context_Matrix_info["aim"])
        savepath = os.path.join(savedir,filename)
        if os.path.exists(savepath):
            print("existed,Jump!")
            return 
        else:
            print("plotting")

    Matrix_cellids_placebins_trials = Context_Matrix_info["Matrix_cellids_placebins_trials"]
    trialtypes = Context_Matrix_info["trials"]
    
    if idx in Context_Matrix_info["cellids"]:
        dim1 = np.where(Context_Matrix_info["cellids"]==idx)[0][0]
    else:
        print("Cell %s doesn't exist"%idx)
        return 
    
    
    placebins = Context_Matrix_info["place_bins"] if placebins is None else placebins
    try:
        dim2 = np.array([np.where(Context_Matrix_info["place_bins"]==i)[0][0] for i in placebins])
    except:        
        dim2 = np.array(Context_Matrix_info["place_bins"])
        print("the placebins you specified is out of index:%s"%dim2)
    # print(dim2)    
    placebins = np.array(placebins)
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
    
    n_type = len(trialtypes)

    fig = plt.figure(figsize=(10,n_type*4),dpi=300)
    plt.rc('font',family='Arial')
    
    
    for i,trialtype in enumerate(trialtypes.keys(),1):
        # 根据条件去对应type 的trials
        dim3 = np.array(Context_Matrix_info["trials"][trialtype])-1
        if len(dim3)==0:
            continue
        matrix = Matrix_cellids_placebins_trials[dim1,:,:]
        matrix = matrix[dim2,:]
        matrix = matrix[:,dim3] # placebins * trials
        
        matrix_speed = Context_Matrix_info["meanspeed"][:,dim3][dim2,:]
        
        #mean 所有的trials
        matrix_mean = np.nanmean(matrix,axis=1) 
        matrix_speed_mean = np.nanmean(matrix_speed,axis=1)
        #sem 所有的trials
        matrix_sem = np.nanstd(matrix,axis=1,ddof=1)/np.sqrt(matrix.shape[1]) 
        matrix_speed_sem = np.nanstd(matrix_speed,axis=1,ddof=1)/np.sqrt(matrix_speed.shape[1]) 

        # matrix_and_mean = np.row_stack((matrix,matrix_mean)) # 在最后一行合并评论值
        
        ## axis =1 每一行的placebins进行 standarization ,每一行是一个trial
        matrix_standarization = np.apply_along_axis(lambda x:(x-np.nanmean(x))/np.nanstd(x,ddof=1)
                                                   ,axis=0
                                                   ,arr=matrix)
        matrix_standarization[np.isnan(matrix_standarization)]=0
        if np.isnan(matrix_standarization).all():
            print("cell has no firing")
            plt.close("all")
            continue
        matrix_standarization_mean = np.nanmean(matrix_standarization,axis=1)
        matrix_standarization_sem = np.nanstd(matrix_standarization,axis=1,ddof=1)/np.sqrt(matrix_standarization.shape[1]) 
        

        plt.subplot(n_type+1,2,i*2-1)
        sns.heatmap(matrix_standarization.T)
        # plt.imshow(matrix_standarization.T)
        plt.ylabel("Trials")
        plt.xlabel("Placebins")
        plt.title("mouse:%s-id:%s in %s"%(Context_Matrix_info["mouse_id"],idx,trialtype))
        
        ax1 = fig.add_subplot(n_type+1,2,i*2)
        line1, = ax1.plot(np.arange(0,len(matrix_mean)),matrix_standarization_mean,color="blue",linestyle="--",label="Fr")
        # ax1.legend([line1],["Fr"])
        ax1.fill_between(np.arange(0,len(matrix_mean))
                         ,matrix_standarization_mean-matrix_standarization_sem
                         ,matrix_standarization_mean+matrix_standarization_sem
                        ,color="blue"
                        ,alpha=0.3
                        ,edgecolor="white")

        for point in spatial_points.keys():
            try:
                if "choice" in point:
                    color = "red"
                elif "turn" in point:
                    color = "green"
                elif "context" in point:
                    color = "orange"
                else:
                    color = "black"
                ax1.axvline(spatial_points[point][0][0],color=color,linestyle="--")
            except:
                pass
        ax1.set_ylabel("Z-score(MeanFr)")
        ax1.set_xlabel("Placebins")
        
        ax2 = ax1.twinx()
        line2,=ax2.plot(np.arange(0,len(matrix_speed_mean)),matrix_speed_mean,color="grey",label="speed")
        
        ax2.fill_between(x=np.arange(0,len(matrix_speed_mean))
                         ,y1=matrix_speed_mean-matrix_speed_sem
                        ,y2=matrix_speed_mean+matrix_speed_sem,color="grey",alpha=.2)
        ax2.set_ylabel(r"speed(cm/s)")
        plt.legend([line1,line2],["Fr","Speed"],frameon =False,loc=(0.65,0.7))
        plt.title("mouse:%s-id:%s in %s"%(Context_Matrix_info["mouse_id"],idx,trialtype))
        plt.tight_layout()
    if save:

        filename = "mouse%sid%spart%sindex%saim%s.%s"%(Context_Matrix_info["mouse_id"],idx,Context_Matrix_info["part"],Context_Matrix_info["index"],Context_Matrix_info["aim"],format)
        savepath = os.path.join(savedir,filename)
        plt.savefig(savepath,format=format)
        print("%s is saved"%filename)
    if show:
        plt.show()
    plt.close('all')

# dotmap that reflect the speed direction 
def SingleCell_trace_in_SingleTrial(s:AnaMini,*args,**kwargs):
    """
    generate a dict containing lists of each context in which are dataframe of each trials
    the columns of the dataframe are [ms_ts,Body_speed_angle,idxes...]

    s.add_Trial_Num_Process()
    s.add_Context(context_map=None) # add Contxt as a set of [0,1,2]
    s.add_alltrack_placebin_num(according = "Head",place_bin_nums=[4,4,30,4,4,4],behavevideo=None) # add "place_bin_No"
    s.add_Body_speed(scale=0.33) # Body_speed & Body_speed_angle are ready

    """

    print("FUNC::SingleCell_trace_in_SingleTrial")
    df,index = s.trim_df(*args,**kwargs)
    df = df[index]
    Trial_Num = s.result["Trial_Num"][index]
    Context= s.result["Context"][index]

    # Trials is ready    

    df["Context"] = s.result["Context"]
    df["place_bin_No"] = s.result["place_bin_No"] # used only in screen data in context. actually s.add_is_in_context is alternative way to screen this.
    df["Trial_Num"] = s.result["Trial_Num"]
    df["ms_ts"] = s.result["aligned_behave2ms"]["corrected_ms_ts"]
    df["Body_speed_angle"] = s.result["Body_speed_angle"]

    #screen contexts
    contexts = np.unique(s.result["Context"]) 
    df = df.loc[df["Context"].isin(contexts)]
    print("screen df according to given contexts")

    #screen placebins
    placebins = np.unique(s.result["place_bin_No"]) 
    df = df.loc[df["place_bin_No"].isin(placebins)]
    df.drop(columns="place_bin_No",inplace=True) 
    print("screen df according to given place bins")

    #screen trials
    trials = np.unique(s.result["Trial_Num"]) 
    df = df.loc[df["Trial_Num"].isin(trials)]
    print("screen df according to given tirals")
    

    Context_dataframe_info=dict()
    
    for context in np.unique(df["Context"]):
        if not context==-1:
            Trial_list_info = list()
            context_df = df[df["Context"]==context]
            context_df.drop(columns="Context",inplace=True)
            for trial in np.unique(context_df["Trial_Num"]):
                temp = context_df[context_df["Trial_Num"]==trial]
                temp.drop(columns="Trial_Num",inplace=True)
                Trial_list_info.append(temp) #Besides idxes only "ms_ts" and "Body_speed_angle" left
            Context_dataframe_info["context%s"%context]=Trial_list_info

    return Context_dataframe_info
def plot_trace_with_running_direction(Context_dataframe_info):
    """
    return internal functions for plotting trace of each trial with two colors means two direction
    """
    def plot(idx,context):
        """
        return an inter function
            That could plot trace along each trial which have "ms_ts" and "Body_speed_angle" Besides idxes 
        """

        if context in Context_dataframe_info.keys():
            trials = Context_dataframe_info[context]

        else:
            print("Context %s doesn't exist."%context)
            return 

        trial_lens = len(trials)

        fig,axes = plt.subplots(trial_lens,1
                                ,figsize=(8,0.5*trial_lens)
                                ,subplot_kw = {"xticks":[],"yticks":[]}
                                )

        for i,ax in enumerate(axes):
            if i==0:
                ax.set_title("incontext(placebins) firing rate cellid:%s, context:%s"%(idx,context))
            if i==len(axes)-1:
                ax.set_xlabel("Trial Time(ctx enter > r-ctx_exit)")
            ax.set_ylabel(i+1,rotation=0)
            ax.set_aspect("auto")
            #ax.set_axis_off() 
            ax.spines['top'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            color=["red" if i>90 and i<280 else "green" for i in trials[i]["Body_speed_angle"]]
            ax.scatter(x=trials[i]["ms_ts"]-np.min(trials[i]["ms_ts"]),y=trials[i][idx],c=color,s=1)
        plt.show()


    return plot



#%% for behavioral analysis

def behave_stat_info(s:AnaMini,):
    """
    计算每一个test or train session的行为学相关参数
    """

    stat_info = {}
    s.add_behave_choice_side()
    Choice_side = s.result["behave_choice_side"]
    s.add_behave_forward_context(according="Enter_ctx")
    Context  = s.result["behave_forward_context"]

    stat_info["date"] = s.result["index"][0]
    stat_info["mouse_id"] = s.result["mouse_id"][0]
    stat_info["aim"] = s.result["aim"][0]
    stat_info["Trial_Num"] = s.result["behavelog_info"].shape[0]

    # Trial_length is defined from nosepoke to reverse context exit        update at 20210219

    stat_info["Trial_length"] = list(s.result["behavelog_time"]["P_r_exit"]-s.result["behavelog_time"]["P_nose_poke"])

    Left_choice = s.result["behavelog_info"]["Left_choice"]
    Right_choice = s.result["behavelog_info"]["Right_choice"]
    Choice_class = s.result["behavelog_info"]["Choice_class"]
    forward_context = s.result["behave_forward_context"]

    stat_info["bias"] =  (max(Left_choice)-max(Right_choice))/(max(Left_choice)+max(Right_choice))
    stat_info["Total_Accuracy"] = sum(Choice_class)/len(Choice_class)
    try:
        stat_info["Left_Accuracy"] = sum(Choice_class[Choice_side=="left"])/len(Choice_class[Choice_side=="left"])
    except:
        stat_info["Left_Accuracy"] = None
    try:
        stat_info["Right_Accuracy"] = sum(Choice_class[Choice_side=="right"])/len(Choice_class[Choice_side=="right"])
    except:
        stat_info["Right_Accuracy"] = None

    for ctx in set(Context):
        try:
            stat_info["ctx_%s_Accuracy"%ctx]= sum(Choice_class[Context==ctx])/len(Choice_class[Context==ctx])
        except:
            stat_info["ctx_%s_Accuracy"%ctx]= None

    return stat_info



def behave_logistic_regression(s:AnaMini,):

    print("FUNC::behave_logistic_regression")
    # organize choice
    Choice = []
    if s.result["behavelog_info"]["Left_choice"].iloc[0]==1:
        Choice.append("left")
    else:
        Choice.append("right")

    for choice in (s.result["behavelog_info"]["Left_choice"].diff()[1:]-s.result["behavelog_info"]["Right_choice"].diff()[1:]):
        if choice == 1:
            Choice.append("left")
        else:
            Choice.append("right")
    s.result["behavelog_info"]["Choice_side"] = Choice

    Choice = [0 if i == "left" else 1 for i in Choice]  ## left for 0 right for 1

    # organize noisy
    noisy=[]
    if s.result["behavelog_info"]["Enter_ctx"].iloc[0]==1:
        noisy.append(0)
    else:
        noisy.append(1)

    for context_change in np.diff(s.result["behavelog_info"]["Enter_ctx"]):
        if context_change == 0:
            noisy.append(0)
        else:
            noisy.append(1)

    # organize dataset for X and y
    context = s.result["behavelog_info"]["Enter_ctx"][1:].values
    last_choice = Choice[0:-1]
    last_reward = s.result["behavelog_info"]["Choice_class"][0:-1].values
    noise = noisy[1:]
    y = pd.Series(Choice[1:],name="choice")
    X = pd.DataFrame()
    X["context"]=context
    X["last_choice"] = last_choice
    X["last_reward"] = last_reward
    X["noise"] = noise
    # logistic regression
    try:
        lrcv = LRCV(Cs=20,max_iter=50,random_state=0,cv=10,refit=True).fit(X,y)

        lg_info ={
        "X":X,
        "y":y,
        "coef":lrcv.coef_[0],
        "coef_names" : X.columns.values,
        "score":lrcv.score(X,y),
        "lrcv":lrcv
        }
    except:
        lg_info=None

    return lg_info

    # 
    
#%% SVM decoding for single neuron with Bayesian optimization


#%% which is about to discard

