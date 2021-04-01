
from mylab.Cfile import File
import glob,os,sys,re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV

class LinearTrackBehavioralLogFile(File):
    def __init__(self,file_path,context_map=["B","A","C","N"]):
        super().__init__(file_path)

        self.context_map = context_map
        # print("context map is %s"%self.context_map)
        self.date = re.findall(r"(\d{8})-\d{6}",self.file_path)[0]
        self.mouse_id = re.findall(r"(\d+)-\d{8}-\d{6}",self.file_path)[0]
        self.aim = re.findall(r"LickWater-(.*)-%s"%self.mouse_id,self.file_path)[0]


        self.data = pd.read_csv(self.file_path ,skiprows=3)

        self.Enter_Context = set(self.data["Enter_ctx"])
        self.Exit_Context = set(self.data["Exit_ctx"])


    @property
    def choice(self):
    
        Choice = []
        if self.data["Left_choice"][0]==1:
            Choice.append("left")
        else:
            Choice.append("right")
            
        for choice in (np.diff(self.data["Left_choice"])-np.diff(self.data["Right_choice"])):
            if choice == 1:
                Choice.append("left")
            else:
                Choice.append("right")
                
        return pd.Series(Choice,name="choice")
    

    @property
    def noise(self):
        """
        Series contains only 0 and 1, 0 means no motion, 1 means motion
        """
        noisy=[]
        if self.data["Enter_ctx"][0]==1:
            noisy.append(0)
        else:
            noisy.append(1)

        for context_change in np.diff(self.data["Enter_ctx"]):
            if context_change == 0:
                noisy.append(0)
            else:
                noisy.append(1)
                
        return pd.Series(noisy,name="noise")


    @property
    def reward(self):
        return self.data["Choice_class"]
    
    @property
    def Trial_duration(self):        
        durations = np.diff(self.data.iloc[:,-6:],axis=1)
        Trial_duration = pd.DataFrame(durations,columns=["process1","process2","processs3","process4","process5"])
        Trial_duration["Trial"] = np.sum(durations,axis=1)
        return Trial_duration

    def behave_stat_info(self):
        state_info = {}
        Choice_side=self.choice
        
        state_info["date"]=self.date
        state_info["mouse_id"]=self.mouse_id
        state_info["aim"]=self.aim
        state_info["Trial_Num"] = self.data.shape[0]
        state_info["bias"] =  (max(self.data["Left_choice"])-max(self.data["Right_choice"]))/(max(self.data["Left_choice"])+max(self.data["Right_choice"]))
        state_info["Total_Accuracy"] = sum(self.data["Choice_class"])/len(self.data["Choice_class"])
        state_info["Left_Accuracy"] = sum(self.data["Choice_class"][Choice_side=="left"])/len(self.data["Choice_class"][Choice_side=="left"])
        state_info["Right_Accuracy"] = sum(self.data["Choice_class"][Choice_side=="right"])/len(self.data["Choice_class"][Choice_side=="right"])

        # for ctx in self.Enter_Context:
        #     state_info["ctx_%s_Accuracy"%ctx]= sum(self.data["Choice_class"][self.data["Enter_Context"]==ctx])/len(self.data["Choice_class"][self.data["Enter_Context"]==ctx])

        return state_info

    def behave_logistic_regression(self):
        Choice = self.choice
        Choice = [0 if i == "left" else 1 for i in Choice]
        noisy = self.noise

        context = self.data["Enter_ctx"][1:].values
        last_choice = Choice[0:-1]
        last_reward = self.data["Choice_class"][0:-1].values
        noise = noisy
        y = pd.Series(Choice[1:],name="choice")
        X = pd.DataFrame()
        X["context"]=context
        X["last_choice"] = last_choice
        X["last_reward"] = last_reward
        X["noise"] = noise
        print(X)
        lrcv = LRCV(Cs=20,max_iter=50,random_state=0,cv=10,refit=True).fit(X,y)

        lg_info ={
        "X":X,
        "y":y,
        "coef":lrcv.coef_[0],
        "coef_names" : X.columns.values,
        "score":lrcv.score(X,y),
        "lrcv":lrcv
        }

        return lg_info

if __name__ == "__main__":
    file = r"C:\Users\Sabri\Desktop\new_method_to_record\behave\LickWater-test-201033-20200825-130238_log.csv"
    data = LinearTrackBehaviorFile(file)
    print(data.Right_Accuracy)
    print(data.Context_A_Accuracy)
    print(data.Context_B_Accuracy)
