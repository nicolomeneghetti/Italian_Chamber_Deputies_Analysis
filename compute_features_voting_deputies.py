
def copmute_features_voting_deputies(df_deputy,df_votes,mesi_thr):
    
    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    

    
    def converti_str_in_array(rr):
        res=[0.,0.,0.,0.,0.,0.,0.,0.]
        if(pd.isnull(rr)==0):
            
            rr=rr.replace('[','')
            rr=rr.replace(']','')
            rr = rr.split(", ")
            res[0]+=int(rr[0])
            res[1]+=int(rr[1])
            res[2]+=int(rr[2])
            res[3]+=int(rr[3])
            res[4]+=int(rr[4])
            res[5]+=int(rr[5])
            res[6]+=int(rr[6])
            res[7]+=int(rr[7])
            
            return np.asarray(res)

        
    
    def aggiusta_df_votes_per_colonne(df_votes, index_Deputy, index_voting_sessions):
        
        riga_considerata = df_votes.iloc[index_Deputy]
        
        
        res_n=[[0.,0.,0.,0.,0.,0.,0.,0.]]
        for indic in index_voting_sessions:
            rr = riga_considerata[indic]
            if(pd.isnull(rr)==0):
                rr=converti_str_in_array(rr)
                res_n.append(rr)
        del res_n[0] 
        
        if(len(res_n)>0):
            normalizzazione=0
            risultato_toto=[0,0,0,0,0,0,0,0]
            for i in range(len(res_n)):
                normalizzazione+=(res_n[i][0]+res_n[i][1])
                risultato_toto[0]+=res_n[i][0]
                risultato_toto[1]+=res_n[i][1]
                risultato_toto[2]+=res_n[i][2]
                risultato_toto[3]+=res_n[i][3]
                risultato_toto[4]+=res_n[i][4]
                risultato_toto[5]+=res_n[i][5]
                risultato_toto[6]+=res_n[i][6]
                risultato_toto[7]+=res_n[i][7]
                
            risultato_toto/=normalizzazione
        else:
            risultato_toto=-1
        return risultato_toto   
    
    
    #%%
        
    # separate deputy who switcher party from those mono group
    df_deputy_monogroup = df_deputy[df_deputy.Ngruppi <= 1]
    df_party_switchers = df_deputy[df_deputy.Ngruppi > 1]
    
    
    col_names=df_votes.columns
    col_names=col_names[4:len(cuscus)]
    data=[]
    for alpha in range(len(col_names)):
        temp=col_names[alpha]
        temp=temp.replace(' 00:00:00','')
        temp = datetime.strptime(temp, '%Y-%m-%d').date()
        data.append(temp)
    dates_voting_sessions=data
        
    
    
    results_of_party_switcher=[]
    has_MP_left_party=[]
    
    for index, p in df_party_switchers.iterrows():
                
        number_of_groups=p.Ngruppi
        served_groups=p.Gruppi
        
        data=[]
        for extract in range(number_of_groups+1):
            match = re.search(r'\d{2}.\d{2}.\d{4}', served_groups)
            date = datetime.strptime(match.group(), '%d.%m.%Y').date()
            served_groups=served_groups.replace(match.group(),'')

            data.append(date)
        
        data_temp=data.copy()
        for loop in range(len(data)):
            data_temp[loop]=min(data)
            min_index = data. index(min(data))
            del data[min_index]
        dates_of_party_leave=data_temp
        
        del dates_of_party_leave[0]
        del dates_of_party_leave[-1]
        
        ###########################################################################
        # inidivdua i voti del parlamentare in questione
        i_deputy = df_votes.index[(df_votes['cognome'] == p['cognome'])].values
        
        if(np.size(i_deputy)>1):
            i_parlamentare = (df_votes.index[(df_votes['nome'] == p['nome']) & (df_votes['cognome'] == p['cognome'])]).values[0]
        else:
            i_parlamentare = df_votes.index[(df_votes['cognome'] == p['cognome'])].values[0]        
        ###########################################################################
        index_voting_sessions=[]
        for loop1 in range(len(dates_voting_sessions)):
            if(dates_voting_sessions[loop1]<mesi_thr):
                if(dates_voting_sessions[loop1]<dates_of_party_leave[0]):
                    index_voting_sessions.append(loop1)
        index_voting_sessions=np.asarray(index_voting_sessions)+4

        output_party_switcher=aggiusta_df_votes_per_colonne(df_votes,i_parlamentare,index_voting_sessions)
        
        
        
        if(isinstance(output_party_switcher, int)==0):
            has_MP_left_party.append(dates_of_party_leave[0]<mesi_thr)
            
        if(type(output_party_switcher)==np.ndarray):
            output_non_timevarying = df_deputy.iloc[i_parlamentare]['categoriaRegione_id']
            output_non_timevarying1 = df_deputy.iloc[i_parlamentare]['categoriaRegione_collegio_id']   
            output_non_timevarying2 = df_deputy.iloc[i_parlamentare]['categoriaIstruzione']   
            output_non_timevarying3 = df_deputy.iloc[i_parlamentare]['sesso_id']
            output_non_timevarying4 = df_deputy.iloc[i_parlamentare]['eta']
            output_non_timevarying5 = df_deputy.iloc[i_parlamentare]['numeroMandati']
            
            
            output_party_switcher = np.append(output_party_switcher,output_non_timevarying)
            output_party_switcher = np.append(output_party_switcher,output_non_timevarying1)
            output_party_switcher = np.append(output_party_switcher,output_non_timevarying2)
            output_party_switcher = np.append(output_party_switcher,output_non_timevarying3)
            output_party_switcher = np.append(output_party_switcher,output_non_timevarying4)
            output_party_switcher = np.append(output_party_switcher,output_non_timevarying5)

            results_of_party_switcher.append(output_party_switcher)
                
            
#%%
    resultats_mono_group=[]
    for index, p in df_deputy_monogroup.iterrows():        

        i_parlamentare = df_votes.index[(df_votes['cognome'] == p['cognome'])].values
        
        if(np.size(i_parlamentare)>1):
            i_parlamentare = (df_votes.index[(df_votes['nome'] == p['nome']) & (df_votes['cognome'] == p['cognome'])]).values[0]
        else:
            i_parlamentare = df_votes.index[(df_votes['cognome'] == p['cognome'])].values[0]
            
        index_voting_sessions=[]
        for pd1 in range(len(dates_voting_sessions)):
            if(dates_voting_sessions[pd1]<mesi_thr):
                    index_voting_sessions.append(pd1)
        
        index_voting_sessions=np.asarray(index_voting_sessions)+4

        
        output_final_single_group=aggiusta_df_votes_per_colonne(df_votes,i_parlamentare,index_voting_sessions)
        
        if(type(output_final_single_group)==np.ndarray):
            i_parl = df_deputy.index[(df_deputy['cognome'] == p['cognome'])].values
            if(np.size(i_parl)>1):
                i_parl = (df_deputy.index[(df_deputy['nome'] == p['nome']) & (df_deputy['cognome'] == p['cognome'])]).values[0]
            else:
                i_parl = df_deputy.index[(df_deputy['cognome'] == p['cognome'])].values[0]

            output_non_timevarying = df_deputy.iloc[i_parlamentare]['categoriaRegione_id']
            output_non_timevarying1 = df_deputy.iloc[i_parlamentare]['categoriaRegione_collegio_id']   
            output_non_timevarying2 = df_deputy.iloc[i_parlamentare]['categoriaIstruzione']   
            output_non_timevarying3 = df_deputy.iloc[i_parlamentare]['sesso_id']
            output_non_timevarying4 = df_deputy.iloc[i_parlamentare]['eta']
            output_non_timevarying5 = df_deputy.iloc[i_parlamentare]['numeroMandati']
            
            output_final_single_group = np.append(output_final_single_group,output_non_timevarying)
            output_final_single_group = np.append(output_final_single_group,output_non_timevarying1)
            output_final_single_group = np.append(output_final_single_group,output_non_timevarying2)
            output_final_single_group = np.append(output_final_single_group,output_non_timevarying3)
            output_final_single_group = np.append(output_final_single_group,output_non_timevarying4)
            output_final_single_group = np.append(output_final_single_group,output_non_timevarying5)
            
            resultats_mono_group.append(output_final_single_group)

    
    
    return has_MP_left_party, resultats_mono_group, results_of_party_switcher
