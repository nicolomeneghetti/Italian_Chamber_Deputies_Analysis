"""
Created on Thu Oct 22 10:25:03 2020

@author: Marina e Nicolo
"""

def compute_features_prediction(df_parlamentari,df_votes,moving_wind,offset_window,random_indexes_fedeli):
    
    
    import pandas as pd
    import numpy as np
    from queries_parlamento import queries_parlamento
    from IPython import get_ipython
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
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
        
        
    def aggiusta_df_votes_per_colonne(df_votes,indice_riga,indici_colonne,parl_considerato):
        
        riga_considerata = df_votes.loc[indice_riga]

        res_n=[[0.,0.,0.,0.,0.,0.,0.,0.]]
        for indic in indici_colonne:
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
    
        

    ###########################################################################
        
    # per prima cosa dividiamo quelli che hanno cambiato gruppo parlamentare da quelli fedelissimi invece
    df_MP_single_group = df_parlamentari[df_parlamentari.Ngruppi <= 1]
    df_MP_pluri_group = df_parlamentari[df_parlamentari.Ngruppi > 1]
    
    
    
    temp=df_votes.columns
    temp=temp[4:len(temp)]
    data2=[0]
    for alpha in range(len(temp)):
        temp=temp[alpha]
        temp=temp.replace(' 00:00:00','')
        temp = datetime.strptime(temp, '%Y-%m-%d').date()
        data2.append(temp)
    del data2[0]
    date_colonne=data2
        
    
    
    result_MPs_pluri_party=[]
    alpha_iterazione=0
    result_MPs_single_party=[]
    index_across_single_group = 0
    
    for index, p in df_MP_pluri_group.iterrows():
        
        alpha_iterazione+=1
        
        numero_gruppi=p.Ngruppi
        vect=p.Gruppi

        # extract dates of beginning and enf of memebership of the MPs to 'numero_gruppi'
        data=[0]
        for extract in range(numero_gruppi+1):
            match = re.search(r'\d{2}.\d{2}.\d{4}', vect)
            
            date = datetime.strptime(match.group(), '%d.%m.%Y').date()
            vect=vect.replace(match.group(),'')
            
            data.append(date)
        del data[0]
        
        data_temp=data.copy()
        for porco in range(len(data)):
            data_temp[porco]=min(data)
            min_index = data. index(min(data))
            del data[min_index]
        data=data_temp
        
        data_inizio_tutto = data[0]
        
        del data[0]
        del data[-1]
        # in data is saved the date of the party switch


        ###########################################################################
        # select the MP in question
        i_parlamentare_vettore = df_votes.index[(df_votes['cognome'] == p['cognome'])].values
        
        if(np.size(i_parlamentare_vettore)>1):
            i_parlamentare = (df_votes.index[(df_votes['nome'] == p['nome']) & (df_votes['cognome'] == p['cognome'])]).values[0]
        else:
            i_parlamentare = df_votes.index[(df_votes['cognome'] == p['cognome'])].values[0]        
        
        
        
        inizio_xvii_leg = datetime(2013, 4, 9)
        inizio_xvii_leg=inizio_xvii_leg.date()
        
        inizio_zona = data[0]-timedelta(moving_wind)-timedelta(offset_window)
        fine_zona = inizio_zona + timedelta(moving_wind)
        
        if(fine_zona>inizio_xvii_leg):
        
            da=0    
            indici_colonne=[0]

            for pd1 in range(len(date_colonne)):                
                if(date_colonne[pd1]>inizio_zona):
                    if(date_colonne[pd1]<=fine_zona):
                        indici_colonne.append(pd1)
            del indici_colonne[0]
                  
            indici_colonne=np.asarray(indici_colonne)+4
            output_final=aggiusta_df_votes_per_colonne(df_votes,i_parlamentare,indici_colonne,p)
            

            if(type(output_final)==np.ndarray):
                
                output_non_timevarying = df_parlamentari.loc[i_parlamentare]['categoriaRegione_id']
                output_non_timevarying1 = df_parlamentari.loc[i_parlamentare]['categoriaRegione_collegio_id']   
                output_non_timevarying2 = df_parlamentari.loc[i_parlamentare]['categoriaIstruzione']   
                output_non_timevarying3 = df_parlamentari.loc[i_parlamentare]['sesso_id']
                output_non_timevarying4 = df_parlamentari.loc[i_parlamentare]['eta']
                output_non_timevarying5 = df_parlamentari.loc[i_parlamentare]['numeroMandati']
                
                output_final = np.append(output_final,output_non_timevarying)
                output_final = np.append(output_final,output_non_timevarying1)
                output_final = np.append(output_final,output_non_timevarying2)
                output_final = np.append(output_final,output_non_timevarying3)
                output_final = np.append(output_final,output_non_timevarying4)
                output_final = np.append(output_final,output_non_timevarying5)                
    
                result_MPs_pluri_party.append(output_final)
            
            if(type(output_final)==np.ndarray):
                output_final_fedeli=-1
                while(type(output_final_fedeli)!=np.ndarray):
                    p_fedele = df_MP_single_group.iloc[random_indexes_fedeli[index_across_single_group]]
                    index_across_single_group += 1
                    
                    i_parlamentare = df_votes.index[(df_votes['cognome'] == p_fedele['cognome'])].values
                    if(np.size(i_parlamentare)>1):
                        i_parlamentare = (df_votes.index[(df_votes['nome'] == p_fedele['nome']) & (df_votes['cognome'] == p_fedele['cognome'])]).values[0]
                    else:
                        i_parlamentare = df_votes.index[(df_votes['cognome'] == p_fedele['cognome'])].values[0]
                        
                    output_final_fedeli=aggiusta_df_votes_per_colonne(df_votes,i_parlamentare,indici_colonne,p_fedele)
                    
                        
                if(type(output_final_fedeli)==np.ndarray):
                    output_non_timevarying = df_parlamentari.loc[i_parlamentare]['categoriaRegione_id']
                    output_non_timevarying1 = df_parlamentari.loc[i_parlamentare]['categoriaRegione_collegio_id']   
                    output_non_timevarying2 = df_parlamentari.loc[i_parlamentare]['categoriaIstruzione']   
                    output_non_timevarying3 = df_parlamentari.loc[i_parlamentare]['sesso_id']
                    output_non_timevarying4 = df_parlamentari.loc[i_parlamentare]['eta']
                    output_non_timevarying5 = df_parlamentari.loc[i_parlamentare]['numeroMandati']
                    
                    output_final_fedeli = np.append(output_final_fedeli,output_non_timevarying)
                    output_final_fedeli = np.append(output_final_fedeli,output_non_timevarying1)
                    output_final_fedeli = np.append(output_final_fedeli,output_non_timevarying2)
                    output_final_fedeli = np.append(output_final_fedeli,output_non_timevarying3)
                    output_final_fedeli = np.append(output_final_fedeli,output_non_timevarying4)
                    output_final_fedeli = np.append(output_final_fedeli,output_non_timevarying5)
                            
                    result_MPs_single_party.append(output_final_fedeli)  
            
            
        
    return result_MPs_pluri_party, result_MPs_single_party


