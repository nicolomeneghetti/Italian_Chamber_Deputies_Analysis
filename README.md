# Italian_Chamber_Deputies_Analysis

This repository contains the codes for running classification and prediction of party switching in the Italian Chamber of Deputies. 
A thorough description of the results and the codes here used can be found in the paper published on iScience (which is open access): https://doi.org/10.1016/j.isci.2023.107098

!! IMPORTANT !!
Please note that this repository does not contain the data for running the machine learning (namley, random forest classifiers) algorithms. Specifically, the repository does not contain the voting records of the voting sessions of the last two legislatures of the Italian Chamber of Deputies as we do not own them. The data are however freely available and can be downloaded from https://dati.camera.it/it/. 



The data structure for running the code is thoroughly described in the codes and is also reported here: 

1) df_deputies is a dataframe that contains the following columns: 
- 'persona' -> link to the Deputy's personal page on the Chamber of Deputies website
- 'cognome' -> Deputy surname
- 'nome' -> Deputy name
- 'info' -> Deputy CV
- 'dataNascita' -> Deputy date of birth
- 'luogoNascita' -> Deputy place of birth
- 'genere' -> Deput gender
- 'inizioMandato' -> Deput date of beginning of service
- 'fineMandato' -> Deput date of service end
- 'collegio' -> Deput region of election
- 'lista' -> Deput party of election
- 'numeroMandati' -> Deput number of served terms
- 'Gruppi' -> Deput parties during their service
- 'Ngruppi' -> Deput number parties during their service

This set of non-voting features can be found in the "anagrafica" section of the freely accessible repository https://dati.camera.it/it/

2) df_votes is a dataframe that contains the following columns: 
- 'persona' -> link to the Deputy's personal page on the Chamber of Deputies website
- 'cognome' -> Deputy surname
- 'nome' -> Deputy name
- a set of columns named as the date (mm-dd-yy) of each voting sessions in the Chamber of  Deputies. Each entry of these columns contains a list that, for each Deputy, specify: 
    a) number of presence to voting sessions
    b) number of absence to voting sessions
    c) number of votes in agreement with party majority
    d) number of abstensions
    e) number of votes in favour of a law that was then approved by the Chamber 
    f) number of votes in opposition to a law
    g) number of votes in favour to a law
    h) number of votes casted in seret ballots

This set of voting features can be found in the freely accessible repository https://dati.camera.it/it/


_______________________________________________________________________________________________


The main codes are the following two: 

-) MAIN_Classification_Deputies_party_switcher: contains the code for running the classification algorithm between single-party and pluri-group Deputies of the Italian Chamber of Deputies


-) MAIN_Prediction_Deputies_party_switcher: contains the code for running the prediction algorithm between single-party and pluri-group Deputies of the Italian Chamber of Deputies. The prediction algorithm considers a two month window for considering the voting of the Deputies; this window is progressively moved away from the party switch and the classification accuracy is then computed

-) two auxilary functions used in the two MAIN codes
