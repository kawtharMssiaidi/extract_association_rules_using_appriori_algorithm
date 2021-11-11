# -*- coding: utf-8 -*-
"""
Created on Mon May 10 04:50:00 2021

@author: pc
"""
import os #lire dossier 
import pandas  #lire les données
from mlxtend.frequent_patterns import apriori,fpgrowth #importation de la fonction apriori
import numpy as np #creer liste 
from mlxtend.frequent_patterns import association_rules #fonction de calcul des règles
import matplotlib.pyplot as plt #tracer courbe

#L'extraction des règles d'association a pour but de découvrir des relations significatives entre
#attributs binaires extraits des bases de données. 



#L'extraction de règles d'association est un processus itératif et interactif constitué de plusieurs
#phases allant de la sélection et la préparation des données jusqu'a l'interprétation des résultats, 


"""1ere phase :Préparation des données
Cette phase consiste à sélectionner les données (attributs et objets)
de la base de données utiles à l'extraction des règles d'association et transformer ces données en
un contexte d'extraction.

 """
 
#notre base de donnée choisit d'une marchet : groceries-groceries

#changement de dossier

os.chdir("K:\M2\entrepot_données")

#importation des données
D = pandas.read_table("groceries-groceries.csv",delimiter=",",header=0)


#affichage de tableau de BD
print("\n ***notre base de donnée(20 premier lignes):***\n")
print(D.head(20))
print("\n",D.shape)


#Transformation en tableau binaire
#on a pris un extrait de notre base donnee 60 premiers ligne pour minimisé le temps d 'execution de programme 
#tableau croisé 0/1 en prennant les atribus de 1er colonne item1  
TC = pandas.crosstab(D.ID[:60],D.Item1)

print("\n***les colonnes de notre nouveau table:***\n")
print( TC.columns)

#boucle pour construire notre tableau en filtrant tous les colonnes 
#d'origine tableau (items1-->item33)

 
for i in range(1,60):
    j=0
    for col in TC.columns: 
        for k in range(1,33):
            if D.iloc[i,k-1]==col :
                TC.iloc[i,j]=1   
        j=j+1   
        
#affichage de 20lignes et 12 colonnes   
print("\n***Notre table binaire***\n")      
print(TC.iloc[:40,:22])
print("\n",TC.shape)



"""2eme phase : Extraction des ensembles fréquents d'attributs (itemsets)
nous avons utiliser "mxtend "
Cette phase consiste à extraire du contexte
tous les ensembles d'attributs binaires, appelés itemsets, qui sont fréquents dans le contexte
Un itemset l est fréquent si son support, qui correspond au nombre d'objets du contexte qui «
contiennent » l, est supérieur ou égal au seuil minimal de support minsupport défini par
l'utilisateur.
"""


#itemsets frequents en utilisant la fonction apriori avec seuil mini : 0.00003 pour avoir plusieur itemsets

freq_itemsets = apriori(TC,min_support=0.0005,max_len=2,use_colnames=True,verbose=1)            
    
#type --> pandas DataFrame
print("\ntype:\n")
type(freq_itemsets)


#nombre d'itemsets  : 3573
print("\n ***nombre des itemsets:***\n")
print(freq_itemsets.shape)

#affichage des 30 premiers itemsets 
print("\n***Notre tableau des itemsets:***\n")
print(freq_itemsets.head(30))


#type du champ '
print("\ntype du champ:")
print(type(freq_itemsets.itemsets))



#fonction de test d'inclusion pour filtrer si un attribut est inclus dans  item :
    
""" 1ere solution avec 'apply' """
    
    
def is_inclus (x,items):
     return items.issubset(x)

#recherche des index des itemsets correspondant à une condition : choisissant "beef"

id = np.where(freq_itemsets.itemsets. apply (is_inclus,items={'beef'}))
print("\n***les index des itemsets dont il y en a 'beef'***\n")
print( id)

#affichage des itemsets
print("\n*** les itemsets de 'beef'***\n")
print(freq_itemsets.loc[id])


""" 2eme solution par fonction 'lamba' """


#passer par une fonction lambda si on est pressé 
id2=np.where(freq_itemsets.itemsets.apply( lambda x,ensemble:ensemble.issubset(x),ensemble={'beef'}))
print("\n***les index des itemsets avec la 2eme solution dont il y en a 'beef'***\n")
print( id2)                      

                                                                       
                                           
"""3eme methode Utilisation des opérateurs de comparaison de pandas.Series"""


                                                   
#itemsets contenant beef passer par les méthodes natives de Series
print("\n***les itemsets avec la 3eme solution dont il y en a 'beef'***\n")
print(freq_itemsets[freq_itemsets['itemsets'].ge({'beef'})])

#L’itemset contenant exclusivement l’item " beef "
print("\n***les  itemsets avec la 3eme solution dont il y en a uniquement 'beef'***\n")
print(freq_itemsets[freq_itemsets['itemsets'].eq({'beef'})])                                                                                                                                                                                         
                                                                                              
#itemsets contenant beef et grapes La position des items dans la requête n’influe pas sur les résultats bien sûr.
print("\n***les itemsets avec la 3eme solution dont il y en a 'beef' et 'grapes'***\n")
print(freq_itemsets[freq_itemsets['itemsets'].ge({'beef','ham'})])
                                                                                              






"""3eme phase:Génération des règles d'association
Durant cette phase, les itemsets fréquents extraits durant
la phase précédente sont utilisés afin de générer les règles d'association qui sont des implications
entre deux itemsets fréquents 
seules celles qui possèdent une
confiance supérieure ou égale au seuil minimal minconfiance défini par l'utilisateur sont
générées
"""




#génération des règles à partir des itemsets fréquents 178 regle 9 caractere
regles = association_rules(freq_itemsets,metric="lift",min_threshold=0.9)
# il y en a d'autre metric comme lift 

#type de l'objet renvoyé
print("\n le type des regles",type(regles))

#dimension
print("\n",regles.shape)

#liste des colonnes
print("\n***les colonnes de tableau des regles:***\n")
print(regles.columns)

#50 "premières" règles
print("\n***tableau des regles p***\n")
print(regles.iloc [:178,:])


 #règles en restreignant l'affichage à qqs colnnes
myRegles = regles.loc[:,['antecedents','consequents','lift']]
#pour afficher toutes les colonnes
pandas.set_option('display.max_columns',5)
pandas.set_option('precision',3)
#affichage des 10 premières règles pour 3 colonnes  antecedents,consequents,lift
print("\n***tableau des regles en affichant antecedents,consequents,lift***\n")
print(myRegles[:10])
                  
  #affichage des règles avec un LIFT supérieur ou égal à 7
print("\n*** lift>=7***\n")
print(myRegles[myRegles['lift'].ge(7.0)])

 #trier les règles dans l'ordre du lift décroissants 10 meilleurs règles
print("\n***triage lift decrossant :les 10 meilleurs regles***\n")
print(myRegles.sort_values(by='lift',ascending=False)[:10])
                
#filtrer les règles menant au conséquent beef
print("\n***filtrer les règles menant au conséquent beef***\n")
print(myRegles[myRegles['consequents'].eq({'beef'})])


#filtrer les règles contenant 'butter' dans leur antécédent
print("\n***filtrer les règles contenant 'butter' dans leur antécédent***\n")
print(myRegles[myRegles['antecedents'].ge({'butter'})])


