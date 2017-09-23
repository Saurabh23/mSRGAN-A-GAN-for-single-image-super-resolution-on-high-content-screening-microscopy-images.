import os
from os import rename
import csv

path = "C:/Users/Saurabh/Desktop/cyto2017/mito_nui/"   #change for king
new_path = "C:/Users/Saurabh/Desktop/cyto2017/mito_nui/renamed/"  #change for king
 
files = os.listdir("C:/Users/Saurabh/Desktop/cyto2017/mito_nui/")  #change for king

print(files)

reader = csv.reader(open("C:/Users/Saurabh/Desktop/mito_nui.csv", "r"), delimiter=",") #change for king

labels = list(reader)

for l in labels:    
    #print(l[0])  # 
    for i in range(0,len(files)):
        a = []
        b = []
        a = files[i]
        #print(a)
        b = a[:-4].split('_', 3)
        #print(b[0])   # b[0] = 1001
        if b[0] == l[0]:
            print("success! for", b[0], "and",l[0]) #
            if l[3] == 'Nucleoli':
                #print("STH")
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                c = b[0] +  '_' + b[1]  + "_" + str("BOTH") + '.tif' + '.gz' # '.gz'
                #print(c) # Modified filename
                os.chdir("C:/Users/Saurabh/Desktop/cyto2017/mito_nui/") # otherwise doesnt work #change for king

                os.rename(a,c) #
                print(a,"a is renamed to",c) #

            elif l[1] == ' Nucleoli':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                c = b[0] +  '_' + b[1]  + "_" + str("Nucleoli") + '.tif' + '.gz' # '.gz'
                #print(c) #
                os.chdir("C:/Users/Saurabh/Desktop/cyto2017/mito_nui/")   #change for king

                os.rename(a,c) #
                print(a,"a is renamed to",c) #
                
            elif l[1] == ' Mitochondria':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                c = b[0] +  '_' + b[1]  + "_" + str("Mito") + '.tif' + '.gz' # '.gz'
                #print(c) #
                os.chdir("C:/Users/Saurabh/Desktop/cyto2017/mito_nui/")     #change for king

                os.rename(a,c) #
                print(a,"a is renamed to",c) #
            

    