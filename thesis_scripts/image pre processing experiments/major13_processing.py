import os
from os import rename
import csv

os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh")


path = "C:/Users/Saurabh/Desktop/_50_/"   #change for king
new_path = "C:/Users/Saurabh/Desktop/cyto2017/mito_nui/renamed/"  #change for king
 
#files = os.listdir("C:/Users/Saurabh/Desktop/_50_/major13/")  #Image files path
files = os.listdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/") 

#print(files)

reader = csv.reader(open("C:/Users/Saurabh/Desktop/_50_/major13soln_hacky.csv", "r"), delimiter=",") #change for king
#reader = csv.reader(open("C:/Users/Saurabh/Desktop/mito_nui.csv", "r"), delimiter=",") #change for king


labels = list(reader)


print(labels)



a= files[0]
print(a)

b = a[:-4].split('.', 3)
print(b)

c =  "00" + "_"+ b[0] +  '.tif'
print(c)



os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh")



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
            #print("success! for", b[0], "and",l[0]) #
            if l[1] == 'abc':
                #print("STH")
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                #a = files[i]
                #print(a)   # orignal filename
                #b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = b[0] +  '_' + b[1]  + "_" + str("BOTH") + '.tif' + '.gz' # '.gz'
                #print(c) # Modified filename
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/") # otherwise doesnt work #change for king

                #os.rename(a,c) #
                #print(a,"a is renamed to",c) #
                
                #### Add Orignal and Train paths ############# 
                 #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/trn/")
                 #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/training/")

            elif l[1] == ' Nucleus':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "00" + "_"+ b[0] +  '_' + b[1]  + "_" + str("Nucleus") + '.tif' + '.gz' # '.gz'
                c = "00" + "_"+ b[0] +  '.png'
                #print(c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")  #change for king
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")

                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c) #

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
                #c = "01" + "_" + b[0] +  '_' + b[1]  + "_" + str("Nucleoli") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "01" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/01/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Nuclear membrane':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "02" + "_" + b[0] +  '_' + b[1]  + "_" + str("Nucmembrane") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "02" + "_"+ b[0] +  '.png'
                #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/02/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Golgi apparatus':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "03" + "_" + b[0] +  '_' + b[1]  + "_" + str("golgi") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "03" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/03/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Endoplasmic reticulum':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "04" + "_" + b[0] +  '_' + b[1]  + "_" + str("ER") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "04" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                #os.rename(a,c)
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/04/"+c) #
                print(a,"a is renamed to",c) #

            elif l[1] == ' Vesicles':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "05" + "_" + b[0] +  '_' + b[1]  + "_" + str("Vesicles") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "05" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/05/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Plasma membrane':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "06" + "_" + b[0] +  '_' + b[1]  + "_" + str("plasma") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "06" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/06/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
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
                #c = "07" + "_" + b[0] +  '_' + b[1]  + "_" + str("MITO") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "07" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/07/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Cytosol':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "08" + "_" + b[0] +  '_' + b[1]  + "_" + str("cytosol") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "08" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/08/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Microtubules':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "09" + "_" + b[0] +  '_' + b[1]  + "_" + str("microtubules") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "09" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/09/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Centrosome':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "10" + "_" + b[0] +  '_' + b[1]  + "_" + str("centrosome") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "10" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/10/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Actin filaments':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "11" + "_" + b[0] +  '_' + b[1]  + "_" + str("actin") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "11" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/11/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #

            elif l[1] == ' Intermediate filaments':
                #print(b[0],"is",l[1])
                # In linux just move the file to nucleoli folder
                a = []
                b = []
                a = files[i]
                #print(a)   # orignal filename
                b = a[:-4].split('_', 3)
                #print(b[0])  # First digits before _
                #c = "12" + "_" + b[0] +  '_' + b[1]  + "_" + str("interfilaments") + '.tif' + '.gz' # '.gz'
                #print(c) #
                c =  "12" + "_"+ b[0] +  '.png'
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/batch1/")  #change for king

                #os.rename("C:/Users/Saurabh/Desktop/_50_/major13/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/12/"+c) #
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/safe/")
                #os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/heh/")
                #os.rename(a,c)
                os.chdir("C:/Users/Saurabh/Desktop/_50_/classes/tst/")
                os.rename("C:/Users/Saurabh/Desktop/_50_/classes/tst/"+a,"C:/Users/Saurabh/Desktop/_50_/classes/testing/"+c)
                print(a,"a is renamed to",c) #
        






