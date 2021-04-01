import csv, os

#what directory are yyour files in? PAste here
directory =r'D:\OneDrive - University of Waterloo\Waterloo\PhD\Manuscripts\2021\DMS_ML_DispersionPlot\Predicted Curves\Unguided\Test'

#get everything with a .csv ending. Split your predicted data from your reference data in separate folders
files = [x for x in os.listdir(directory) if x.lower().endswith('.csv')]

#buncha empty lists for printing
mz = []
CCS = []
Classifier = []
SV1500 = []
SV2000 = []
SV2500 = []
SV3000 = []
SV3250 = []
SV3500 = []
SV3750 = []
SV4000 = []

for file in files:
    opf = open(directory+'//'+file,'r')
    data = opf.readlines() #without a file header
    data = opf.readlines()[1:]  #with a file header
    opf.close()
    
    for line in data:
        line = line.split(',')

        mz.append(line[0])
        CCS.append(line[1])
        Classifier.append(line[2])
        SV1500.append(line[3])
        SV2000.append(line[4])
        SV2500.append(line[5])
        SV3000.append(line[6])
        SV3250.append(line[7])
        SV3500.append(line[8])
        SV3750.append(line[9])   
        SV4000.append(line[10])           

opf = open(directory+'//Data.csv','w')
opf.write('m/z,CCS,Classifier,SV1500,SV2000,SV2500,SV3000,SV3250,SV3500,SV3750,SV4000\n')
opf.close()

rows = zip(mz,CCS,Classifier,SV1500,SV2000,SV2500,SV3000,SV3250,SV3500,SV3750,SV4000)
with open(directory+'//Data.csv','a',newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(round(float(elem),3) for elem in row)

