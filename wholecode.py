-----------------------one-----------------------------------

import csv
import pandas as pd
useridlist=[]
itemlist=[]

for i in range(1,6):
    if i==4 or i==5:
            with open('C:/Users/Eshwar C V S/Documents/Project DM/lastfm-dataset-1K/last.fm split files/p%d.csv'%i, mode='r') as infile:
                    reader = csv.reader(infile)
                    for rows in reader:
                        useridlist.append(rows[0])
                        itemlist.append(rows[5])
    else :
        with open('C:/Users/Eshwar C V S/Documents/Project DM/lastfm-dataset-1K/last.fm split files/p%d.csv' % i, mode='r',encoding='utf-8') as infile:
                
                reader = csv.reader(infile)
                for rows in reader:
                    useridlist.append(rows[0])
                    itemlist.append(rows[5])
d=pd.DataFrame()
d['userid'] = useridlist
d['songname'] = itemlist
print(d)
d=d.drop_duplicates(keep='first')
print(d)
d=d.groupby(['songname']).size().reset_index(name='count')
print(d)
d.to_csv('C:/Users/Eshwar C V S/Documents/Project DM/lastfm-dataset-1K/s12.csv')


-----------------------------------------two------------------------------------------------------------
import numpy as np
import pandas as pd
import collections
sid=0
min_songs=6
df = pd.read_csv('C:/Users/Eshwar C V S/Documents/Project DM/lastfm-dataset-1K/s12.csv',encoding='latin-1',sep=',')
df=df[df['count'] >= min_songs]
songlist=df['songname'].tolist()
Count_Col=df.shape[0]
songidcol=[]
for son in songlist:
    songidcol.append('s%d'%sid)
    sid=sid+1
df['songid']=songidcol
Count_Row=100
useridrow=[]
for u in range(1,101):
    useridrow.append('user%d'%u)
Matrix = pd.DataFrame(np.zeros((Count_Row,Count_Col)),index=useridrow,columns=songidcol)
for i in range(1,101):
    userdf=pd.read_csv('C:/Users/Eshwar C V S/Documents/Project DM/lastfm-dataset-1K/UserSplitFiles/user%d.csv'%i)
    usersonglist=userdf.iloc[:,5].tolist()
    counter = collections.Counter(usersonglist)
    for j in usersonglist:
        if (j in songlist):
            c = songlist.index(j)
            Matrix.iloc[i-1][c]=counter[j]
    counter.clear()
print(Matrix)
Matrix.to_csv('C:/Users/Eshwar C V S/Documents/Project DM/lastfm-dataset-1K/UIMatrix_sctest.csv', encoding='utf-8')
df.to_csv('C:/Users/Eshwar C V S/Documents/Project DM/lastfm-dataset-1K/postmatriclisttest.csv',encoding='utf-8')



-------------------------------three---------------------------------------------
import pandas as pd
import scipy as scipy
import sklearn
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
df=pd.read_csv('C:\Users\Eshwar C V S\Documents\Project DM\New folder/UIMatrix_sctest1.csv')
print("full matrix",df)
dft=df.T
tsidcol = list(dft.columns.values)
df.drop(df.columns[0], axis=1, inplace=True)
print("after drop",df)

def normalize(d) :
    ''' df_norm=(((d-d.mean())**2)/d.shape[1])**(1/2)
    return df_norm'''
    x = d.values  # returns a numpy array
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df
'''data_norm = df  # Has training + test data frames combined to form single data frame
    normalizer = StandardScaler()
    data_array = normalizer.fit_transform(data_norm.as_matrix())

    return pd.DataFrame(data_array)'''

d=normalize(df)
row_count=d.shape[0]
print("after normalize",d)



wcss = []
l=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200]
for i in range(5,201,5):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(d)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(l, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
------------------------------------------------------------------Four------------------------------------------------------
p2 = [5,wcss[0]]
p1 = [200,wcss[39]]
def line_eq(p1,p2):
    a=-1*(p2[1]-p1[1])/(p2[0]-p1[0])
    c=-1*(p1[1]-(a*(p1[0])))
    b=1
    l=[a,b,c]
    return l
line=line_eq(p1,p2)
def distance_to_line(p3):
    n=abs((line[0]*p3[0])+(line[1]*p3[1])+(line[2]))
    denom=((line[0]*line[0])+(line[1]*line[1]))**(1/2)
    dis=n/denom
    return dis
maxdis=0
j=0
k=0
for i in range(5,201,5):
    p = [i,wcss[j]]
    dis=distance_to_line(p)
    if dis>maxdis:
        maxdis=dis
        k=i
    j=j+1
print(k)


    
 ---------------------------------------------Five-----------------------------------------------------
p2 = [5,wcss[0]]
p1 = [200,wcss[39]]
def line_eq(p1,p2):
    a=-1*(p2[1]-p1[1])/(p2[0]-p1[0])
    c=-1*(p1[1]-(a*(p1[0])))
    b=1
    l=[a,b,c]
    return l
line=line_eq(p1,p2)
def distance_to_line(p3):
    n=abs((line[0]*p3[0])+(line[1]*p3[1])+(line[2]))
    denom=((line[0]*line[0])+(line[1]*line[1]))**(1/2)
    dis=n/denom
    return dis
maxdis=0
j=0
k=0
for i in range(5,201,5):
    p = [i,wcss[j]]
    dis=distance_to_line(p)
    if dis>maxdis:
        maxdis=dis
        k=i
    j=j+1
print(k)

-----------------------------------six----------------------------------------
import random
import operator
import math
def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    #print ("mem",membership_mat)
    return membership_mat


def calculateClusterCenter(membership_mat):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(0,n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
        print("center",center)
    return cluster_centers


def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(map(operator.sub, x, cluster_centers[j])) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat


def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels



def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    cluster_centers = calculateClusterCenter(membership_mat)
    print("this is the matrix",membership_mat)
    print("\n")
    print("clusterlabels", cluster_labels)
    print("\n")
    print("clustercenters", cluster_centers)
    return cluster_labels, cluster_centers, membership_mat


labels, centers, matrix = fuzzyCMeansClustering()

   
        
        
-----------------------------------------seven--------------------------------------------------------
def euclidean_distance(train,test):
   train = np.asarray(train)
   test = np.asarray(test)
   temp=train-test
   temp=[x**2 for x in temp]
   dist=np.sum(temp)
   dist=dist**(1/2)
   return dist

def manhattan_distance(train,test):
    train = np.asarray(train)
    test = np.asarray(test)
    temp = train - test
    temp=np.absolute(temp)
    d=np.sum(temp)
    return d


def evaluate_nearest(mv,test) :
    flag=[]
    print(len(test))
    print(len(mv))
    for i in range (0,len(test)) :
        f=0
        max=1000000000
        for j in range (0,len(mv)):
            temp = manhattan_distance(test[i],mv[j])
            if(max > temp ):
                max = temp
                f=j
        flag.append(f)
    return flag
cluster_number = []
x=[]
x=[tedf.iloc[i] for i in range(0,len(tedf))]
cluster_number=evaluate_nearest(centers,x)
print ('cluster_number',cluster_number)

rvdfl=[]
rv50dfl=[]
rv60dfl=[]
testdfl=[]
all_recommend_vector=[]
top50_recommend_vector=[]
top60_recommend_vector=[]


--------------------------------eight--------------------------------------------------
import numpy
def binarize(d):
    x=d.values
    transformer=sklearn.preprocessing.Binarizer().fit(x)
    x_binarized=transformer.transform(x)
    bdf=pd.DataFrame(x_binarized)
    return bdf
print(d)
bldf=binarize(d)
print("after binarize",bldf)

def songs(bldf,r):
    count=-1
    songvector=[]
    l=bldf.iloc[int(r)+1].tolist()
    for i in l:
        count=count+1
        if(i==1):
            songvector.append(count)
    return songvector

rvdfl=[]
rv200dfl=[]
testdfl=[]

tedf1=pd.read_csv('TestUIMatrix.csv')
tedft=tedf1.T
testlist=[]
testlist=tedft.iloc[0].tolist()
tlist=[]
for i in testlist:
    tlist.append(int(i))

for i in range(len(tlist)):
    all_recommend_vector=[]
    top200_recommend_vector=[]
    k=0
    test=tedf.iloc[i]
    print(len(test))
    testdfl.append(tedf.iloc[i])
    recommend_vector=[]
    r=cluster_number

    print(tlist[i])
    for j in matrix[r[i]]:  
        all_recommend_vector=all_recommend_vector+songs(bldf,j)
    all_recommend_vector=list(set(all_recommend_vector))
    all_len=len(all_recommend_vector)
    if(all_len<200):
        top200_recommend_vector=all_recommend_vector
    else:
        top200_recommend_vector=all_recommend_vector[0:200]
    rvdfl.append(all_recommend_vector)
    rv200dfl.append(top200_recommend_vector)
    print("User",tlist[i],"all:",all_recommend_vector)
    print("\n")
    print("User",tlist[i],"top200:",top200_recommend_vector)
    print()
len_rvdfl=len(rvdfl)

print(len_rvdfl)


------------------------------------------nine------------------------------------------------
'''prfm'''
def listtobinary(listpassed):
    temp=listpassed
    for i in range(0, len(temp)):
        if (temp[i] != 0):
            temp[i] = 1
    return temp
def call_for_precision(rvdfl_row,test_row):
    tp,tn,fp,fn=(0,0,0,0)
    print(len(rvdfl_row))
    print(len(test_row))
    bin_test_row= listtobinary(test_row)
    for j in range(len(rvdfl_row)):
        if(j in rvdfl_row and (bin_test_row[j]==1)):
            tp+=1
        elif(j in rvdfl_row and (bin_test_row[j]==0)):
            fn+=1
        elif(j not in rvdfl_row and (bin_test_row[j]==1)):
            fp+=1
        elif(j not in rvdfl_row and (bin_test_row[j]==0)):
            tn+=1
        else:
            pass
    print(tp,tn,fp,fn)
    return (tp,tn,fp,fn)

tp,tn,fp,fn=(0,0,0,0)
sum_of_precision=0
sum_of_recall=0
sum_of_fm=0
sum_of_accuracy=0
avg_of_precision=0
avg_of_recall=0
avg_of_fm=0
avg_of_accuracy=0
for i in (0,60):
    precision=0
    recall=0
    fm=0
    accuracy=0
    tp,tn,fp,fn=call_for_precision(rv200dfl[i],tedf1.iloc[i])
    if((tp==0) and (fp==0)):
        precision=0
        print("p")
    else:
        precision =(tp / (tp+fp))
        print("pe")
    if ((tp==0) and (fn==0)):
        recall=0
        print("r")
    else:
        recall = (tp/(tp+fn))
        print("re")
    if (precision==0 and recall==0):
        print("pr")
        pass
    else:
        fm=(2*precision*recall)/ (precision+recall)
        print("fe")
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    prfm=[precision,recall,fm,accuracy]
    sum_of_precision+=precision
    sum_of_recall+=recall
    sum_of_fm+=fm
    sum_of_accuracy+=accuracy
    conmatrix=[tp,tn,fp,fn]
    print("User",tlist[i],prfm)
    
avg_of_precision =sum_of_precision/60
avg_of_recall=sum_of_recall/60
avg_of_fm=sum_of_fm/60
avg_of_accuracy=sum_of_accuracy/60

print(avg_of_precision)
print(avg_of_recall)
print(avg_of_fm)
print(avg_of_accuracy)

-------------------------------------------ten-------------------------------
done.
