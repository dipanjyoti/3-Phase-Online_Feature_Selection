
def Sort(F,i):
	return (sorted(F))#,key=lambda x:x[i]))
def Max(F,i):
	return (max(F,key=lambda x:x[i])[i])
def Min(F,i):
	return (min(F,key=lambda x:x[i])[i])

def crowding_distance(Font):
	F=[]
	F=Font
	distance=[]
	for i in range(len(F)):
		distance.append(0)
	noOfObjectives=2

	for i in range (noOfObjectives):
		F_new=Sort(F,i)
		if Max(F_new,i)==Min(F_new,i):
			continue
		distance[0]=99999.0
		distance[len(F)-1]=99999.0
		for j in range (1,len(F)-1):
			distance[j]+=(float)(F_new[j+1][i]-F_new[j-1][i])/((Max(F_new,i)-Min(F_new,i))+0.0)
	return distance

def GenerateSolution(Fonts):
    global F1_len

    k=nPOP
    distance=[]
    i=0
    x=0
    F=[]
    F=Fonts
    while True:
	    if i>=len(F):
		    return
	    else:
		    if len(F[i])>k:
			    distance=crowding_distance(F[i])
			    e=dict()
			    
			    for j in range(len(F[i])):
				    e[distance[j]]=j

			    distance.sort(reverse=True)
	
			    for j in range (k):
				    sol[x]=F[i][e.get(distance[j])]
				    x=x+1
			    break
		    else:
			    for j in range(len(F[i])):
				    sol[x]=F[i][j]
				    x=x+1
			    k=k-len(F[i])
	    if i==0:
		    F1_len=len(F[0])
	    i=i+1

def non_dominating(P):
	
	population1=[]
	for i in range (nPOP):
		population1.append(tuple(P[i]))

	F_one_set=[]
	for i in range(nPOP): #2*nPOP
		F_one_set.append([])
	ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=population1)
	for i in range(nPOP): #2*nPOP
		F_one_set[ndr[i]].append(population1[i])
	F_one_set2 = [x for x in F_one_set if x != []]
	return F_one_set2

def Mutual_Info(CF_set, SF_set,p,q):
	a=(CF_set[:,p])
	b=(SF_set[:,q])

	nmi=normalized_mutual_info_score(a,b)
	return nmi

def get_selected(data,x):
	x=np.array(x)
	x_sel=data[:,x>=0.25]
	return x_sel

def get_not_selected(data,x):
	x=np.array(x)
	x_not_sel=data[:,x<0.25]
	return x_not_sel

def fun1(sel,not_sel):
	n_sel=len(sel[0])
	n_not_sel=len(not_sel[0])
	beta=0.0
	for i in range (n_not_sel):
		for j in range (n_sel):
			beta+=Mutual_Info(not_sel,sel,i,j)

	if (n_sel==0 or n_not_sel==0):
		beta=0
		print ('not_sel or n_not_sel is Zero')
	else:
		beta/=(n_not_sel*n_sel)

	alpha=0.0
	for i in range (n_sel):
		for j in range (i+1,n_sel):
			alpha+=Mutual_Info(sel,sel,i,j)

	alpha =alpha/((n_sel*(n_sel-1))/2)
	return alpha-beta

#correlation
def fun2(sel_feat_subset):
	f2f_cor=0   #feature to feature correlation
	f2c_cor=0   #feature to class correlation
	n_sel_f=len(sel_feat_subset[0])

	for i in range(n_sel_f):
		for j in range(i+1,n_sel_f):
			f2f_cor+=Mutual_Info(sel_feat_subset,sel_feat_subset,i,j)

	for i in range(n_sel_f):
		for j in range(number_of_classes):
			f2c_cor+=Mutual_Info(sel_feat_subset,class_set,i,j)

	correlation=f2f_cor/f2c_cor
	return correlation

def costFunction(solution,feat_subset):
	X_sel=get_selected(feat_subset,solution)
	X_not_sel=get_not_selected(feat_subset,solution)

	F1 = fun1(X_sel,X_not_sel)
	F2 = fun2(X_sel)
	F=(F1,F2)
	return F


def partition(B,data):

    #data = data[0:2000,:] # need to remove

    U=list(range(len(data)))
    P=[]
    while (len(U)>0):
        XB=[]
        X=[data[U[0]][index] for index in B]
        XB.append(U[0])
        U.pop(0)
        j=0
        while (len(U)>0) and (len(U)) !=j:
            Y=[data[U[j]][index] for index in B]
            if (X==Y):
                XB.append(U[j])
                U.pop(j)
            else:
                j=j+1
        P.append(list(set(XB)))

    return P

def gamma_C_D(C,D,data):
    P1=partition(C,data)
    P2=partition(D,class_set)
    S=[]
    T1=0
    for i in range(len(P1)):
        x=set(P1[i])
        for j in range(len(P2)):
            y=set(P2[j])
            if (x.issubset(y)):
                S.append(list(x))

    for i in range(len(S)):
        T1=T1+len(S[i])
    return T1

def sigma(RR,D,g,data):

    R=[]
    for i in range(len(RR)):
        R.append(RR[i])

    I1=gamma_C_D(R,D,data)
    R.pop(g)
    I2=gamma_C_D(R,D,data)
    sigma_C_D_f=(I1-I2)
    return sigma_C_D_f

def non_significant(RR,f,D,data): #R, F contains the insdx of the features selected in SF and RF respectively
    
    B=[]
    T=[]
    R=[]

    for i in range(len(RR)):
        R.append(RR[i])

    for i in range(len(RR)):
        T.append(RR[i])

    T.remove(f)

    while (len(T)!=0):

        if len(T)>1:
            g=random.randint(0,len(T)-1)
        else:
            g=0
        if sigma(R,D,g,data)==0:
            B.append(T[g])
            R.pop(g)
        T.pop(g)

    return B


