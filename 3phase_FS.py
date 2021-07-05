from util import *


#"feature_set" is a matrix of the group of incoming feature


def Phase_One_FS(feat_subset):
	c1 = 2.8
	c2 = 1.3 
	nOF = 2     #no. of functions
	upB = np.array([1]*window_size)
	lowB = np.array([0]*window_size)
	stepOV = np.array([0.1]*window_size)
	nStep = (upB - lowB)/stepOV
	vmax = 2*(upB - lowB)/nStep
	nSim = 0  

	#:::: Initialize population ::::#
	Particles = np.random.random((nPOP, window_size))
	Velocities = np.zeros((nPOP, window_size))
	Fitness = np.zeros(shape=(nPOP, nOF))
	PbestParticles = np.zeros((nPOP, window_size))
	PbestFitness = np.zeros((nPOP, nOF))
	
	for i in range (nPOP):
		Fitness[i,:]=costFunction(Particles[i,:],feat_subset)
		PbestParticles[i,:] = Particles[i,:]
		PbestFitness[i,:] = Fitness[i,:]

	Front=non_dominating(PbestFitness)
	GenerateSolution(Front)
	gbestF=sol[0]
	PbestFitness=PbestFitness.tolist()
	index=PbestFitness.index(list(gbestF))
	gbestP=PbestParticles[index]

	while (nSim<maxSim):

		w = 0.9 -((0.9 - 0.4)/maxSim)*nSim

		for i in range(0, nPOP):

			Velocities[i,:] = w*Velocities[i,:] + random.random()*c1*(PbestParticles[i,:] - Particles[i,:]) + random.random()*c2*(gbestP - Particles[i,:]);
			Particles[i,:] = Particles[i,:] + Velocities[i,:]
			#:::: Check that the particles do not fly out of the search space ::::#
			Particles[i,Particles[i,:] < lowB] = lowB[Particles[i,:] < lowB]
			Particles[i,Particles[i,:] > upB] = upB[Particles[i,:] > upB]
			#:::: Change the velocity direction ::::#
			Velocities[i,Particles[i,:] < lowB] = -Velocities[i,Particles[i,:] < lowB]
			Velocities[i,Particles[i,:] > upB] = -Velocities[i,Particles[i,:] > upB]
			#:::: Constrain the velocity ::::#
			Velocities[i, Velocities[i,:] > vmax] = vmax[Velocities[i,:] > vmax]
			Velocities[i, Velocities[i,:] < -vmax] = vmax[Velocities[i,:] < -vmax]

		PbestFitness=np.asarray(PbestFitness) 

		for i in range(0,nPOP):
			Fitness[i,:]=costFunction(Particles[i,:],feat_subset)
			if (all(Fitness[i,:] <= PbestFitness[i,:]) & any(Fitness[i,:] < PbestFitness[i,:])):
				PbestFitness[i,:] = Fitness[i,:]
				PbestParticles[i,:] = Particles[i,:]
			elif (all(PbestFitness[i,:] <= Fitness[i,:]) & any(PbestFitness[i,:] < Fitness[i,:])):
				pass
			else:
				if (random.randint(0,1) == 0):
					PbestFitness[i,:] = Fitness[i,:]
					PbestParticles[i,:] = Particles[i,:]

		Front=non_dominating(PbestFitness)
		GenerateSolution(Front)
		gbestF=sol[0]
		PbestFitness=PbestFitness.tolist()
		index=PbestFitness.index(list(gbestF))
		gbestP=PbestParticles[index]

		nSim=nSim+1

	sel_feat_subset=get_selected(feat_subset,gbestP)
	sel_feat_subset=np.asarray(sel_feat_subset)
	n_selected=len(sel_feat_subset[0])

	# print("(No. of features selected after 1st phase): ",n_selected)

	return sel_feat_subset


def Phase_Two_FS(Already_selected_features, Phase_1_SF):

	Phase_2_SF=[]


	#Finding the maximum of the corelation b/w any pair of features in the already selected features

	max_thresh=-1
	for c1 in range(0,len(Already_selected_features[0])):
		for c2 in range(c1+1,len(Already_selected_features[0])):
			max_thresh=max(max_thresh, Mutual_Info(Already_selected_features,Already_selected_features,c1,c2))


	# Finding the maximum of the co-relation b/w a features in the current selected set to all the features in the already selected features set

	for ci in range(len(Phase_1_SF[0])):
		cur_max_red=-1
		for pi in range(len(Already_selected_features[0])):
			cur_max_red=max(cur_max_red, Mutual_Info(Phase_1_SF,Already_selected_features,ci,pi))
		if cur_max_red <= max_thresh:
			# get cur feat column and Add cur feature to Phase_2_SF
			f= np.reshape(Phase_1_SF[:,ci],(-1,1))

			if Phase_2_SF==[]:
				Phase_2_SF=f
			else:
				Phase_2_SF=np.hstack((Phase_2_SF,f))
		else:
			#discard ci i.e dont add in Phase_2_SF
			pass

	joint_set=np.hstack((Already_selected_features,Phase_2_SF))

	return joint_set, Phase_2_SF



def Phase_Three_FS(R,F,D,data):

    for i in range(len(F)):

        R.append(F[i]) 
        B=non_significant(R,F[i],D,data) 

        for i in range(len(B)):
            R.remove(B[i]) #Features that are not significant are removed
    return R


##############################################
####### 3-Phase filtering process  ###########
##############################################

# Phase-1 PSO based evolutionary Multi-objective based selection
# "feature_set" is a matrix, consists of set of incoming features in the form of group of size N*G, N= Number of instances, G= number of features in the group
# "Phase_1_SF" is the features selected from the group of incoming features after the first phase

Phase_1_SF= Phase_One_FS(feature_set)

#For the first batch "Already_selected_features" is "Phase_1_SF" itself
#Already_selected_features=Phase_1_SF

# "Phase_2_SF" is the features selected from the group of incoming features after the first and second phases
# "Already_selected_features" is a matrix consisting of features that are selected already selected from the begining till the second phase of the current batch  

Already_selected_features, Phase_2_SF= Phase_Two_FS(Already_selected_features, Phase_1_SF)


# join_Phase_1_2_id=list( range( len(Already_selected_features[0])))
# Already_selected_features_prev_id=join_Phase_1_2_id[:len(join_Phase_1_2_id)-len(Phase_2_SF[0])] # len(Phase_2_SF[0] is the number of features selected after 2nd phase from the group
# Phase_2_id=join_Phase_1_2_id[len(Already_selected_features_prev_id):len(join_Phase_1_2_id)]
# class_id=list( range( N_COL_CLASS ) ) # N_COL_CLASS is the number of labels in the multi-label data set


# "Already_selected_features_prev_id" is a list of indexes of festures (from the feature matrix "Already_selected_features") that are selected till the last group
# "Phase_2_id" is a list of indexes  of features (from the feature matrix "Already_selected_features") that are selected in the current group after the second phase filtering
# "class_id" indexes of all the  classes
# "Already_selected_features" is a matrix consisting of features that are selected already selected from the begining till the second phase of the current batch  
Already_selected_features_index=Phase_Three_FS(Already_selected_features_prev_id,Phase_2_id,class_id,Already_selected_features)
# "Already_selected_features_index" is the indexes (from the feature matrix "Already_selected_features") of the finally selected features


Already_selected_features=pd.DataFrame(data=Already_selected_features, index=None, columns=None)
Already_selected_features=Already_selected_features.iloc[:,Already_selected_features_index]

#"Already_selected_features" is the final seleted features matrix