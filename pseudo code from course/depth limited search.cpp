TreeNd depth_limited_search(TreeNd node,int limit){
	return recursive_DLS(node,limit);
}

TreeNd recursive_DLS(TreeNd node,int limit){
	if(node == goalNode){
		return node;
	}
	if(limit == 0){
		return cutoff;
	}
	bool cutff_occured = false;
	TreeNd result;
	for(TreeNd nd = node.child; nd != null; nd = nd.next){
		result = recursive_DLS(nd,limit-1);
		if(result == cutoff){
			cutoff = true;
		}else if(result != failure){
			return result;
		}
	}
	if(cutff_occured == true){
		return cutoff;
	}else{
		return failure;
	}
}

TreeNd iterative_deepening_search(TreeNd node){
	int depth = 1;
	TreeNd result;
	while(depth++){
		result = depth_limited_search(node,depth);
		if(result != cutoff){
			return result;
		}
	}
}