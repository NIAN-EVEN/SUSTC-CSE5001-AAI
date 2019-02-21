TreeNd Bi_diratical_search(TreeNd start,TreeNd finish){
	start.isLastOne = true;
	finish.isLastOne = true;//用于判断该节点是否是本层最后一个节点
	int visited[];//用于记录某个节点是被正向扩展还是反向扩展的
	visited[start] = 1;
	visited[finish] = 2;
	int startStep = 0;
	int finishStep = 0;
	Queue frontSearch;
	Queue backSearch;
	frontSearch.push(start);
	backSearch.push(finish);
	while(!frontSearch.isEmpty() || !backSearch.isEmpty()){
		TreeNd node;
		if(!frontSearch.isEmpty()){
			//正向扩展
			//从i层到i+1层扩展
			startStep++;
			do{
				node = frontSearch.pop()
				if(visited[node] == 2){
					return node;
				}
				visited[node] = 1;
				for(TreeNd nd = node.child; nd != null; nd = ne->next){
					frontSearch.push(nd);
				}
			}while(!node.step)
			node = frontSearch.pop();
			node.step = true;
			frontSearch.push(nd);
		}		
		//反向扩展
		if(!backSearch.isEmpty()){
			//正向扩展
			//从i层到i+1层扩展
			startStep++;
			do{
				node = backSearch.pop()
				if(visited[node] == 2){
					return node;
				}
				visited[node] = 1;
				for(TreeNd nd = node.child; nd != null; nd = ne->next){
					backSearch.push(nd);
				}
			}while(!node.step)
			node = backSearch.pop();
			node.step = true;
			backSearch.push(nd);
		}
	}
		
}