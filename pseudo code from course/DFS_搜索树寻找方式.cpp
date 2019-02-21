DFS(TreeNode node){
	Queue q;
	q.push(node);
	TreeNode n;
	TreeNode visited[];
	while(!q.isEmpty()){
		n = q.pop();
		if(n == target){
			return n;
		}
		for(TreeNode nd = n.child; nd != null; nd = nd.next()){
			if(!visited[].has(nd)){
				q.push(nd);
			}
		}
	}
}