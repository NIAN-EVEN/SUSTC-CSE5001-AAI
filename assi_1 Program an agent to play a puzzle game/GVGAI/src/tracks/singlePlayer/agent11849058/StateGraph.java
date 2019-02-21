package tracks.singlePlayer.agent11849058;

import core.game.StateObservation;
import core.game.Observation;
import ontology.Types;

import java.lang.reflect.Array;
import java.util.Comparator;
import java.util.PriorityQueue;

import java.util.*;

public class StateGraph {

    /**
     * 枚举类型，分别代表地图上的元素
     */
    public enum sprit{
        //代表地板，箱子，洞，墙，蘑菇，钥匙，门，人
        WALL(0), HOLE(3), AVATAR(4), MUSHROOM(6), KEY(7), DOOR(8), BOX(9), FLOOR(2), MIS(-1);
        private final int index;
        private sprit(int index){
            this.index = index;
        }
        public sprit indexOf(int index){
            switch(index){
                case 0: return WALL;
                case 3: return HOLE;
                case 4: return AVATAR;
                case 6: return MUSHROOM;
                case 7: return KEY;
                case 8: return DOOR;
                case 9: return BOX;
                case 2: return FLOOR;
                default:return MIS;
            }
        }
    }

    /**
     * 从stateobs里抓下来的地图
     */
    public static  sprit[][] graph;

    //用于存储箱子的位置
    private static ArrayList<Position> boxesPos;
    private static  ArrayList<Position> holePos;
    /**
     * 用于存储小人可以到达的位置
     */
    private static  ArrayList<Position> availPos;

    //存储四个特殊点的位置
    private static  Position avaPos = null;
    private static  Position keyPos = null;
    private static  Position doorPos = null;
    private static  Position mhrPos = null;
    //动作执行序列
    public static Queue<Types.ACTIONS> actions;

    /**
     * 构造函数，构造可执行的
     * @param stateObs
     */
    public StateGraph(StateObservation stateObs){
        actions = new LinkedList<Types.ACTIONS>();
        formGraph(stateObs);
        formOthers(graph);
        BFSformAvailPos(avaPos);
        //graph = new int[][];
        //获取stateObs的数值，构建二维数组
    }

    /**
     * 从obj中抓取需要的信息存储
     */
    private void formGraph(StateObservation stateObs){
        ArrayList<Observation> grid[][] = stateObs.getObservationGrid();
        boxesPos = new ArrayList<Position>();
        holePos = new ArrayList<Position>();
        graph = new sprit[grid.length][grid[0].length];
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[i].length; j++){
                //将地图信息存储到数组中
                if(grid[i][j].size() == 0){
                    graph[i][j] = sprit.FLOOR;
                }else{
                    graph[i][j] = sprit.FLOOR.indexOf(grid[i][j].get(0).itype);
                }
            }
        }
    }

    /**
     * 储存特殊点的位置方便以后的操作
     * @param graph
     */
    private void formOthers(sprit[][] graph){
        //将过后需要遍历的东西存储到相应的list当中
        for(int i = 0; i < graph.length; i++){
            for(int j = 0; j < graph[i].length; j++){
                Position p = new Position(i,j);
                switch(graph[i][j]){
                    case BOX: boxesPos.add(p);break;
                    case AVATAR: avaPos = p; break;
                    case KEY: keyPos = p; break;
                    case MUSHROOM: mhrPos = p; break;
                    case DOOR: doorPos = p; break;
                    case HOLE: holePos.add(p);break;
                }
            }
        }
    }

    /**
     * 通过BFS的方式获取玩家的可达域
     */
    private void BFSformAvailPos(Position pos){
        availPos = new ArrayList<Position>();
        //使用队列存储未访问过的子节点
        Queue<Position> queue = new LinkedList<Position>();
        Position p;
        queue.offer(pos);
        while(queue.size()>0){
            p = queue.poll();
            availPos.add(p);
            pointPermit(queue,p.x-1,p.y);
            pointPermit(queue,p.x+1,p.y);
            pointPermit(queue,p.x,p.y-1);
            pointPermit(queue,p.x,p.y+1);
        }
    }

    /**
     * 判断某个点在BFS的过程中是否在已经遍历的集合中存在
     * @param queue
     * @param x
     * @param y
     */
    private void pointPermit(Queue<Position> queue,int x,int y){
        if(graph[x][y] == sprit.FLOOR || graph[x][y] == sprit.KEY || graph[x][y] == sprit.MUSHROOM  || graph[x][y] == sprit.DOOR || graph[x][y] == sprit.AVATAR){
            Position p = new Position(x,y);
            for(Position p1 : availPos){
                if(p1.equals(p)){
                    return;
                }
            }
            queue.offer(p);
        }
    }

    /**
     * 判断钥匙和目标是否在寻找范围内的函数
     * @param target
     * @return
     */
    public boolean findPathTo(sprit target){
        //寻找到target的路径
        switch(target){
            case KEY:
                if(AcontainsB(availPos,keyPos)){
                    fromWayTo(keyPos);
                    return true;
                }else{
                    return false;
                }
            case DOOR:
                availPos.add(doorPos);
                if(AcontainsB(availPos,doorPos)){
                    fromWayTo(doorPos);
                    return true;
                }else{
                    return false;
                }
            default: System.out.println("not find target");return false;
        }
    }

    public void moveBox(){
        //只做两种事儿，
        //1.地图上没有洞，推箱子到空的地方
        if(mhrPos!=null && AcontainsB(availPos,mhrPos)){
            fromWayTo(mhrPos);
        }
        if(holePos.isEmpty()){
            for(Position p : boxesPos){
                //箱子向右推
                Position p1 = new Position(p.x-1,p.y);
                if(graph[p.x+1][p.y]==sprit.FLOOR && AcontainsB(availPos,p1)){
                    //人走到箱子前面
                    fromWayTo(p1);
                    actions.offer(Types.ACTIONS.ACTION_RIGHT);
                    avaPos.x = p.x;
                    avaPos.y = p.y;
                    graph[p.x][p.y]=sprit.FLOOR;
                    graph[p.x+1][p.y]=sprit.BOX;
                    p.x+=1;
                    p1 = new Position(p.x+1,p.y);
                    if(AcontainsB(availPos,p1)){
                        AremoveB(availPos,p1);
                    }
                    BFSformAvailPos(avaPos);
                    continue;
                }
                //箱子向左推
                p1 = new Position(p.x+1,p.y);
                if(graph[p.x-1][p.y]==sprit.FLOOR && AcontainsB(availPos,p1)){
                    fromWayTo(p1);
                    actions.offer(Types.ACTIONS.ACTION_LEFT);
                    avaPos.x = p.x;
                    avaPos.y = p.y;
                    graph[p.x][p.y]=sprit.FLOOR;
                    graph[p.x-1][p.y]=sprit.BOX;
                    p.x-=1;
                    p1 = new Position(p.x-1,p.y);
                    if(AcontainsB(availPos,p1)){
                        AremoveB(availPos,p1);
                    }
                    BFSformAvailPos(avaPos);
                    continue;
                }
                //箱子向上推，因为记录的地图和实际的地图相反所以是DOWN
                p1 = new Position(p.x,p.y-1);
                if(graph[p.x][p.y+1]==sprit.FLOOR && AcontainsB(availPos,p1)){
                    fromWayTo(p1);
                    actions.offer(Types.ACTIONS.ACTION_DOWN);
                    avaPos.x = p.x;
                    avaPos.y = p.y;
                    graph[p.x][p.y]=sprit.FLOOR;
                    graph[p.x][p.y+1]=sprit.BOX;
                    p.y+=1;
                    p1 = new Position(p.x,p.y+1);
                    if(AcontainsB(availPos,p1)){
                        AremoveB(availPos,p1);
                    }
                    BFSformAvailPos(avaPos);
                    continue;
                }
                //箱子向下推
                p1 = new Position(p.x,p.y+1);
                if(graph[p.x][p.y-1]==sprit.FLOOR && AcontainsB(availPos,p1)){
                    fromWayTo(p1);
                    actions.offer(Types.ACTIONS.ACTION_UP);
                    avaPos.x = p.x;
                    avaPos.y = p.y;
                    graph[p.x][p.y]=sprit.FLOOR;
                    graph[p.x][p.y-1]=sprit.BOX;
                    p.y-=1;
                    p1 = new Position(p.x,p.y-1);
                    if(AcontainsB(availPos,p1)){
                        AremoveB(availPos,p1);
                    }
                    BFSformAvailPos(avaPos);
                    continue;
                }
            }
        }else{
            //1.地图上有洞，把它推过去，两个格变成地板，BFS增加可到达区域
            //为石头找路径的办法
            Position tb = boxesPos.get(0);
            Position th = holePos.get(0);
            tb.setTarget(th);
            int minDis = Integer.MAX_VALUE;
            for(Position p : boxesPos){
                //把箱子向洞的地方推，首先遍历距离箱子最近的洞
                for(Position t:holePos){
                    p.setTarget(t);
                    //如果箱子之间有通路,且箱子周围人可以推的地方有通路
                    if(minDis>=p.getLengthToTarget() && !A_star(tb,th).isEmpty()){
                        tb = p;
                        th = t;
                        minDis = p.getLengthToTarget();
                    }
                }
            }
            //如果可以挪动，只做横挪和竖着挪
            //向右边挪的情况
            if(tb.x < th.x){
                Position p1 = new Position(tb.x-1,tb.y);
                if(AcontainsB(availPos,p1)){
                    //首先判断人可以到箱子后面
                    boolean flag = true;
                    for(int i = tb.x+1; i <= th.x; i++){
                        if(graph[i][tb.y] == sprit.BOX || graph[i][tb.y] == sprit.WALL || graph[i][tb.y] == sprit.KEY ||graph[i][tb.y] == sprit.DOOR){
                            flag = false;
                            break;
                        }
                    }
                    //判断箱子平移是否会遇到障碍，如果没遇到则移动箱子
                    if(flag){
                        Position p2 = new Position(th.x-1,tb.y);
                        fromWayTo(p1);
                        availPos.add(tb);
                        graph[tb.x][tb.y]=sprit.FLOOR;
                        AremoveB(boxesPos,tb);
                        fromWayTo(p2);
                        if(tb.y==th.y){
                            graph[th.x][th.y]=sprit.FLOOR;
                            AremoveB(holePos,th);
                            BFSformAvailPos(new Position(th.x,tb.y));
                        }else{
                            boxesPos.add(new Position(th.x,tb.y));
                            graph[th.x][tb.y] = sprit.BOX;
                        }
                    }
                }
            }else if(tb.x > th.x){
                //向左边挪的情况
                Position p1 = new Position(tb.x+1,tb.y);
                if(AcontainsB(availPos,p1)){
                    //首先判断人可以到箱子右面
                    boolean flag = true;
                    for(int i = tb.x-1; i >= th.x; i--){
                        if(graph[i][tb.y] == sprit.BOX || graph[i][tb.y] == sprit.WALL || graph[i][tb.y] == sprit.KEY ||graph[i][tb.y] == sprit.DOOR){
                            flag = false;
                            break;
                        }
                    }
                    //判断箱子平移是否会遇到障碍，如果没遇到则移动箱子
                    if(flag){
                        Position p2 = new Position(th.x+1,tb.y);
                        fromWayTo(p1);
                        availPos.add(tb);
                        graph[tb.x][tb.y]=sprit.FLOOR;
                        AremoveB(boxesPos,tb);
                        fromWayTo(p2);
                        if(tb.y==th.y){
                            graph[th.x][th.y]=sprit.FLOOR;
                            AremoveB(holePos,th);
                            BFSformAvailPos(new Position(th.x,tb.y));
                        }else{
                            boxesPos.add(new Position(th.x,tb.y));
                            graph[th.x][tb.y] = sprit.BOX;

                        }
                    }
                }
            }else{
                //两者相等的情况
                if(tb.y > th.y){
                    //向下边挪的情况
                    Position p1 = new Position(tb.x,tb.y+1);
                    if(AcontainsB(availPos,p1)){
                        //首先判断人可以到箱子后面
                        boolean flag = true;
                        for(int i = tb.y-1; i >= th.y; i--){
                            if(graph[tb.x][i] == sprit.BOX || graph[tb.x][i] == sprit.WALL || graph[tb.x][i] == sprit.KEY ||graph[tb.x][i] == sprit.DOOR){
                                flag = false;
                                break;
                            }
                        }
                        //判断箱子平移是否会遇到障碍，如果没遇到则移动箱子
                        if(flag){
                            Position p2 = new Position(tb.x,th.y+1);
                            fromWayTo(p1);
                            availPos.add(tb);
                            graph[tb.x][tb.y]=sprit.FLOOR;
                            AremoveB(boxesPos,tb);
                            fromWayTo(p2);
                            graph[th.x][th.y]=sprit.FLOOR;
                            AremoveB(holePos,th);
                            BFSformAvailPos(new Position(th.x,th.y));
                        }
                    }
                }else if(tb.y < th.y){
                    //向上边挪的情况
                    Position p1 = new Position(tb.x,tb.y-1);
                    if(AcontainsB(availPos,p1)){
                        //首先判断人可以到箱子后面
                        boolean flag = true;
                        for(int i = tb.y+1; i <= th.y; i++){
                            if(graph[tb.x][i] == sprit.BOX || graph[tb.x][i] == sprit.WALL || graph[tb.x][i] == sprit.KEY ||graph[tb.x][i] == sprit.DOOR){
                                flag = false;
                                break;
                            }
                        }
                        //判断箱子平移是否会遇到障碍，如果没遇到则移动箱子
                        if(flag){
                            Position p2 = new Position(tb.x,th.y-1);
                            fromWayTo(p1);
                            availPos.add(tb);
                            graph[tb.x][tb.y]=sprit.FLOOR;
                            AremoveB(boxesPos,tb);
                            fromWayTo(p2);
                            graph[th.x][th.y]=sprit.FLOOR;
                            AremoveB(holePos,th);
                            BFSformAvailPos(new Position(th.x,th.y));
                        }
                    }
                }

            }
        }
    }

    public void fromWayTo(Position pos){
        LinkedList<Position> path = A_star(avaPos,pos);
        Position p = path.getFirst();
        path.removeFirst();
        while(!path.isEmpty()){
            Position p1 = path.getFirst();
            path.removeFirst();
            if(p1.x == p.x+1 && p1.y == p.y){
                actions.offer(Types.ACTIONS.ACTION_RIGHT);
            }
            if(p1.x == p.x-1 && p1.y == p.y){
                actions.offer(Types.ACTIONS.ACTION_LEFT);
            }
            if(p1.x == p.x && p1.y == p.y+1){
                actions.offer(Types.ACTIONS.ACTION_DOWN);
            }
            if(p1.x == p.x && p1.y == p.y-1){
                actions.offer(Types.ACTIONS.ACTION_UP);
            }
            p = p1;
        }
        graph[pos.x][pos.y] = sprit.FLOOR;
        avaPos = pos;
    }


    /**
     * A-star算法，记录从小人位置找到目标位置的路径并返回给formWayTo函数转换成相应的操作
     * @param from to
     * @return
     */
    public LinkedList<Position> A_star(Position from,Position to){
        LinkedList<Position> path = new LinkedList<Position>();
        //A-star算法搜索到达目标的路径并存储到actions中
        Queue<Position> queue = new PriorityQueue<Position>(graph.length*graph[0].length, posComparator);
        ArrayList<Position> accPos = new ArrayList<Position>();
        Position p;
        queue.offer(from);
        while(!queue.isEmpty()){
            p = queue.poll();
            //将p存储到到达p的路径当中
            while(!path.isEmpty() && !path.getLast().equals(p.getFront())){
                path.removeLast();
            }
            path.add(p);
            if(p.equals(to)){
                //如果到达目标节点,则退出循环
                break;
            }
            //否则访问过p节点
            accPos.add(p);
            //层次增加p的子节点
            addPoint(p.x+1,p.y,p,to,accPos,queue);
            addPoint(p.x-1,p.y,p,to,accPos,queue);
            addPoint(p.x,p.y+1,p,to,accPos,queue);
            addPoint(p.x,p.y-1,p,to,accPos,queue);
        }
        return path;
    }

    private void addPoint(int x,int y,Position ft,Position pos,ArrayList<Position> accPos,Queue<Position> queue){
        if(graph[x][y] == sprit.FLOOR || graph[x][y] == sprit.KEY || graph[x][y] == sprit.MUSHROOM  || graph[x][y] == sprit.DOOR || graph[x][y] == sprit.AVATAR){
            Position p = new Position(x,y);
            p.setFront(ft);
            p.setTarget(pos);
            if(AcontainsB(accPos,p)){
                return;
            }
            queue.offer(p);
        }
    }

    private static Comparator<Position> posComparator = new Comparator<Position>(){
        @Override
        public int compare(Position p1, Position p2 ){
            return p1.getLengthToTarget() - p2.getLengthToTarget();
        }
    };

    private boolean AcontainsB(ArrayList<Position> A, Position B ){
        for(Position p : A){
            if(p.equals(B)){
                return true;
            }
        }
        return false;
    }

    private void AremoveB(ArrayList<Position> A, Position B ){
        for(Position p : A){
            if(p.equals(B)){
                A.remove(p);
                return;
            }
        }
    }
}
