package tracks.singlePlayer.tools;


import java.util.LinkedList;
import java.util.Queue;

public class LearnJava {
    public static void main(String[] args){
        Queue<String> queue = new LinkedList<>();
        if(queue == null){
            System.out.println("null");
        }
        if(queue.size()==0){
            System.out.println("queue==0");
        }
    }
}
