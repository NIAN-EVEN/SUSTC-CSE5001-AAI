package tracks.singlePlayer.agent11849058;

import java.util.ArrayList;
import java.util.Queue;


import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tracks.multiPlayer.tools.heuristics.SimpleStateHeuristic;

import javax.swing.plaf.nimbus.State;

import tracks.singlePlayer.agent11849058.StateGraph.sprit;

/**
 * Created with IntelliJ IDEA.
 * User: ssamot
 * Date: 14/11/13
 * Time: 21:45
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Agent extends AbstractPlayer {

    public static Queue<Types.ACTIONS> actions;
    private static int path;
    private static boolean findKey;
    private static boolean findDoor;

    private static StateGraph state;


    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer) {
//        actions = new LinkedList<Types.ACTIONS>();
        path = 0;
        findKey = false;
        findDoor = false;
        state = new StateGraph(so);

    }

    /**
     * Picks an action. This function is called every game step to request an
     * action from the player.
     * @param stateObs Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    @Override
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {

        //计算时间部分参考自tracks.singlePlayer.simple.sampleRamdom.Agent
        long remaining = elapsedTimer.remainingTimeMillis();
        int remainingLimit = 5;

        int numIters = 0;
        double avgTimeTaken = 0;
        double acumTimeTaken = 0;


        while(remaining > 2*avgTimeTaken && remaining > remainingLimit)  {
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            //首先找到钥匙的路径，如果没有到钥匙的路径则找能够一下推进洞的箱子并返回这个路径
            if(!findKey){
                //没有找到钥匙的状态，目标是钥匙，尝试寻找钥匙
                findKey = state.findPathTo(sprit.KEY);
                if(!findKey) {
                    state.moveBox();
                }
            }else if(!findDoor){
                //找到钥匙以后，目标是门，一次性返回找到门的行动集合
                findDoor = state.findPathTo(sprit.DOOR);
                if(!findDoor) {
                    state.moveBox();
                }
            }
            numIters++;
            acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
            //System.out.println(elapsedTimerIteration.elapsedMillis() + " --> " + acumTimeTaken + " (" + remaining + ")");
            avgTimeTaken  = acumTimeTaken/numIters;
            remaining = elapsedTimer.remainingTimeMillis();
        }
        //如果队列里有元素则返回队列元素，否则返回空动作
        return StateGraph.actions.size()>0?StateGraph.actions.poll():stateObs.getAvailableActions().get((int)(4*Math.random()));
    }

}
