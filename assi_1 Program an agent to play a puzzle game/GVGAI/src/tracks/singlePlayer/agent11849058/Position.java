package tracks.singlePlayer.agent11849058;

public class Position {
    public int x;
    public int y;
    public Position target;
    public Position front;

    public Position(int x, int y){
        this.x = x;
        this.y = y;
    }

    public void setTarget(Position target) {
        this.target = target;
    }

    public void setFront(Position front){
        this.front = front;
    }

    public Position getFront() {
        return front;
    }

    public int getLengthToTarget(){
        return Math.abs(this.x - target.x)+Math.abs(this.y - target.y);
    }

    public boolean equals(Position p){
        if(this.x == p.x && this.y == p.y){
            return true;
        }else{
            return false;
        }
    }
}
