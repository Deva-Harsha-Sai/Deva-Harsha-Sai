#include<stdio.h>
int FR[4]={-1,-1,-1,-1};
int N = 4, front = -1, rear = -1;
void enqueue(int x){
	if(rear==N-1){
		return;
	}
	if(front = -1){
		front = 0;
	}
	rear = (rear+1)%4;
	FR[rear]= x;
	return;
}

void dequeue(){
	if(front==-1){
		return;
	}
	if(front== rear){
		FR[front]=-1;
		front = -1 ;
		rear = -1;
		return;
	}
	FR[front]= -1;
	front = (front+1)%4;
	return;
}

int main(){
	int i,j,hit=0,pagefault=0;
	int PL[10]={1,2,1,3,4,3,6,1,2,3};
	
	for (i=0;i<10;i++){
		for(j=0;j<N;j++){
			if(PL[i]==FR[j]){
				hit= hit + 1;
				break;
			}
			else{
				pagefault++;
				dequeue();
				enqueue(PL[i]);
				break;
			}
		}
	}
	printf("%d",&hit);
}
