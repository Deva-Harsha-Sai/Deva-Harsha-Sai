#include<stdio.h>
int n = 4;
int frame[4];
int front = -1, rear = -1;
void enqueue(int d)
{
	if(rear==n-1)
	{
		return;
	}
	else if(rear==-1)
	{
		rear++;
		front++;
		frame[rear]=d;
	
		return;
	}
	else
	{
		rear++;
		frame[rear]=d;
		return;
	}
}
void dequeue()
{
	if(front==-1) 
	{
		return;
	}
	else if (front==rear)
	{
		front=-1;
		rear=-1;
	}
	else
	{
		front++;
		return;
	}
}

int main(){
	int i,j,hit=0,pagefault=0;
	int pagelist[]={1,2,3,2,4,6,2,3,1,5};  
	for(i=0;i<10;i++){
		int flag = 0;
		for(j=0;j<n;j++){
			
			if(frame[j]==pagelist[i]){
				hit++;
				break;
			}
			else{
				flag = 1;
				break;
			}
		}
		if (flag == 1){
			pagefault++;
			enqueue(pagelist[i]);
		}
	}
	printf("%d",hit);
}

