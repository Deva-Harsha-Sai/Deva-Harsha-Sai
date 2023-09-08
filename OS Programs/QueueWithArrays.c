//Queues with Arrays
#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
int a[10], front=-1, rear=-1;
void enqueue(int d)
{
	if(rear==9)
	{
		printf("\n Queue is full\n");
		return;
	}
	else if(rear==-1)
	{
		rear++;
		front++;
		a[rear]=d;
	
		return;
	}
	else
	{
		rear++;
		a[rear]=d;
		return;
	}
}
void dequeue()
{
	if(front==-1) 
	{
		printf("\n Empty Queue");
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
void display()
{
	int i;
	for(i=front;i<=rear;i++)
	{
		printf("%d ",a[i]);
	}
	return;
}
int main()
{
	int ch, d;
	while(1)
	{
		printf("OPT");
		scanf("%d",&ch);
		switch(ch)
		{
			case 1:
				printf("\n Enter number");
				scanf("%d",&d);
				enqueue(d);
				display();
				break;
			case 2:
				dequeue();
				display();
				break;
			case 3:
				display();
				break;
			case 4:
				exit(0);
		}
	}
}
