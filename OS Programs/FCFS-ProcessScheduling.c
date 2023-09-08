//FCFS CPU SCHEDULING ALGORITHM
#include<stdio.h>
#include<stdlib.h>
int main()
{
	int p[10],at[10],bt[10],ct[10],tat[10],wt[10],i,j,temp=0,n;
	float awt=0, atat=0;
	printf("Enter the number of processes: ");
	scanf("%d",&n);
	printf("Enter %d processes: ",n);
	for(i=0; i<n; i++)
	{
		scanf("%d",&p[i]);
	}
	printf("Enter %d Arrival times: ",n);
	for(i=0; i<n; i++)
	{
		scanf("%d",&at[i]);
	}
	printf("Enter %d Burst Time: ",n);
	for(i=0; i<n; i++)
	{
		scanf("%d",&bt[i]);
	}
	//Calculating CT
	ct[0]=at[0]+bt[0];
	for(i=1;i<n;i++)
	{
		temp=0;
		if(ct[i-1]<at[i])
		{
			temp=at[i]-ct[i-1];
		}
		ct[i]=ct[i-1]+bt[i]+temp;
	}
	printf("\np\t A.T\t B.T\t C.T\t TAT\t WT");
	for(i=0; i<n; i++)
	{
		tat[i]=ct[i]-at[i];
		wt[i]=tat[i]-bt[i];
		atat+=tat[i];
		awt+=wt[i];
	}
	atat=atat/n;
	awt=awt/n;
	for(i=0;i<n;i++)
	{
		printf("\nP%d\t %d\t %d\t %d\t %d\t %d",p[i],at[i],bt[i],ct[i],tat[i],wt[i]);
	}
	printf("\nAverage Turnaround Time Is %f",atat);
	printf("\nAverage Waitingtime is %f",awt);
	return 0;
}
