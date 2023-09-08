//BubbleSort
#include<stdio.h>
int main(){
	int a[20],i,n,temp;
	printf("Enter Size Of Array: ");
    scanf("%d",&n);
    printf("Enter %d elements",n);
    for(i=0;i<n;++i)
        scanf("%d",&a[i]);
    for(i=0;i<n;i++)
    	if(a[i]>a[i+1])
    		a[i]=a[i+1];
    		a[i+1]=temp;
    		temp=a[i];
    for(i=0;i<n;i++)
    	printf("%d ",a[i]);
}
