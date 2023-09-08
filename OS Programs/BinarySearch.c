#include<stdio.h>
#include<stdlib.h>
int main()
{
    int a[10],n,key,i;
    int l=a[0];
    int h=a[n];
    int mid=(l+h)/2;
    printf("enter no of elements of the array : ");
    scanf("%d",&n);
    printf("enter the elements of the array : ");
    for(i=0;i<n;i++)
      scanf("%d",&a[i]);
    printf("enter the key element to search : ");
    scanf("%d",&key);
    if(key==mid)
      printf("element found at %d position ",mid);
    if(key>mid)
    {
        for(i=mid+1;i<n;i++)
          if(a[i]==key)
             printf("key found at %d position",i);
          else
             i++;
    }
    else
    {
        for(i=mid;i>=0;i--)
            if(a[i]==key)
               printf("element found at %d position : ",i);
            else
               i--;
    }
}
