#include<stdio.h>
#include<conio.h>
void main(){
	int i,j,k,s;
	int m=5,n=3,f[m], need[m][n],ans[5]={0,0,0,0,0},o=0;
	int alloc[5][3]= {{0,1,0}, {2,0,0},{3,0,2},{2,1,1},{0,0,2}};
	int max[5][3]={{7,5,3},{3,2,2},{9,0,2},{2,2,2},{4,3,3}};
	int avail[3]={3,3,2};
	for(i=0;i<m;i++){
		//f[i]=0;
		ans[i]=0;
		for(j=0;j<n;j++){
			need[i][j]=max[i][j]-alloc[i][j];
		}
	}
	for(k=0;k<5 ;k++){
		for(i=0;i<5;i++){
			if (f[i]==0){
				int flag = 0;
				for(j=0;j<n;j++){
					if(need[i][j]>avail[j]){
						flag=1;
						break;
					}
				}
				if(flag==0){
					ans[o]=i;
					o++;
					for(s=0;s<n;s++){
						avail[s]=alloc[i][s]+avail[s];
					}
					f[i]=1;
					break; 
				}
			}
		}
	}
	if(o==m-1){
		for(i=0;i<m;i++){
			printf("%d ",ans[i]);
		}
	}
	else{
		printf("Unsafe State!!!");
	}
}
