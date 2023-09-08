//Program For Banker's Algorithm
#include<stdio.h>
int main(){
	int i,j,k,m,n; 
	printf("Enter The Number of Processes");
	scanf("%d",&m);
	printf("Enter the number of resources");
	scanf("%d",&n);
	int alloc[m][n];
	int max[m][n], need[m][n], avail[n];
	for(i=0;i<m;i++){
		printf("Enter the Allocated Resources for P%d\n",i);
		for(j=0;j<n;j++){
			scanf("%d",&alloc[i][j]);
		}
	}
	for(i=0;i<m;i++){
		printf("Enter the Maximum Resources for P%d\n",i);
		for(j=0;j<n;j++){
			scanf("%d",&max[i][j]);
		}
	}
	printf("Enter the available resources:\n");
	for(i=0;i<n;i++){
		scanf("%d",&avail[i]);
	}
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			need[i][j]=max[i][j]-alloc[i][j];
		}
	}
	printf("The Allocated Resorces are:\n");
	printf("   ");
	for(i=0;i<n;i++){
		printf("R%d ",i);
	}
	printf("\n");
	for(i=0;i<m;i++){
		printf("P%d ",i);
		for(j=0;j<n;j++){
			printf("%d  ",alloc[i][j]);
		}
		printf("\n");
	}
	printf("The Maximum Resorces are:\n");
	printf("   ");
	for(i=0;i<n;i++){
		printf("R%d ",i);
	}
	printf("\n");
	for(i=0;i<m;i++){
		printf("P%d ",i);
		for(j=0;j<n;j++){
			printf("%d  ",max[i][j]);
		}
		printf("\n");
	}
	printf("The Needed Resorces are:\n");
	printf("   ");
	for(i=0;i<n;i++){
		printf("R%d ",i);
	}
	printf("\n");
	for(i=0;i<m;i++){
		printf("P%d ",i);
		for(j=0;j<n;j++){
			printf("%d  ",need[i][j]);
		}
		printf("\n");
	}
	int y = 0;
    int ind = 0;
    int f[n],ans[n]; 
	for (k = 0; k < n; k++) { 
		f[k] = 0; 
	} 
    for (k = 0; k < m; k++) {
        for (i = 0; i < m; i++) {
            if (f[i] == 0) {
                int flag = 0;
                for (j = 0; j < n; j++) {
                    if (need[i][j] > avail[j]) {
                        flag = 1;
                        break;
                    }
                }

                if (flag == 0) {
                    // Process i can be executed, so mark it as completed and update available resources
                    ans[ind++] = i;
                    for (y = 0; y < n; y++) {
                        avail[y] += alloc[i][y];
                    }
                    f[i] = 1;
                }
            }
        }
    }

    if (ind != m) {
        // If the number of processes in the safe sequence is not equal to the total number of processes,
        // then the system is in an unsafe state.
        printf("The system is in an unsafe state.\n");
    } else {
        printf("The SAFE Sequence is as follows:\n");
        for (i = 0; i < m; i++) {
            printf(" P%d", ans[i]);
            if (i < m - 1) {
                printf(" ->");
            }
        }
        printf("\n");
    }
	for(i=0;i<m;i++){
		printf("%d",ans[i]);
	}
    return 0;
}
