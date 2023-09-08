#include <stdio.h>

// Maximum number of processes and resources
#define P 5
#define R 3

// Function to find the need of each process
void calculateNeed(int need[P][R], int max[P][R], int alloc[P][R])
{
	int i,j;
    for ( i = 0; i < P; i++)
        for ( j = 0; j < R; j++)
            need[i][j] = max[i][j] - alloc[i][j];
}

// Function to check if the requested resources can be granted
int isSafe(int processes[], int avail[], int max[][R], int alloc[][R])
{
    int need[P][R];
    calculateNeed(need, max, alloc);
    
    int finish[P] = {0};
    int safeSeq[P];
    int work[R];
    int i;
    for ( i = 0; i < R; i++)
        work[i] = avail[i];

    int count = 0;
    while (count < P)
    {
        int found = 0;
        int p;
        for ( p = 0; p < P; p++)
        {
            if (!finish[p])
            {
                int j;
                for (j = 0; j < R; j++)
                    if (need[p][j] > work[j])
                        break;
                
                if (j == R)
                {
                	int k;
                    for (k = 0; k < R; k++)
                        work[k] += alloc[p][k];
                    
                    safeSeq[count++] = p;
                    finish[p] = 1;
                    found = 1;
                }
            }
        }
        
        if (!found)
        {
            printf("System is not in safe state.\n");
            return 0;
        }
    }
    
    printf("System is in safe state. Safe sequence: ");
    for (i = 0; i < P - 1; i++)
        printf("P%d -> ", safeSeq[i]);
    printf("P%d\n", safeSeq[P - 1]);
    
    return 1;
}

int main()
{
    int processes[P] = {0, 1, 2, 3, 4};

    int available[R] = {3, 3, 2};

    int max[P][R] = {{7, 5, 3},
                     {3, 2, 2},
                     {9, 0, 2},
                     {2, 2, 2},
                     {4, 3, 3}};

    int allocation[P][R] = {{0, 1, 0},
                            {2, 0, 0},
                            {3, 0, 2},
                            {2, 1, 1},
                            {0, 0, 2}};

    isSafe(processes, available, max, allocation);

    return 0;
}

