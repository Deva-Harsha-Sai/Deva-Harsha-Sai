#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    char d[100];
    int c, op;
    DIR *e;
    struct dirent *sd;
    while(1)
    {
    	printf("**menu**\n1.Create DIR\n2.Remove DIR\n3.Read DIR\n4.Exit\nEnter Your Option: ");
    	scanf("%d", &op);

    	switch (op) {
        	case 1:
            	printf("Enter directory name: ");
            	scanf("%s", d);
            	c = mkdir(d);
            	if (c == -1)
                	printf("Directory is not created");
            	else
                	printf("Directory is created");
            	break;
        	case 2:
            	printf("Enter dir name: ");
            	scanf("%s", d);
            	c = rmdir(d);
            	if (c == -1)
                	printf("Dir is not removed");
            	else
                	printf("Dir is removed!!");
            	break;
        	case 3:
            	printf("Enter dir name to open: ");
            	scanf("%s", d);
            	e = opendir(d);
            	if (e == NULL)
                	printf("DIR Doesn't exist");
            	else {
                	printf("Dir found\n");
                	while ((sd = readdir(e)) != NULL)
                    	printf("%s\t", sd->d_name);
            	}
            	closedir(e);
            	break;
            case 4:
            	exit(0);
    	}
	}
    return 0;
}

