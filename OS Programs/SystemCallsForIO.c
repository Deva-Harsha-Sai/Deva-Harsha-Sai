//System calls for file I/O
#include<stdio.h>
#include<fcntl.h>
int main()
{
	int n, i = 0;
	int f1, f2;
	char c, str[100];
	f1 = open("data",O_RDWR|O_CREAT|O_TRUNC);
	while((c=getchar())!= '\n')
	{
		str[i++]=c;
	}
	str[i]='\0';
	write(f1,str,i);
	close(f1);
	f2=open("data",O_RDONLY);
	read(f2,str,i);
	printf("\n%s\n",str);
	close(f2);
	return 0;
}
