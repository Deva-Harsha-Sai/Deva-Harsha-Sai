//Stack Using Arrays
#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
int a[5],top=-1;
void push(int d)
{
    if(top==4)
        {
        printf("\noverflow");
        return;
        }
    else{
        top++;
        a[top]=d;
    }    
}

void pop()
{
    if(top==-1)
    {
        printf("\nunder flow");
        return;
    }
    else{
        top--;
    }
}

void display()
{
    int i;
  if(top==-1)
  {
      printf("\nunderflow");
      return;
  }    
  else
  {
      for(i=0;i<=top;i++)
      {
      printf("%d ",a[i]);
          
    }
  }
}

void size()
{
    printf("\nThe size of the array is %d",top+1);
}

void empty()
{
    if(top==-1)
    {
        printf("\nstack is empty");
        return;
    }
    else{
        printf("\nstack is not empty");
    }
}

void main()
{
    int ch,d;
    while(1)
    {
    
    printf("\n1.push 2.pop 3.display 4.size 5.empty 6.Exit");
    printf("\nEnter your choice: ");
    scanf("%d",&ch);
    switch(ch)
    {
        case 1:printf("\nEnter the element:");
        scanf("%d",&d);
        push(d);
        display();
        break;
        case 2:pop();
        display();
        break;
        case 3:display();
        break;
        case 4:size();
        break;
        case 5:empty();
        break;
        case 6:{
			exit(0);
		}
    }
}


}
