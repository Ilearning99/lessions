#include<iostream>
#include<cstdio>
#include<cstring>
using namespace std;

char graph[10][10];
bool iscover[10];
int ans;

void gao(int id,int size,int count)
{
    int i;
    if(count==0)
    {
        ans++;
        return;
    }
    if(id==size)
        return;
    for(i=0;i<size;i++)
    {
        if(graph[id][i]=='#'&&iscover[i]==false)
        {
            iscover[i]=true;
            gao(id+1,size,count-1);
            iscover[i]=false;
        }
    }
    gao(id+1,size,count);
}


int main()
{
    int i,n,k;
    while(true)
    {
        scanf("%d%d",&n,&k);
        if(n==-1&&k==-1)
            break;
        for(i=0;i<n;i++)
            scanf("%s",graph+i);
        ans=0;
        memset(iscover,false,sizeof(iscover));
        gao(0,n,k);
        printf("%d\n",ans);
    }
    return 0;
}